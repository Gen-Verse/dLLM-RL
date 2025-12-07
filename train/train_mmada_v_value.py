# Path: ./train/train_mmada_value.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging
import math
import shutil
import time
from pathlib import Path

from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from train.utils import get_config, flatten_omega_conf, AverageMeter
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from torch.utils.data import Dataset, DataLoader

logger = get_logger(__name__, log_level="INFO")

SOI_ID = 126084
EOI_ID = 126085
MMU_ID = 126089
IPAD_ID = 126093


class TrainDataset(Dataset):
    def __init__(self,
                 input_ids,        # (N_rows, L)
                 p_mask,           # (N_rows, L) bool
                 row_step_ids,     # (N_rows,)
                 seq_ids,          # (N_rows,)
                 rewards_raw       # length = B, each item: scalar or 1D list/array/tensor
                 ):
        self.input_ids = input_ids
        self.p_mask = p_mask.to(torch.bool)
        self.row_step_ids = torch.as_tensor(row_step_ids, dtype=torch.long)
        self.seq_ids = torch.as_tensor(seq_ids, dtype=torch.long)

        self.raw_rewards = list(rewards_raw)
        B = int(self.seq_ids.max().item()) + 1
        if len(self.raw_rewards) != B:
            raise ValueError(
                f"len(rewards_raw)={len(self.raw_rewards)} != num sequences B={B}"
            )

        all_scalar = True
        scalar_vals = []
        for r in self.raw_rewards:
            if isinstance(r, (int, float)):
                scalar_vals.append(float(r))
            elif torch.is_tensor(r) and r.ndim == 0:
                scalar_vals.append(float(r.item()))
            else:
                all_scalar = False
                break

        self.per_seq_reward = (
            torch.tensor(scalar_vals, dtype=torch.float32)
            if all_scalar else None
        )

        N, L = input_ids.shape
        self.old_values = torch.zeros((N, L), dtype=torch.float32)
        self.Return     = torch.zeros((N, L), dtype=torch.float32)
        self.adv        = torch.zeros((N, L), dtype=torch.float32)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return (
            idx,
            self.input_ids[idx],
            self.p_mask[idx],
            self.Return[idx],  # placeholder until computed
        )


# for shrink
def collapse_k_unique(lst, k: int):
    if k <= 0:
        raise ValueError("k must be > 0")
    uniq = sorted(set(lst))
    mapping = {}
    n = len(uniq)
    for idx, val in enumerate(uniq):
        group = idx // k
        end_idx = min((group + 1) * k - 1, n - 1)
        rep = uniq[end_idx]
        mapping[val] = rep
    return [mapping[x] for x in lst]


def left_pad_to_len(x: torch.Tensor, L: int, pad_id: int):
    out = torch.full((L,), pad_id, dtype=x.dtype, device=x.device)
    out[-x.numel():] = x
    return out


def build_rows_text(config, tokenizer, uni_prompting, prompts, responses, step_maps, rewards):
    # Build fixed-length sequences via UniversalPrompting
    input_ids_lm, labels_lm, start_pos, drop_num = uni_prompting((prompts, responses))
    device = input_ids_lm.device
    mask_id = tokenizer.encode('<|mdm_mask|>')[0]
    pad_id = tokenizer.encode('<|endoftext|>')[0]

    B, L = input_ids_lm.shape
    L0 = start_pos
    L1 = L - L0
    post_num = config.training.post_num

    rows_inputs = []
    rows_pmask = []
    rows_labels = []
    rows_seq_ids = []
    rows_step_ids = []

    for b in range(B):
        base_ids = input_ids_lm[b].to(device)
        # shrink full response step map
        sm = step_maps[b]
        if len(sm) > L1:
            sm = sm[:L1]
        else:
            sm = sm + [max(sm) + 1] * (L1 - len(sm))
        sm = collapse_k_unique(sm, config.training.shrink)
        order_full = torch.full((L,), -1, dtype=torch.long, device=device)
        order_full[L0:] = torch.as_tensor(sm, dtype=torch.long, device=device)

        if post_num is not None:
            pad_mask_b = (base_ids == pad_id)
            pad_mask_b[:L0] = False
            keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= post_num)
            tail_pad_b = pad_mask_b & ~keep_first_pad_b
        else:
            tail_pad_b = torch.zeros(L, dtype=torch.bool, device=device)

        uniq_steps = torch.unique(order_full[L0:], sorted=True)
        has_any = False
        for sv in uniq_steps.tolist():
            if sv < 0:
                continue
            pmask_this = (order_full == sv) & ~tail_pad_b
            if not pmask_this.any():
                continue
            noisy = base_ids.clone()
            noisy[order_full >= sv] = mask_id
            rows_inputs.append(noisy)
            rows_pmask.append(pmask_this)
            rows_labels.append(base_ids)          # 新增：每行的 labels = 原始序列
            rows_seq_ids.append(b)
            rows_step_ids.append(int(sv))
            has_any = True

        if not has_any:
            # ensure at least one position
            valid = (~tail_pad_b).clone()
            valid[:L0] = False
            if valid.any():
                first_idx = torch.nonzero(valid, as_tuple=False)[0, 0]
                noisy = base_ids.clone()
                noisy[first_idx] = mask_id
                pmask_this = torch.zeros(L, dtype=torch.bool, device=device)
                pmask_this[first_idx] = True
                rows_inputs.append(noisy)
                rows_pmask.append(pmask_this)
                rows_labels.append(base_ids)          
                rows_seq_ids.append(b)
                rows_step_ids.append(0)

    if not rows_inputs:
        return None

    input_rows   = torch.stack(rows_inputs, dim=0)
    pmask_rows   = torch.stack(rows_pmask, dim=0)
    labels_rows  = torch.stack(rows_labels, dim=0)          
    seq_ids_rows = torch.as_tensor(rows_seq_ids, dtype=torch.long)
    step_ids_rows= torch.as_tensor(rows_step_ids, dtype=torch.long)

    return input_rows, pmask_rows, labels_rows, step_ids_rows, seq_ids_rows, rewards, start_pos, drop_num



def build_rows_mmu(config, tokenizer, prompts, responses, step_maps, rewards, image_token_ids_list):
    # Build full MMU input per sample, then left-pad to batch max, then construct rows per step.
    device = torch.device("cpu")
    mask_id = tokenizer.encode('<|mdm_mask|>')[0]
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")

    # 1) Build base sequences per sample
    base_list = []
    start_pos_raw = []
    for p_str, r_str, img_ids in zip(prompts, responses, image_token_ids_list):
        chat_ids = tokenizer([p_str], add_special_tokens=False)["input_ids"][0]
        chat_t = torch.tensor(chat_ids, dtype=torch.long, device=device)
        img_t = torch.tensor(img_ids, dtype=torch.long, device=device)
        resp_ids = tokenizer([r_str], add_special_tokens=False)["input_ids"][0]
        resp_t = torch.tensor(resp_ids, dtype=torch.long, device=device)
        mmu = torch.tensor([MMU_ID, SOI_ID], dtype=torch.long, device=device)
        eoi = torch.tensor([EOI_ID], dtype=torch.long, device=device)
        full = torch.cat([mmu, img_t, eoi, chat_t, resp_t], dim=0)  # [T]
        # assistant start: last <|end_header_id|> + 1 after the image and within chat
        pos = (chat_t == end_header_id).nonzero(as_tuple=False)
        if pos.numel() == 0:
            s_pos = 2 + img_t.numel() + 1 + chat_t.numel()
        else:
            last = int(pos[-1].item())
            s_pos = 2 + img_t.numel() + 1 + (last + 1)
        base_list.append(full)
        start_pos_raw.append(s_pos)

    # 2) Left-pad base to batch-max length
    max_len = max(t.numel() for t in base_list)
    base_pad = torch.stack([left_pad_to_len(t, max_len, pad_id) for t in base_list], dim=0)  # (B, N)
    B, N = base_pad.shape
    pad_shifts = [max_len - t.numel() for t in base_list]
    start_pos_list = [int(s + sh) for s, sh in zip(start_pos_raw, pad_shifts)]

    # 3) For each sample, build rows by step
    rows_inputs = []
    rows_pmask = []
    rows_labels = []
    rows_seq_ids = []
    rows_step_ids = []

    post_num = config.training.post_num

    for b in range(B):
        base_ids = base_pad[b]
        L0_b = start_pos_list[b]
        L1_b = N - L0_b  # response spans until end
        sm = step_maps[b]
        if len(sm) > L1_b:
            sm = sm[:L1_b]
        else:
            sm = sm + [max(sm) + 1] * (L1_b - len(sm))
        sm = collapse_k_unique(sm, config.training.shrink)

        order_full = torch.full((N,), -1, dtype=torch.long)
        order_full[L0_b:L0_b + L1_b] = torch.as_tensor(sm, dtype=torch.long)

        if post_num is not None:
            pad_mask_b = (base_ids == pad_id)
            pad_mask_b[:L0_b] = False
            keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= post_num)
            tail_pad_b = pad_mask_b & ~keep_first_pad_b
        else:
            tail_pad_b = torch.zeros(N, dtype=torch.bool)

        uniq_steps = torch.unique(order_full[L0_b:], sorted=True)
        has_any = False
        for sv in uniq_steps.tolist():
            if sv < 0:
                continue
            pmask_this = (order_full == sv) & ~tail_pad_b
            if not pmask_this.any():
                continue
            noisy = base_ids.clone()
            noisy[order_full >= sv] = mask_id
            rows_inputs.append(noisy)
            rows_pmask.append(pmask_this)
            rows_labels.append(base_ids)        
            rows_seq_ids.append(b)
            rows_step_ids.append(int(sv))
            has_any = True

        if not has_any:
            valid = (~tail_pad_b).clone()
            valid[:L0_b] = False
            if valid.any():
                first_idx = torch.nonzero(valid, as_tuple=False)[0, 0]
                noisy = base_ids.clone()
                noisy[first_idx] = mask_id
                pmask_this = torch.zeros(N, dtype=torch.bool)
                pmask_this[first_idx] = True
                rows_inputs.append(noisy)
                rows_pmask.append(pmask_this)
                rows_labels.append(base_ids)        
                rows_seq_ids.append(b)
                rows_step_ids.append(0)

    if not rows_inputs:
        return None

    input_rows   = torch.stack(rows_inputs, dim=0)
    pmask_rows   = torch.stack(rows_pmask, dim=0)
    labels_rows  = torch.stack(rows_labels, dim=0)
    seq_ids_rows = torch.as_tensor(rows_seq_ids, dtype=torch.long)
    step_ids_rows= torch.as_tensor(rows_step_ids, dtype=torch.long)

    return input_rows, pmask_rows, labels_rows, step_ids_rows, seq_ids_rows, rewards, start_pos_list[0], 0



def save_checkpoint(model, tokenizer, config, accelerator, name):
    from pathlib import Path
    import time, json, shutil, os, glob, importlib, inspect

    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]) if "-" in p.name else -1,
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        save_dir = save_base / name
        model_to_save.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_dir))

        def _copy_dynamic_modules(dst_dir, model_obj, tok_obj):
            copied = 0
            modules = set()
            for obj in [model_obj, getattr(model_obj, "config", None), tok_obj]:
                if obj is None:
                    continue
                modname = getattr(obj.__class__, "__module__", None)
                if modname:
                    modules.add(modname)
            for modname in modules:
                try:
                    mod = importlib.import_module(modname)
                    src_file = inspect.getsourcefile(mod)
                    if not src_file or not os.path.exists(src_file):
                        continue
                    base_dir = os.path.dirname(src_file)
                    for pattern in ("modeling_*.py", "configuration_*.py", "tokenization_*.py", "processing_*.py"):
                        for fn in glob.glob(os.path.join(base_dir, pattern)):
                            dst = os.path.join(dst_dir, os.path.basename(fn))
                            if os.path.exists(dst):
                                continue
                            shutil.copy2(fn, dst)
                            copied += 1
                except Exception as e:
                    logger.warning(f"Skip copying from module {modname}: {e}")

        _copy_dynamic_modules(str(save_dir), model_to_save, tokenizer)

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_dir}")


def main():
    config = get_config()

    project_name = config.experiment.project
    pretrained_model = "./" + project_name + "/ckpt/" + config.model.optimized_value_name

    # value model class with value_head
    from train.init_mmada_value_model import _get_value_model
    from models import MMadaModelLM
    value_model_class = _get_value_model(MMadaModelLM, "value_head")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    value_model = value_model_class.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=None,
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    # logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.project,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    # optimizer
    optimizer_config = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in value_model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in value_model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if config.optimizer.name != "adamw":
        raise ValueError(f"Optimizer {config.optimizer.name} not supported")
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=optimizer_config.value_learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

    # if config.training.gradient_checkpointing_enable:
    #     value_model.gradient_checkpointing_enable()
    #     if hasattr(value_model, "config"):
    #         value_model.config.use_cache = False
    # else:
    value_model = value_model.to(accelerator.device)

    # load rl data
    with open("./" + project_name + "/temp_data/" + config.dataset.optimization_data + f"-step{config.experiment.current_epoch}.json", 'r') as f:
        dataset_load = json.load(f)
    if len(dataset_load) == 0:
        logger.warning("No optimization data found. Exiting value training early.")
        accelerator.end_training()
        return

    prompts = [x["prompt"] for x in dataset_load]
    responses = [x["response"] for x in dataset_load]
    rewards = [x["reward"] for x in dataset_load]
    # step maps: ensure existence
    step_maps = []
    for x in dataset_load:
        sm = x.get("step_map", None)
        if sm is None or len(sm) == 0:
            # fallback to linear steps
            # need response length; approximate by tokenized response length
            resp_len = len(tokenizer([x["response"]], add_special_tokens=False)["input_ids"][0])
            sm = list(range(resp_len))
        step_maps.append(sm)

    uni_prompting = UniversalPrompting(tokenizer,
                                       max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)

    # build rows for text or mmu
    if config.dataset.data_type == "mmu":
        image_tok_list = [x.get("image_token_ids", None) for x in dataset_load]
        assert all(v is not None for v in image_tok_list), "Missing image_token_ids in rl_data for MMU."
        rows_pack = build_rows_mmu(config, tokenizer, prompts, responses, step_maps, rewards, image_tok_list)
    else:
        rows_pack = build_rows_text(config, tokenizer, uni_prompting, prompts, responses, step_maps, rewards)

    if rows_pack is None:
        logger.warning("Failed to build any training rows. Exiting.")
        accelerator.end_training()
        return

    #input_rows, pmask_rows, labels_rows, step_ids_rows, seq_ids_rows, per_seq_reward, start_pos, drop_num = rows_pack
    input_rows, pmask_rows, labels_rows, step_ids_rows, seq_ids_rows, rewards_raw, start_pos, drop_num = rows_pack

    dataset_lm = TrainDataset(
        input_rows, pmask_rows, step_ids_rows, seq_ids_rows, rewards_raw, #per_seq_reward
    )

    # dataloader + lr scheduler
    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = max(1, math.ceil(len(dataset_lm) / total_batch_size_lm))
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    config.lr_scheduler.params.learning_rate = config.optimizer.params.value_learning_rate
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    def simple_collate(batch):
        idx, input_ids, p_mask, Return = zip(*batch)
        return {
            "ids": torch.tensor(idx),
            "input_ids": torch.stack(input_ids),
            "p_mask": torch.stack(p_mask),
            "Return": torch.stack(Return),
        }

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0
    )

    value_model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        value_model, optimizer, lr_scheduler, train_dataloader_lm
    )

    @torch.no_grad()
    def compute_old_value_parallel(accelerator, dataset, dataloader):
        value_model.eval()
        for batch in dataloader:
            ids = batch["ids"]  # (b,)
            input_ids = batch["input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)

            values = value_model(input_ids=input_ids)  # (B, T)
            values = torch.where(p_mask, values, torch.zeros_like(values))

            if accelerator.num_processes > 1:
                ids_dev = ids.to(accelerator.device)
                ids_pad = accelerator.pad_across_processes(ids_dev, dim=0, pad_index=-1)
                values_pad = accelerator.pad_across_processes(values, dim=0)

                ids_all = accelerator.gather(ids_pad)
                values_all = accelerator.gather(values_pad)

                valid = ids_all.ne(-1)
                idx_cpu = ids_all[valid].long().cpu()
                vals_cpu = values_all[valid].float().cpu()
                dataset.old_values[idx_cpu] = vals_cpu
            else:
                dataset.old_values[ids] = values.float().cpu()

        accelerator.wait_for_everyone()
        value_model.train()

    logger.info("***** Running old value inference *****")
    compute_old_value_parallel(accelerator, dataset_lm, train_dataloader_lm)

    def compute_returns_and_advantages_rows(dataset: TrainDataset, gamma: float, lam: float, atol: float = 1e-5):

        def build_step_rewards(raw_r, uniq_steps):
            S = len(uniq_steps)
            r_star = torch.zeros(S, dtype=torch.float32)

            # case 1: scalar -> reward only on last step
            if isinstance(raw_r, (int, float)) or (torch.is_tensor(raw_r) and raw_r.ndim == 0):
                r_star[-1] = float(raw_r)
                return r_star

            if torch.is_tensor(raw_r):
                if raw_r.ndim != 1:
                    raise ValueError(f"Vector reward tensor must be 1D, got shape {tuple(raw_r.shape)}")
                raw_r = raw_r.tolist()

            if isinstance(raw_r, (list, tuple)):
                # case 2a: len == #steps, interpret as r_star directly (按 uniq_steps 排序顺序)
                if len(raw_r) == S:
                    return torch.as_tensor(raw_r, dtype=torch.float32)

                # case 2b: len >= max_step_id+1, interpret as indexed by step_id
                max_sid = max(uniq_steps)
                if len(raw_r) >= max_sid + 1:
                    for i, sid in enumerate(uniq_steps):
                        r_star[i] = float(raw_r[sid])
                    return r_star

                raise ValueError(
                    f"Cannot map reward of length {len(raw_r)} to step ids {uniq_steps}"
                )

            raise TypeError(f"Unsupported reward type: {type(raw_r)}")

        N_rows, L = dataset.p_mask.shape
        B = int(dataset.seq_ids.max().item()) + 1

        # group rows by original sequence
        rows_by_seq = [[] for _ in range(B)]
        for r in range(N_rows):
            s = int(dataset.seq_ids[r].item())
            rows_by_seq[s].append(r)

        Return_mat = dataset.Return.clone()
        adv_mat    = dataset.adv.clone()
        old_vals   = dataset.old_values.clone()

        for s in range(B):
            rows = rows_by_seq[s]
            if not rows:
                continue

            # sort rows by step id
            rows_sorted = sorted(rows, key=lambda r: int(dataset.row_step_ids[r].item()))
            step_ids_sorted = [int(dataset.row_step_ids[r].item()) for r in rows_sorted]
            uniq_steps = sorted(set(step_ids_sorted))
            S = len(uniq_steps)
            sid_to_idx = {sid: i for i, sid in enumerate(uniq_steps)}

            V_star = torch.zeros(S, dtype=torch.float32)
            for r in rows_sorted:
                sid = int(dataset.row_step_ids[r].item())
                i = sid_to_idx[sid]
                pm = dataset.p_mask[r]
                ov = old_vals[r]
                V_star[i] = ov[pm].mean() if pm.any() else 0.0

            # generalized step-level rewards
            raw_r = dataset.raw_rewards[s]
            r_star = build_step_rewards(raw_r, uniq_steps)

            # backward returns on step-level
            R_star = torch.zeros(S, dtype=torch.float32)
            for i in range(S - 1, -1, -1):
                R_star[i] = r_star[i] + (gamma * R_star[i + 1] if i + 1 < S else 0.0)

            # TD residual δ_t = r_t - V_t + γ V_{t+1}
            delta_star = torch.zeros(S, dtype=torch.float32)
            for i in range(S):
                v_next = V_star[i + 1] if i + 1 < S else 0.0
                delta_star[i] = r_star[i] - V_star[i] + gamma * v_next

            # step-level GAE
            A_star = torch.zeros(S, dtype=torch.float32)
            for i in range(S - 1, -1, -1):
                A_star[i] = delta_star[i] + (gamma * lam * A_star[i + 1] if i + 1 < S else 0.0)

            # map back to token rows
            for r in rows_sorted:
                sid = int(dataset.row_step_ids[r].item())
                i = sid_to_idx[sid]
                pm = dataset.p_mask[r]
                ov = old_vals[r]

                r_j    = r_star[i]
                R_next = R_star[i + 1] if i + 1 < S else 0.0
                V_next = V_star[i + 1] if i + 1 < S else 0.0
                A_next = A_star[i + 1] if i + 1 < S else 0.0

                Return_mat[r][pm] = r_j + gamma * R_next
                adv_vals = (r_j - ov[pm]) + gamma * V_next + gamma * lam * A_next
                adv_mat[r][pm] = adv_vals

            if dataset.per_seq_reward is not None:
                scalar_r = float(dataset.per_seq_reward[s].item())

                if abs(gamma - 1.0) < 1e-8 and abs(lam - 1.0) < 1e-8:
                    for r in rows_sorted:
                        pm = dataset.p_mask[r]
                        if not pm.any():
                            continue
                        R_row = Return_mat[r][pm]
                        V_row = old_vals[r][pm]
                        A_row = adv_mat[r][pm]
                        assert torch.allclose(R_row, torch.full_like(R_row, scalar_r), atol=atol), \
                            f"gamma=lambda=1 check failed (R) at seq {s}, row {r}"
                        assert torch.allclose(A_row, R_row - V_row, atol=atol), \
                            f"gamma=lambda=1 check failed (A=R-V) at seq {s}, row {r}"

                if abs(gamma - 1.0) < 1e-8 and abs(lam - 0.0) < 1e-8:
                    for r in rows_sorted:
                        sid = int(dataset.row_step_ids[r].item())
                        i = sid_to_idx[sid]
                        V_next = V_star[i + 1] if i + 1 < S else 0.0
                        r_j = scalar_r if i == (S - 1) else 0.0
                        pm = dataset.p_mask[r]
                        if not pm.any():
                            continue
                        V_row = old_vals[r][pm]
                        A_expected = (r_j - V_row) + V_next
                        assert torch.allclose(adv_mat[r][pm], A_expected, atol=atol), \
                            f"gamma=1, lambda=0 check failed (A=r-V+V_next*) at seq {s}, row {r}"

        dataset.Return = Return_mat
        dataset.adv    = adv_mat


    gam = config.training.get("gam", 1.0)
    lam = config.training.get("lam", 1.0)
    logger.info("***** Calculate Returns and Advantages *****")
    compute_returns_and_advantages_rows(dataset_lm, gam, lam, atol=1e-5)

    def save_dataset_tensors(dataset_lm, labels_rows, save_dir, name, accelerator, *, start_pos: int, drop_num: int):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "extended_input_ids": dataset_lm.input_ids,  # (N_rows, L)
            "p_mask":            dataset_lm.p_mask,      # (N_rows, L) bool
            "labels":            labels_rows,            # (N_rows, L)
            "adv":               dataset_lm.adv,         # (N_rows, L) float
            "meta": {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "start_pos": int(start_pos),
                "drop_num":  int(drop_num),
            },
        }
        if accelerator.is_main_process:
            torch.save(payload, save_dir / f"{name}.pt")

    save_dataset_tensors(
        dataset_lm,
        labels_rows,
        save_dir=Path(config.experiment.project) / "temp_data",
        name=f"{config.dataset.optimization_data}-step{config.experiment.current_epoch}",
        accelerator=accelerator,
        start_pos=start_pos,
        drop_num=drop_num
    )


    # ------------------ Debug dump for first two sequences ------------------
    if accelerator.is_main_process:
        debug_dir = Path(config.experiment.project) / "temp_data"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"debug_mmada_value.txt"
        with open(debug_path, "w", encoding="utf-8") as df:
            df.write(f"DEBUG MMADA VALUE TRAINING\n")
            df.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            df.write(f"Project: {config.experiment.project}\n")
            df.write(f"Epoch: {config.experiment.current_epoch}\n\n")

            # model / env info
            try:
                device_info = str(accelerator.device)
            except Exception:
                device_info = "unknown"
            n_params = sum(p.numel() for p in value_model.parameters())
            df.write(f"Device: {device_info}\n")
            df.write(f"Value model params: {n_params}\n")
            df.write(f"Dataset rows: {len(dataset_lm)}   Row shape: {dataset_lm.input_ids.shape}\n")
            df.write(f"start_pos (meta): {start_pos}   drop_num: {drop_num}\n\n")

            # shapes and dtypes
            df.write("Tensors overview (show shapes and dtypes):\n")
            df.write(f" input_rows.shape = {tuple(dataset_lm.input_ids.shape)} dtype={dataset_lm.input_ids.dtype}\n")
            df.write(f" p_mask.shape = {tuple(dataset_lm.p_mask.shape)} dtype={dataset_lm.p_mask.dtype}\n")
            df.write(f" old_values.shape = {tuple(dataset_lm.old_values.shape)} dtype={dataset_lm.old_values.dtype}\n")
            df.write(f" Return.shape = {tuple(dataset_lm.Return.shape)} dtype={dataset_lm.Return.dtype}\n")
            df.write(f" adv.shape = {tuple(dataset_lm.adv.shape)} dtype={dataset_lm.adv.dtype}\n\n")

            # For first two original sequences, dump detailed info
            max_seq_to_dump = 2
            for s in range(max_seq_to_dump):
                df.write(f"--- Sequence {s} ---\n")
                seq_obj = dataset_load[s]
                df.write(f"prompt (raw): {seq_obj.get('prompt')}\n")
                df.write(f"response (raw): {seq_obj.get('response')}\n")
                df.write(f"ground_truth_answer: {seq_obj.get('ground_truth_answer')}\n")
                if 'image' in seq_obj:
                    df.write(f"image path: {seq_obj.get('image')}\n")
                if 'image_token_ids' in seq_obj:
                    df.write(f"image_token_ids (len): {len(seq_obj.get('image_token_ids'))}\n")

                # step map (original)
                sm = seq_obj.get("step_map", None)
                df.write(f"original step_map (len={len(sm) if sm is not None else 'N/A'}): {sm}\n")

                # rows that correspond to this sequence
                rows_idx = (dataset_lm.seq_ids == s).nonzero(as_tuple=True)[0].tolist()
                df.write(f"rows indices for this seq ({len(rows_idx)}): {rows_idx}\n")
                for rid in rows_idx:
                    df.write(f"  row {rid}:\n")
                    row_input_ids = dataset_lm.input_ids[rid].tolist()
                    # decoded text (try safe decode)
                    try:
                        decoded = tokenizer.decode(row_input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    except Exception:
                        decoded = "<decode error>"
                    df.write(f"    input_ids (len={len(row_input_ids)}): {row_input_ids}\n")
                    df.write(f"    decoded: {decoded}\n")
                    pmask_list = dataset_lm.p_mask[rid].int().tolist()
                    df.write(f"    p_mask (len={len(pmask_list)}): {pmask_list}\n")
                    old_vals_row = dataset_lm.old_values[rid].tolist()
                    df.write(f"    old_values (masked positions nonzero): {[old_vals_row[i] for i,v in enumerate(pmask_list) if v]}\n")
                    ret_row = dataset_lm.Return[rid].tolist()
                    df.write(f"    Return: {[ret_row[i] for i,v in enumerate(pmask_list) if v]}\n")
                    adv_row = dataset_lm.adv[rid].tolist()
                    df.write(f"    adv: {[adv_row[i] for i,v in enumerate(pmask_list) if v]}\n")
                    df.write(f"    step_id_for_row: {int(dataset_lm.row_step_ids[rid].item())}\n")
                    df.write("\n")

                df.write("\n")

            # also dump first 10 rows global overview
            df.write("First 10 rows overview:\n")
            n_show = min(10, len(dataset_lm.input_ids))
            for i in range(n_show):
                ids = dataset_lm.input_ids[i].tolist()
                pm = dataset_lm.p_mask[i].int().tolist()
                df.write(f"row {i}: seq_id={int(dataset_lm.seq_ids[i].item())} step_id={int(dataset_lm.row_step_ids[i].item())} p_mask_sum={int(sum(pm))} len_input={len(ids)}\n")
            df.write("\n")
            df.write("End of debug dump.\n")


    # Skip actual training if configured to not train this round
    if config.experiment.current_epoch % config.experiment.train_value_every != 0:
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return

    # train
    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    logger.info("***** Running value training *****")
    logger.info(f"  Num samples (rows) = {len(dataset_lm)}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    def forward_process(input_ids, p_mask, Return, old_values):
        values = value_model(input_ids=input_ids)  # (B, T)
        values = torch.where(p_mask, values, torch.zeros_like(values))

        v_clipped = old_values + (values - old_values).clamp(-config.training.eps, config.training.eps)
        loss_unclipped = (values - Return) ** 2
        loss_clipped = (v_clipped - Return) ** 2
        loss_tok = 0.5 * torch.maximum(loss_unclipped, loss_clipped) * p_mask

        # average over masked tokens per row, then mean over batch
        num_mask = p_mask.sum(dim=1).clamp(min=1)
        loss = (loss_tok.sum(dim=1) / num_mask).mean()
        return loss

    from tqdm.auto import tqdm
    loss_list = []
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(num_train_epochs):
        value_model.train()
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True
        )
        for step, batch in enumerate(progress_bar, start=1):
            data_time_m.update(time.time() - end)

            input_ids = batch["input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            Return = dataset_lm.Return[batch["ids"].cpu()].to(accelerator.device)
            old_vals = dataset_lm.old_values[batch["ids"].cpu()].to(accelerator.device)

            loss = forward_process(input_ids, p_mask, Return, old_vals)
            loss = loss / accelerator.gradient_accumulation_steps

            if step < 10:
                print(loss)

            accelerator.backward(loss)

            if (step % accelerator.gradient_accumulation_steps) == 0:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(value_model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            loss_list.append(loss.detach().float().cpu().item())

        # handle remaining gradients if any (when num steps not divisible by grad accumulation)
        if (step % accelerator.gradient_accumulation_steps) != 0:
            if config.training.max_grad_norm is not None:
                accelerator.clip_grad_norm_(value_model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            try:
                del input_ids, p_mask, Return, old_vals
            except Exception:
                pass
            torch.cuda.empty_cache()


    accelerator.wait_for_everyone()

    # save checkpoint
    save_checkpoint(value_model, tokenizer, config, accelerator, config.model.optimized_value_name)
    if config.experiment.current_epoch % config.experiment.save_every == 0:
        save_checkpoint(value_model, tokenizer, config, accelerator, f"epoch-{config.experiment.current_epoch}-value")

    accelerator.end_training()

    # write results
    from termcolor import cprint
    if accelerator.is_main_process:
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + config.dataset.train_dataset

        def _mean(x):
            return float(sum(x) / max(1, len(x)))

        temp_len = 50
        first = loss_list[:temp_len]
        last = loss_list[-temp_len:] if len(loss_list) >= temp_len else loss_list

        first_few_avg_loss = _mean(first)
        last_few_avg_loss = _mean(last)
        avg_loss = _mean(loss_list)

        output_text = (
            f"train step: {config.experiment.current_epoch}  "
            f"first_few_avg_loss: {first_few_avg_loss:.6f}  "
            f"last_few_avg_loss: {last_few_avg_loss:.6f}  "
            f"avg_loss: {avg_loss:.6f}  "
        )

        results_dir = Path(".") / project_name / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        outputs_result_name = results_dir / f"results-{outputs_name}.txt"

        cprint("\n\n" + output_text, color="green")
        with open(outputs_result_name, "a", encoding="utf-8", buffering=1) as f:
            f.write(output_text + "\n")


if __name__ == "__main__":
    main()
