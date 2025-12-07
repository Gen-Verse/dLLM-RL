# Path: ./train/train_lladav_value.py
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
from typing import List

from omegaconf import OmegaConf
import wandb
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from train.utils import get_config, flatten_omega_conf, AverageMeter
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from torch.utils.data import Dataset, DataLoader

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates

from train.init_lladav_value_model import _get_value_model
from models import LlavaLLaDAModelLM

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__, log_level="INFO")

MASK_ID_LLADAV = 126336 


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


def left_pad_1d(x: torch.Tensor, L: int, pad_val):
    out = torch.full((L,), pad_val, dtype=x.dtype, device=x.device)
    out[-x.numel():] = x
    return out


def left_pad_2d(x: torch.Tensor, L: int, pad_val: float = 0.0):
    # x: (T, D) -> (L, D)
    if x.size(0) == L:
        return x
    out = torch.full((L, x.size(1)), pad_val, dtype=x.dtype, device=x.device)
    out[-x.size(0):] = x
    return out


def build_llava_conv_prompt(question: str, response: str) -> str:
    import copy
    conv = copy.deepcopy(conv_templates["llava_llada"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], response)
    return conv.get_prompt().removesuffix("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


@torch.no_grad()
def build_rows_lladav(
    config,
    tokenizer,
    value_model,
    image_processor,
    prompts: List[str],
    responses: List[str],
    step_maps: List[List[int]],
    rewards: List[float],
    image_abs_list: List[str | None],
):

    rows_pmask: List[torch.Tensor] = []
    rows_labels: List[torch.Tensor] = []
    rows_seqids: List[int] = []
    rows_stepids: List[int] = []
    rows_attn:   List[torch.Tensor] = []

    row_mask_pos_list: List[torch.Tensor] = []
    row_to_seq_map:    List[int] = []
    clean_embeds_per_seq: List[torch.Tensor] = []

    dev = next(value_model.parameters()).device
    mask_embed = value_model.get_input_embeddings()(
        torch.tensor([MASK_ID_LLADAV], device=dev)
    ).squeeze(0).to(torch.bfloat16).detach().cpu()

    end_header_token = "<|end_header_id|>\n\n"
    end_header_ids = tokenizer.encode(end_header_token, add_special_tokens=False)

    for b in range(len(prompts)):
        prompt_text = prompts[b]
        response_text = responses[b]
        step_map_full = step_maps[b]
        img_path = image_abs_list[b]

        conv_text = build_llava_conv_prompt(prompt_text, response_text)
        ids_text = tokenizer_image_token(conv_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").squeeze(0).to(torch.long).to(dev)

        start_pos = 0
        if len(end_header_ids) > 0 and ids_text.numel() >= len(end_header_ids):
            ids_list = ids_text.tolist()
            for i in range(len(ids_list) - len(end_header_ids), -1, -1):
                if ids_list[i:i + len(end_header_ids)] == end_header_ids:
                    start_pos = i + len(end_header_ids)
                    break
        labels_text = ids_text.clone()
        labels_text[:start_pos] = -100

        if img_path is not None:
            from PIL import Image
            image = Image.open(img_path).convert("RGB")
            image_list = process_images([image], image_processor, value_model.config)
            image_list = [_im.to(dtype=next(value_model.parameters()).dtype, device=dev) for _im in image_list]
            image_sizes = [image.size]
        else:
            image_list, image_sizes = None, None

        (_in_ids, _pos_ids, _attn_mask, _pkv, clean_embeds, new_labels) = value_model.prepare_inputs_labels_for_multimodal(
            ids_text.unsqueeze(0), None, None, None, labels_text.unsqueeze(0), image_list, ["image"], image_sizes=image_sizes
        )
        clean_embeds = clean_embeds.squeeze(0).to(torch.bfloat16).detach().cpu()  # (L, D)
        new_labels = new_labels.squeeze(0).to(torch.long).detach().cpu()          # (L,)
        L = int(clean_embeds.size(0))

        eot_id = 126348
        eot_indices = (new_labels == eot_id).nonzero(as_tuple=False)
        if eot_indices.numel() > 0:
            last_eot_pos = eot_indices[-1, 0].item()
            new_labels[last_eot_pos:] = -100
        asst_mask = (new_labels != -100)
        asst_idx = asst_mask.nonzero(as_tuple=False).squeeze(-1)  # (M,)
        M = int(asst_idx.numel())

        if len(step_map_full) < M:
            order = torch.as_tensor(
                collapse_k_unique(step_map_full + [max(step_map_full) + 1] * (M - len(step_map_full)), config.training.shrink),
                dtype=torch.long
            )
        else:
            order = torch.as_tensor(collapse_k_unique(step_map_full[:M], config.training.shrink), dtype=torch.long)

        order_full = torch.full((L,), -1, dtype=torch.long)
        order_full[asst_idx] = order

        uniq_steps = torch.unique(order_full[asst_idx], sorted=True)

        if config.training.post_num is not None:
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            pad_mask_b = (new_labels == pad_token_id)
            pad_mask_b[:start_pos] = False
            keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
            tail_pad_b = pad_mask_b & ~keep_first_pad_b
        else:
            tail_pad_b = torch.zeros(L, dtype=torch.bool)

        clean_embeds_per_seq.append(clean_embeds)

        has_any = False
        for sv in uniq_steps.tolist():
            if sv < 0:
                continue
            pmask_this = (order_full == sv) & ~tail_pad_b
            if not pmask_this.any():
                continue

            mask_pos = (order_full >= sv)

            row_mask_pos_list.append(mask_pos.detach().cpu())
            rows_pmask.append(pmask_this.detach().cpu())
            rows_labels.append(new_labels.detach().cpu())
            rows_seqids.append(b)
            rows_stepids.append(int(sv))
            rows_attn.append(torch.ones(L, dtype=torch.bool))
            row_to_seq_map.append(b)
            has_any = True

        if not has_any:
            valid = torch.ones(L, dtype=torch.bool)
            valid[:start_pos] = False
            if valid.any():
                first_idx = int(valid.nonzero(as_tuple=False)[0, 0])
                pmask_this = torch.zeros(L, dtype=torch.bool)
                pmask_this[first_idx] = True
                mask_pos = torch.zeros(L, dtype=torch.bool)
                mask_pos[first_idx] = True

                row_mask_pos_list.append(mask_pos.detach().cpu())
                rows_pmask.append(pmask_this.detach().cpu())
                rows_labels.append(new_labels.detach().cpu())
                rows_seqids.append(b)
                rows_stepids.append(0)
                rows_attn.append(torch.ones(L, dtype=torch.bool))
                row_to_seq_map.append(b)

    if not row_mask_pos_list:
        return None

    Lmax = max(t.size(0) for t in clean_embeds_per_seq)

    row_mask_pos_list = [
        left_pad_1d(t.to(torch.long), Lmax, pad_val=0).to(torch.bool) for t in row_mask_pos_list
    ]
    pmask_rows = torch.stack([
        left_pad_1d(t.to(torch.long), Lmax, pad_val=0).to(torch.bool) for t in rows_pmask
    ], dim=0)
    labels_rows = torch.stack([
        left_pad_1d(t, Lmax, pad_val=-100) for t in rows_labels
    ], dim=0)
    attn_rows = torch.stack([
        left_pad_1d(t.to(torch.long), Lmax, pad_val=0).to(torch.bool) for t in rows_attn
    ], dim=0)

    step_ids_rows = torch.as_tensor(rows_stepids, dtype=torch.long)
    seq_ids_rows  = torch.as_tensor(rows_seqids,  dtype=torch.long)
    row_to_seq_map = torch.as_tensor(row_to_seq_map, dtype=torch.long)

    clean_embeds_per_seq = [
        left_pad_2d(t, Lmax, pad_val=0.0) for t in clean_embeds_per_seq
    ]

    rewards_raw = rewards

    return clean_embeds_per_seq, row_to_seq_map, row_mask_pos_list, mask_embed, pmask_rows, labels_rows, step_ids_rows, seq_ids_rows, rewards_raw, 0, 0, attn_rows




class TrainDataset(Dataset):
    def __init__(self,
                 clean_embeds_per_seq,  # List[(Lmax, D) bf16 CPU]
                 mask_embed,            # (D,) bf16 CPU
                 row_mask_pos_list,     # List[(Lmax,) bool CPU]
                 attn_mask,             # (N_rows, Lmax) bool
                 p_mask,                # (N_rows, Lmax) bool
                 row_step_ids,          # (N_rows,)
                 seq_ids,               # (N_rows,)
                 rewards_raw,           # 长度 = B; 标量或1D向量
                 row_to_seq_map         # (N_rows,)
                 ):

        self.clean_embeds = [t.to(torch.bfloat16).contiguous() for t in clean_embeds_per_seq]
        self.mask_embed = mask_embed.to(torch.bfloat16).contiguous()
        self.row_mask_pos_list = row_mask_pos_list

        self.attn_mask = attn_mask.to(torch.bool)
        self.p_mask = p_mask.to(torch.bool)
        self.row_step_ids = torch.as_tensor(row_step_ids, dtype=torch.long)
        self.seq_ids = torch.as_tensor(seq_ids, dtype=torch.long)
        self.row_to_seq_map = torch.as_tensor(row_to_seq_map, dtype=torch.long)

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
        self.per_seq_reward = (torch.tensor(scalar_vals, dtype=torch.float32) if all_scalar else None)

        N, L = self.p_mask.shape
        self.old_values = torch.zeros((N, L), dtype=torch.float32)
        self.Return     = torch.zeros((N, L), dtype=torch.float32)
        self.adv        = torch.zeros((N, L), dtype=torch.float32)

        D = int(self.mask_embed.numel())
        self._shape_for_debug = (N, L, D)

    def __len__(self):
        return self.p_mask.size(0)

    def __getitem__(self, idx):
        seq_idx = int(self.row_to_seq_map[idx].item())
        clean_embed = self.clean_embeds[seq_idx]      # (Lmax, D) bf16 CPU
        mask_pos = self.row_mask_pos_list[idx]        # (Lmax,) bool CPU

        noisy_embed = clean_embed.clone()
        noisy_embed[mask_pos] = self.mask_embed

        return (
            idx,
            noisy_embed,             # (Lmax, D) bf16 CPU
            self.attn_mask[idx],     # (Lmax,) bool
            self.p_mask[idx],        # (Lmax,) bool
            self.Return[idx],        # (Lmax,) float
        )

    @property
    def input_embeds(self):
        class _ShapeOnlyTensorProxy:
            def __init__(self, shape):
                self.shape = shape
        return _ShapeOnlyTensorProxy(self._shape_for_debug)



def save_checkpoint(model, tokenizer, config, accelerator, name):
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
        model_to_save.save_pretrained(
            save_base / name,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_base / name))

        metadata = {"save_time": time.strftime("%Y-%m-%d %H:%M:%S")}
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_base / name}")


def main():
    config = get_config()

    project_name = config.experiment.project
    pretrained_value = "./" + project_name + "/ckpt/" + config.model.optimized_value_name

    value_model_class = _get_value_model(LlavaLLaDAModelLM, "value_head")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_value, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    value_model = value_model_class.from_pretrained(pretrained_value, trust_remote_code=True, torch_dtype="auto")

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
        cfg_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {cfg_path}")
        OmegaConf.save(config, cfg_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

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
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=optimizer_config.value_learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

    value_model = value_model.to(accelerator.device)

    with open("./" + project_name + "/temp_data/" + config.dataset.optimization_data + f"-step{config.experiment.current_epoch}.json", "r") as f:
        dataset_load = json.load(f)
    if len(dataset_load) == 0:
        logger.warning("No optimization data found. Exiting value training early.")
        accelerator.end_training()
        return

    prompts   = [x["prompt"]   for x in dataset_load]
    responses = [x["response"] for x in dataset_load]
    rewards   = [x["reward"]   for x in dataset_load]
    step_maps = []
    for x in dataset_load:
        sm = x.get("step_map", None)
        if sm is None or len(sm) == 0:
            resp_len = len(tokenizer([x["response"]], add_special_tokens=False)["input_ids"][0])
            sm = list(range(resp_len))
        step_maps.append(sm)
    image_abs_list = [
        x.get("image_abs_path", None) or x.get("image", None) or x.get("image_path", None) for x in dataset_load
    ]

    _tok_tmp, _mod_tmp, image_processor, _ = load_pretrained_model(
       config.model.pretrained_model, None, "llava_llada", attn_implementation="sdpa", device_map="cpu"
    )
    del _tok_tmp, _mod_tmp


    rows_pack = build_rows_lladav(
        config, tokenizer, value_model, image_processor,
        prompts, responses, step_maps, rewards, image_abs_list
    )
    if rows_pack is None:
        logger.warning("Failed to build any training rows. Exiting.")
        accelerator.end_training()
        return

    clean_embeds_per_seq, row_to_seq_map, row_mask_pos_list, mask_embed, pmask_rows, labels_rows, step_ids_rows, seq_ids_rows, rewards_raw, start_pos, drop_num, attn_rows = rows_pack

    dataset_lm = TrainDataset(
        clean_embeds_per_seq=clean_embeds_per_seq,
        mask_embed=mask_embed,
        row_mask_pos_list=row_mask_pos_list,
        attn_mask=attn_rows,
        p_mask=pmask_rows,
        row_step_ids=step_ids_rows,
        seq_ids=seq_ids_rows,
        rewards_raw=rewards_raw,
        row_to_seq_map=row_to_seq_map,
    )


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
        idx, inp_e, attn, pmask, Ret = zip(*batch)
        return {
            "ids": torch.tensor(idx),
            "input_embeds": torch.stack(inp_e),   # (B, L, D)
            "attn_mask": torch.stack(attn),       # (B, L)
            "p_mask": torch.stack(pmask),         # (B, L)
            "Return": torch.stack(Ret),           # (B, L)
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
            input_embeds = batch["input_embeds"].to(accelerator.device)  # (B, L, D)
            attn_mask = batch["attn_mask"].to(accelerator.device)        # (B, L)
            p_mask = batch["p_mask"].to(accelerator.device)              # (B, L)

            m = getattr(value_model, "module", value_model)
            param_dtype = next(m.parameters()).dtype
            input_embeds = input_embeds.to(dtype=param_dtype)
            values = value_model(inputs_embeds=input_embeds, attention_mask=attn_mask)
            values = values * attn_mask.float()
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

            if isinstance(raw_r, (int, float)) or (torch.is_tensor(raw_r) and raw_r.ndim == 0):
                r_star[-1] = float(raw_r)
                return r_star

            if torch.is_tensor(raw_r):
                if raw_r.ndim != 1:
                    raise ValueError(f"Vector reward tensor must be 1D, got shape {tuple(raw_r.shape)}")
                raw_r = raw_r.tolist()

            if isinstance(raw_r, (list, tuple)):
                if len(raw_r) == S:
                    return torch.as_tensor(raw_r, dtype=torch.float32)

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

            R_star = torch.zeros(S, dtype=torch.float32)
            for i in range(S - 1, -1, -1):
                R_star[i] = r_star[i] + (gamma * R_star[i + 1] if i + 1 < S else 0.0)

            delta_star = torch.zeros(S, dtype=torch.float32)
            for i in range(S):
                v_next = V_star[i + 1] if i + 1 < S else 0.0
                delta_star[i] = r_star[i] - V_star[i] + gamma * v_next

            A_star = torch.zeros(S, dtype=torch.float32)
            for i in range(S - 1, -1, -1):
                A_star[i] = delta_star[i] + (gamma * lam * A_star[i + 1] if i + 1 < S else 0.0)

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
                            f"gamma=1, lambda=0 check failed at seq {s}, row {r}"

        dataset.Return = Return_mat
        dataset.adv    = adv_mat


    gam = config.training.get("gam", 1.0)
    lam = config.training.get("lam", 1.0)
    logger.info("***** Calculate Returns and Advantages *****")
    compute_returns_and_advantages_rows(dataset_lm, gam, lam, atol=1e-5)

    def save_dataset_tensors(dataset_lm, labels_rows, attn_rows, save_dir, name, accelerator, *, start_pos: int, drop_num: int):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if accelerator.is_main_process:
            print("Reconstructing full input_embeds for saving...")
            reconstructed_list = []
            for i in range(len(dataset_lm)):
                _, noisy_embed, _, _, _ = dataset_lm[i]
                reconstructed_list.append(noisy_embed)
            full_input_embeds = torch.stack(reconstructed_list, dim=0)
            print("Reconstruction complete.")

            payload = {
                "extended_input_embeds": full_input_embeds,  # (N_rows, Lmax, D)
                "attn_mask":            dataset_lm.attn_mask,      # (N_rows, Lmax) bool
                "p_mask":               dataset_lm.p_mask,         # (N_rows, Lmax) bool
                "labels":               labels_rows,               # (N_rows, Lmax)
                "adv":                  dataset_lm.adv,            # (N_rows, Lmax) float
                "meta": {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "start_pos": int(start_pos),
                    "drop_num":  int(drop_num),
                },
            }
            torch.save(payload, save_dir / f"{name}.pt")


    save_dataset_tensors(
        dataset_lm,
        labels_rows,
        attn_rows,
        save_dir=Path(config.experiment.project) / "temp_data",
        name=f"{config.dataset.optimization_data}",
        accelerator=accelerator,
        start_pos=start_pos,
        drop_num=drop_num
    )

    if config.experiment.current_epoch % config.experiment.train_value_every != 0:
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return

    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    logger.info("***** Running value training *****")
    logger.info(f"  Num samples (rows) = {len(dataset_lm)}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (parallel & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    def forward_process(input_embeds, attn_mask, p_mask, Return, old_values):
        m = getattr(value_model, "module", value_model)
        param_dtype = next(m.parameters()).dtype
        input_embeds = input_embeds.to(dtype=param_dtype)
        values = value_model(inputs_embeds=input_embeds, attention_mask=attn_mask)
        values = values * attn_mask.float()
        values = torch.where(p_mask, values, torch.zeros_like(values))

        v_clipped = old_values + (values - old_values).clamp(-config.training.eps, config.training.eps)
        loss_unclipped = (values - Return) ** 2
        loss_clipped = (v_clipped - Return) ** 2
        loss_tok = 0.5 * torch.maximum(loss_unclipped, loss_clipped) * p_mask

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

            input_embeds = batch["input_embeds"].to(accelerator.device)
            attn_mask = batch["attn_mask"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            Return = dataset_lm.Return[batch["ids"].cpu()].to(accelerator.device)
            old_vals = dataset_lm.old_values[batch["ids"].cpu()].to(accelerator.device)

            loss = forward_process(input_embeds, attn_mask, p_mask, Return, old_vals)
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

        if (step % accelerator.gradient_accumulation_steps) != 0:
            if config.training.max_grad_norm is not None:
                accelerator.clip_grad_norm_(value_model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            try:
                del input_embeds, attn_mask, p_mask, Return, old_vals
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
        outputs_name = "rl-" + pretrained_value.replace("/", ".") + "-" + config.dataset.train_dataset

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
