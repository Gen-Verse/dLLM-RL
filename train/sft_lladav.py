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
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from omegaconf import OmegaConf
import wandb

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from train.utils import get_config, flatten_omega_conf, AverageMeter
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

# LLaDA-V / LLaVA
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates

logger = get_logger(__name__, log_level="INFO")


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


def build_llava_conv_prompt(prompt_text: str, response_text: str) -> str:
    import copy
    conv = copy.deepcopy(conv_templates["llava_llada"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt_text)
    conv.append_message(conv.roles[1], response_text)
    return conv.get_prompt().removesuffix("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def tokenize_conv_with_image(prompt_text: str, response_text: str, tokenizer) -> torch.Tensor:
    conv_prompt = build_llava_conv_prompt(prompt_text, response_text)
    ids = tokenizer_image_token(conv_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    return ids.squeeze(0)  # (L,)


def load_and_process_image(abs_path: str, image_processor, model_cfg, device, dtype=torch.bfloat16):
    from PIL import Image
    from types import SimpleNamespace

    image = Image.open(abs_path).convert("RGB")

    if isinstance(model_cfg, dict):
        model_cfg = SimpleNamespace(**model_cfg)
    if not hasattr(model_cfg, "image_aspect_ratio") or getattr(model_cfg, "image_aspect_ratio") is None:
        setattr(model_cfg, "image_aspect_ratio", "pad")
    if not hasattr(model_cfg, "mm_patch_merge_type") or getattr(model_cfg, "mm_patch_merge_type") is None:
        setattr(model_cfg, "mm_patch_merge_type", "flat")

    image_tensor_list = process_images([image], image_processor, model_cfg)
    image_tensor_list = [_im.to(dtype=dtype, device=device) for _im in image_tensor_list]
    image_sizes = [image.size]
    return image_tensor_list, image_sizes


def left_pad_1d(x: torch.Tensor, L: int, pad_val: int):
    out = torch.full((L,), pad_val, dtype=x.dtype, device=x.device)
    out[-x.numel():] = x
    return out


def left_pad_2d(x: torch.Tensor, L: int, pad_val: float = 0.0):
    if x.size(0) == L:
        return x
    out = torch.full((L, x.size(1)), pad_val, dtype=x.dtype, device=x.device)
    out[-x.size(0):] = x
    return out


def resolve_image_abs_path(sample: dict, dataset_image_root: str | None) -> str | None:
    p = sample.get("image_abs_path")
    if p is None:
        p = sample.get("image") or sample.get("image_path")
        if p is None:
            return None
        if not os.path.isabs(p) and dataset_image_root:
            p = os.path.join(dataset_image_root, p)
    try:
        return os.path.abspath(p)
    except Exception:
        return p


class SFTVariantDatasetLLADAV(Dataset):

    def __init__(self, samples: List[dict], method: str, lower: float, upper: float, block_size: int, mask_times: int, max_gen_length: int):
        super().__init__()
        self.samples = samples
        self.method = method
        self.lower = lower
        self.upper = upper
        self.block_size = block_size
        self.mask_times = mask_times
        self.max_gen_length = max_gen_length

        self.variants: List[dict] = []
        self._build_variants()

    def _build_variants(self):
        rng = np.random.default_rng(seed=42)
        for sid, s in enumerate(self.samples):
            if self.method == "semi-ar":
                step_map = s.get("step_map", None)
                if step_map is None:
                    step_map = list(range(self.max_gen_length))
                order_list = collapse_k_unique(step_map, self.block_size)
                uniq_steps = sorted(set(order_list))
                for st in uniq_steps:
                    self.variants.append(dict(
                        sample_id=sid,
                        prompt=s["prompt"],
                        response=s["response"],
                        image_abs_path=s.get("image_abs_path", None),
                        image=s.get("image", None),
                        image_path=s.get("image_path", None),
                        var_type="semi_ar",
                        var_param=int(st),
                        step_map=order_list,
                        mask_seed=None,
                    ))
            elif self.method == "random_masking":
                for _ in range(self.mask_times):
                    t = float(rng.uniform(self.lower, self.upper))
                    mask_seed = int(rng.integers(0, 2**31 - 1))
                    self.variants.append(dict(
                        sample_id=sid,
                        prompt=s["prompt"],
                        response=s["response"],
                        image_abs_path=s.get("image_abs_path", None),
                        image=s.get("image", None),
                        image_path=s.get("image_path", None),
                        var_type="random_masking",
                        var_param=t,
                        step_map=None,
                        mask_seed=mask_seed,
                    ))
            else:
                raise ValueError(f"Unknown SFT method: {self.method}")

    def __len__(self):
        return len(self.variants)

    def __getitem__(self, idx):
        v = self.variants[idx]
        return {
            "idx": idx,
            "sample_id": v["sample_id"],
            "prompt": v["prompt"],
            "response": v["response"],
            "image_abs_path": v.get("image_abs_path", None),
            "image": v.get("image", None),
            "image_path": v.get("image_path", None),
            "var_type": v["var_type"],
            "var_param": v["var_param"],
            "step_map": v.get("step_map", None),
            "mask_seed": v.get("mask_seed", None),
        }


def collate_variants(batch: List[dict]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = [b[k] for b in batch]
    return out


@torch.no_grad()
def prepare_batch_noisy_embeds(
    accelerator,
    model,
    tokenizer,
    image_processor,
    batch: Dict[str, List[Any]],
    mask_id: int,
    dataset_image_root: str | None,
) -> tuple | None:

    device = accelerator.device
    bsz = len(batch["idx"])

    group: Dict[int, List[int]] = {}
    for i, sid in enumerate(batch["sample_id"]):
        group.setdefault(int(sid), []).append(i)

    sample_cache: Dict[int, Dict[str, Any]] = {}
    for sid, pos_list in group.items():
        first_pos = pos_list[0]
        prompt_text = batch["prompt"][first_pos]
        response_text = batch["response"][first_pos]

        input_ids_text = tokenize_conv_with_image(prompt_text, response_text, tokenizer).to(device)

        end_header_seq = tokenizer.encode("<|end_header_id|>\n\n", add_special_tokens=False)
        seq_len = len(end_header_seq)
        pos_eh = -1
        if seq_len > 0 and input_ids_text.numel() >= seq_len:
            ids_list = input_ids_text.tolist()
            for i in range(len(ids_list) - seq_len, -1, -1):
                if ids_list[i:i + seq_len] == end_header_seq:
                    pos_eh = i + seq_len - 1
                    break
        assert pos_eh != -1, " <|end_header_id|> not found"
        start_pos = pos_eh + 1

        labels_text = input_ids_text.clone()
        labels_text[:start_pos] = -100

        sample_stub = {
            "image_abs_path": batch.get("image_abs_path", [None])[first_pos],
            "image": batch.get("image", [None])[first_pos],
            "image_path": batch.get("image_path", [None])[first_pos],
        }
        p_abs = resolve_image_abs_path(sample_stub, dataset_image_root)

        if p_abs is not None:
            img_list, image_sizes = load_and_process_image(
                p_abs, image_processor, model.config, device, dtype=next(model.parameters()).dtype
            )
        else:
            img_list, image_sizes = None, None

        (_in_ids, _pos_ids, _attn_mask, _pkv, clean_embeds, new_labels) = model.prepare_inputs_labels_for_multimodal(
            input_ids_text.unsqueeze(0), None, None, None, labels_text.unsqueeze(0), img_list, ["image"], image_sizes=image_sizes
        )
        clean_embeds = clean_embeds.squeeze(0)  # (L, D)
        new_labels = new_labels.squeeze(0)      # (L,)

        eot_id = 126348
        eot_indices = (new_labels == eot_id).nonzero(as_tuple=False)
        if eot_indices.numel() > 0:
            last_eot_pos = eot_indices[-1, 0]
            new_labels[last_eot_pos:] = -100
        asst_mask = (new_labels != -100)
        asst_idx = asst_mask.nonzero(as_tuple=False).squeeze(-1)  # (M,)

        sample_cache[sid] = dict(
            clean_embeds=clean_embeds,
            new_labels=new_labels,
            asst_idx=asst_idx,
        )

    lengths = []
    for i in range(bsz):
        sid = int(batch["sample_id"][i])
        lengths.append(int(sample_cache[sid]["clean_embeds"].shape[0]))
    Lmax = max(lengths)
    D = next(iter(sample_cache.values()))["clean_embeds"].shape[1]

    mask_embed = model.get_model().embed_tokens(torch.tensor([mask_id], device=device)).squeeze(0)

    noisy_list = []
    labels_list = []
    p_mask_list = []
    attn_list = []

    for i in range(bsz):
        sid = int(batch["sample_id"][i])
        var_type = batch["var_type"][i]
        var_param = batch["var_param"][i]
        step_map = batch["step_map"][i]

        clean_embeds = sample_cache[sid]["clean_embeds"].clone()
        new_labels = sample_cache[sid]["new_labels"].clone()
        asst_idx = sample_cache[sid]["asst_idx"]

        L = clean_embeds.size(0)
        pmask = torch.zeros(L, dtype=torch.bool, device=device)

        if var_type == "semi_ar":
            order = torch.as_tensor(step_map, device=device)
            M = asst_idx.numel()
            if order.numel() != M:
                m = min(M, order.numel())
                asst_idx = asst_idx[:m]
                order = order[:m]
            order_full = torch.full((L,), -1, dtype=torch.long, device=device)
            order_full[asst_idx] = order
            step_val = int(var_param)

            p = torch.empty(L, device=device).uniform_(0.0, 1.0)  
            lower = 0.0 + 1e-6
            upper = 1.0 - 1e-6
            thresh = (upper - lower) * torch.rand(1, device=device) + lower
            block_mask = (order_full == step_val) & (p < thresh)

            mask_pos = (order_full > step_val) | block_mask
            noisy_embeds = clean_embeds.clone()
            noisy_embeds[mask_pos] = mask_embed

            pmask = block_mask

        elif var_type == "random_masking":
            M = asst_idx.numel()
            if M == 0:
                noisy_embeds = clean_embeds
                pmask = torch.zeros(L, dtype=torch.bool, device=device)
            else:
                t = float(var_param)
                mask_seed = batch.get("mask_seed", [None] * bsz)[i]
                if mask_seed is None:
                    mask_seed = int((int(batch["sample_id"][i]) * 1000003 + int(i)) & 0x7fffffff)
                gen = torch.Generator(device="cpu")
                gen.manual_seed(int(mask_seed))
                rnd_cpu = torch.rand(M, generator=gen) < t
                rnd = rnd_cpu.to(device=device)

                pmask = torch.zeros(L, dtype=torch.bool, device=device)
                pmask[asst_idx] = rnd

                noisy_embeds = clean_embeds.clone()
                noisy_embeds[pmask] = mask_embed

        else:
            continue

        if not pmask.any():
            continue

        noisy_list.append(left_pad_2d(noisy_embeds, Lmax, pad_val=0.0))
        labels_list.append(left_pad_1d(new_labels, Lmax, pad_val=-100))
        p_mask_list.append(left_pad_1d(pmask.to(torch.long), Lmax, pad_val=0).to(torch.bool))
        attn = torch.zeros(Lmax, dtype=torch.bool, device=device)
        attn[-L:] = True
        attn_list.append(attn)

    if len(noisy_list) == 0:
        return None

    noisy_batch = torch.stack(noisy_list, dim=0)
    labels_batch = torch.stack(labels_list, dim=0)
    p_mask_batch = torch.stack(p_mask_list, dim=0)
    attn_batch = torch.stack(attn_list, dim=0)
    return noisy_batch, labels_batch, p_mask_batch, attn_batch


def save_checkpoint(model, tokenizer, config, accelerator, name):
    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
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

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_base / name}")


def main():
    config = get_config()

    # TF32
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
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    pretrained_model = config.model.pretrained_model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        pretrained_model, None, "llava_llada", attn_implementation="sdpa", device_map=accelerator.device
    )
    model = model.to(accelerator.device)

    special_tokens = {"additional_special_tokens": [DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]}
    num_new = tokenizer.add_special_tokens(special_tokens)
    if num_new > 0:
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(config.training, "gradient_checkpointing_enable", False):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    mask_id = 126336

    with open("./data/" + config.dataset.optimization_data + ".json", 'r') as f:
        data_list = json.load(f)

    ds = SFTVariantDatasetLLADAV(
        samples=data_list,
        method=config.training.method,
        lower=config.training.lower_p,
        upper=config.training.upper_p,
        block_size=config.training.block_size,
        mask_times=config.training.mask_times_per_sample,
        max_gen_length=config.training.max_gen_length
    )

    # Scheduler & DataLoader
    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(ds) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    optimizer_config = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    dl = DataLoader(
        ds,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=collate_variants,
        num_workers=0
    )

    model, optimizer, lr_scheduler, dl = accelerator.prepare(model, optimizer, lr_scheduler, dl)

    logger.info("***** Running LLaDA-V SFT training *****")
    logger.info(f"  Num variants = {len(ds)}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (parallel * accum) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    data_time_m = AverageMeter()
    end = time.time()

    from tqdm.auto import tqdm

    def forward_process_batch(batch):
        out = prepare_batch_noisy_embeds(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch=batch,
            mask_id=mask_id,
            dataset_image_root=config.dataset.get("image_root", None),
        )
        if out is None:
            return None
        noisy_embeds, labels, p_mask, attn = out

        outputs = model.get_model()(
            inputs_embeds=noisy_embeds,
            attention_mask=attn,
            use_cache=False,
            return_dict=True
        )
        logits = model.lm_head(outputs.last_hidden_state).float()  # (B, L, V)
        log_probs = F.log_softmax(logits, dim=-1)

        safe_labels = labels.clone()
        safe_labels[labels == -100] = 0
        logp_tok = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, L)

        loss_tok = - (logp_tok * p_mask).sum(dim=1)
        num_mask = torch.clamp(p_mask.sum(dim=1), min=1)
        loss = (loss_tok / num_mask).mean()
        return loss

    first_epoch = 0
    grad_accum = config.training.gradient_accumulation_steps

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        progress_bar = tqdm(
            dl,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True
        )

        step = 0
        for step, batch in enumerate(progress_bar, start=1):
            data_time_m.update(time.time() - end)

            loss = forward_process_batch(batch)
            if loss is None:
                continue
            loss = loss / grad_accum
            accelerator.backward(loss)

            if step % grad_accum == 0:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        if step % grad_accum != 0:
            if config.training.max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)
    accelerator.end_training()


if __name__ == "__main__":
    main()
