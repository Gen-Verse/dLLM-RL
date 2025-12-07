# Path: ./train/train_lladav_policy.py
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

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from omegaconf import OmegaConf
import wandb

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from train.utils import get_config, flatten_omega_conf, AverageMeter
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

# LLaDA‑V / LLaVA
from llava.model.builder import load_pretrained_model
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__, log_level="INFO")


class TrainDataset(Dataset):
    def __init__(self, input_embeds, labels, p_mask, adv, attn_mask):
        """
        input_embeds: (N, L, D) float32
        labels:       (N, L)     int64  (-100 for ignore)
        p_mask:       (N, L)     bool   positions supervised by PPO loss
        adv:          (N, L)     float32 advantages from value stage
        attn_mask:    (N, L)     bool   left-pad等无效位置
        """
        self.inputs   = input_embeds
        self.labels   = labels
        self.pmasks   = p_mask.to(torch.bool)
        self.adv      = adv
        self.attn     = attn_mask.to(torch.bool)
        N, L, _ = input_embeds.shape
        self.logp_old_tok = torch.full((N, L), float('-inf'), dtype=torch.float32)  # filled after inference

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return (
            idx,
            self.inputs[idx],
            self.labels[idx],
            self.pmasks[idx],
            self.adv[idx],
            self.attn[idx],
        )


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

        metadata = {"save_time": time.strftime("%Y-%m-%d %H:%M:%S")}
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_base / name}")


@torch.no_grad()
def compute_logp_old_tok_parallel(accelerator, model, dataset, dataloader):
    model.eval()
    for batch in dataloader:
        ids, input_embeds, labels, p_mask, adv, attn = batch["ids"], batch["input_embeds"], batch["labels"], batch["p_mask_lm"], batch["adv"], batch["attn_mask"]
        input_embeds = input_embeds.to(accelerator.device, dtype=next(model.parameters()).dtype)
        labels       = labels.to(accelerator.device)
        attn         = attn.to(accelerator.device)

        # 前向（embedding 版）
        outputs = model.get_model()(inputs_embeds=input_embeds, attention_mask=attn, use_cache=False, return_dict=True)
        logits  = model.lm_head(outputs.last_hidden_state).float()  # (B, L, V)
        log_probs = F.log_softmax(logits, dim=-1)

        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0
        tok_lp  = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, L)

        dataset.logp_old_tok[ids, :labels.shape[1]] = tok_lp.detach().float().cpu()
    accelerator.wait_for_everyone()
    model.train()


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    project_name = config.experiment.project
    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "./" + project_name + "/ckpt/" + config.model.optimized_name

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
        split_batches=False,
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

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading LLaDA‑V policy model")

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

    # no decay on bias and layernorm and embedding
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

    ##################################
    #       Preprocess data          #
    #################################
    logger.info("Loading optimization tensors for policy PPO")

    pt_path = Path(project_name) / "temp_data" / f"{config.dataset.optimization_data}.pt"
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        gloo_group = dist.new_group(backend='gloo')
        
        metadata_list = [None]
        
        if rank == 0:
            dataset_pt = torch.load(pt_path, map_location="cpu")
            all_tensors = {
                "input_embeds": dataset_pt["extended_input_embeds"],
                "attn_mask":    dataset_pt["attn_mask"],
                "p_mask_lm":    dataset_pt["p_mask"],
                "labels":       dataset_pt["labels"],
                "adv_t":        dataset_pt["adv"],
            }

            N = all_tensors["input_embeds"].size(0)
            pad = (world_size - (N % world_size)) % world_size
            if pad > 0:
                rem = N % world_size
                pad_idx = torch.randint(low=0, high=rem, size=(pad,), dtype=torch.long)
                for key in all_tensors:
                    all_tensors[key] = torch.cat([all_tensors[key], all_tensors[key][pad_idx]], dim=0)

            N_padded = all_tensors["input_embeds"].size(0)
            local_size = N_padded // world_size

            metadata = {key: ( (local_size,) + tensor.shape[1:], tensor.dtype) for key, tensor in all_tensors.items()}
            metadata_list = [metadata]
            
            tensors_to_scatter = {key: list(torch.chunk(tensor, world_size, dim=0)) for key, tensor in all_tensors.items()}

        dist.broadcast_object_list(metadata_list, src=0, group=gloo_group)
        metadata = metadata_list[0]

        local_tensors = {key: torch.empty(shape, dtype=dtype) for key, (shape, dtype) in metadata.items()}

        for key in metadata.keys():
            dist.scatter(
                local_tensors[key], 
                tensors_to_scatter[key] if rank == 0 else None, 
                src=0, 
                group=gloo_group
            )
        
        input_embeds = local_tensors["input_embeds"]
        attn_mask    = local_tensors["attn_mask"]
        p_mask_lm    = local_tensors["p_mask_lm"]
        labels       = local_tensors["labels"]
        adv_t        = local_tensors["adv_t"]




    dataset_lm = TrainDataset(input_embeds, labels, p_mask_lm, adv_t, attn_mask)


    ##################################
    #       Prepare accelerator       #
    #################################
    logger.info("Preparing dataloader and scheduler")

    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = max(1, math.ceil((len(dataset_lm) * accelerator.num_processes) / total_batch_size_lm))
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    def simple_collate(batch):
        idx, inp, lbl, msk, adv, attn = zip(*batch)
        return {
            "ids":        torch.tensor(idx),
            "input_embeds": torch.stack(inp),     # (B, L, D)
            "labels":     torch.stack(lbl),       # (B, L)
            "p_mask_lm":  torch.stack(msk),       # (B, L)
            "adv":        torch.stack(adv),       # (B, L)
            "attn_mask":  torch.stack(attn),      # (B, L)
        }

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0
    )

    if hasattr(accelerator.state, "deepspeed_plugin") and accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = int(config.training.batch_size_lm)

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )


    #################################
    #       Old log-prob inference  #
    #################################
    logger.info("***** Running old policy log-prob inference *****")
    compute_logp_old_tok_parallel(
        accelerator=accelerator,
        model=model,
        dataset=dataset_lm,
        dataloader=train_dataloader_lm
    )

    ##################################
    #             Training           #
    #################################
    logger.info("***** Running PPO training *****")
    logger.info(f"  Num rows (global) = {len(dataset_lm) * accelerator.num_processes}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (parallel * accum) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    data_time_m = AverageMeter()
    end = time.time()

    def forward_process(input_embeds, labels, p_mask_lm, adv, attn_mask, logp_old_tok):
        logits = model.get_model()(inputs_embeds=input_embeds, attention_mask=attn_mask, use_cache=False, return_dict=True).last_hidden_state
        logits = model.lm_head(logits).float()  # (B, L, V)

        B, L, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0
        logp_new_tok = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, L)

        eff_mask = p_mask_lm & attn_mask
        num_mask = torch.clamp(eff_mask.sum(dim=1), min=1)

        ratio   = torch.exp((logp_new_tok - logp_old_tok).clamp(-10.0, 10.0))  # (B, L)
        clipped = torch.clamp(ratio, 1 - config.training.eps, 1 + config.training.eps)

        surrogate_tok = torch.min(ratio * adv, clipped * adv) * eff_mask
        surrogate_seq = surrogate_tok.sum(dim=1) / num_mask
        policy_loss = - surrogate_seq.mean()

        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        if config.training.beta > 0:
            kl_seq = (logp_new_tok - logp_old_tok)
            if config.training.use_kl_estimator_k3:
                t = (-kl_seq).clamp(-10.0, 10.0)
                kl_seq = t.exp() - 1.0 + kl_seq
            kl_seq = (kl_seq * eff_mask).sum(dim=1) / num_mask
            kl_loss = config.training.beta * kl_seq.mean()

        return policy_loss + kl_loss

    from tqdm.auto import tqdm
    grad_accum = config.training.gradient_accumulation_steps

    for epoch in range(num_train_epochs):
        model.train()
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True
        )

        step = 0
        for step, batch in enumerate(progress_bar, start=1):
            data_time_m.update(time.time() - end)

            input_embeds = batch["input_embeds"].to(accelerator.device, dtype=next(model.parameters()).dtype)
            labels       = batch["labels"].to(accelerator.device)
            p_mask_lm    = batch["p_mask_lm"].to(accelerator.device)
            adv          = batch["adv"].to(accelerator.device)
            attn_mask    = batch["attn_mask"].to(accelerator.device)
            old_lp       = dataset_lm.logp_old_tok[batch["ids"].cpu()].to(accelerator.device)[:, :labels.size(1)]

            loss = forward_process(
                input_embeds=input_embeds,
                labels=labels,
                p_mask_lm=p_mask_lm,
                adv=adv,
                attn_mask=attn_mask,
                logp_old_tok=old_lp
            )
            loss = loss / grad_accum

            if step <= 10:
                print(loss)

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
    if config.experiment.current_epoch % config.experiment.save_every == 0:
        save_checkpoint(model, tokenizer, config, accelerator, f"epoch-{config.experiment.current_epoch}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
