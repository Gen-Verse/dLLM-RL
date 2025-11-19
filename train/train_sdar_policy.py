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
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed



from models import SDARForCausalLM
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import Dataset, DataLoader

SYSTEM_PROMPT_LEN = 28

from train.utils import get_config, flatten_omega_conf, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")







class TrainDataset(Dataset):
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels, adv):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels
        self.adv   = adv
        self.logp_old_tok = torch.full(
            (len(extended_input_ids), p_mask.shape[1]), 
            float('-inf')
        )

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            idx,
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
            self.adv[idx],
        )


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
        split_batches=True,
    )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")


    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)

    #from transformers import AutoModelForCausalLM
    #model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
    model = SDARForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

    # calculate loss ourselves, needs logits，so aviod fuse CE
    if hasattr(model, "config"):
        model.config.fuse_cross_entropy = False   
    

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
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

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.policy_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")




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
    
    


    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")


    def simple_collate(batch):
        idx, extended_input_ids, p_mask, tok_idx_ext, labels, adv = zip(*batch)     # Tensor(L)
        return {
            "ids":        torch.tensor(idx),
            "extended_input_ids":  torch.stack(extended_input_ids),
            "p_mask":  torch.stack(p_mask),
            "tok_idx_ext":  torch.stack(tok_idx_ext),
            "labels":  torch.stack(labels),
            "adv":    torch.stack(adv)
        }
    


    dataset_load = torch.load(Path(project_name) / "temp_data" / f"{config.dataset.optimization_data}.pt", map_location="cpu")
    extended_input_ids = dataset_load["extended_input_ids"]
    p_mask            = dataset_load["p_mask"]
    tok_idx_ext       = dataset_load["tok_idx_ext"]
    labels            = dataset_load["labels"]
    adv               = dataset_load["adv"]
    start_pos = dataset_load["meta"]["start_pos"]
    drop_num  = dataset_load["meta"]["drop_num"]
    


    _, L = p_mask.shape
    L0    = start_pos
    L1    = L - L0
    post_num = config.training.post_num


    

    
    
    def make_basic_block_attention(
        N: int,
        start_pos: int,            # = L0
        block_size: int,           # = b
    ) -> torch.Tensor:
        B = 1
        L0     = start_pos
        L1     = (N - L0) // 2          # N = L0 + 2·L1 
        assert L0 + 2 * L1 == N, "input length must be L0 + 2*L1"

        # all -inf first
        bias = torch.full((B, 1, N, N), 0)


        rows = torch.arange(L0 + L1, L0 + 2 * L1)              # (L1,)
        rows_token = torch.arange(L0, L0 + L1)              # (L1,)

        # update block by block
        for bi in range((L1 + block_size - 1) // block_size):
            #  [bi*b , min((bi+1)*b, L1))
            left_end   = L0 + min((bi) * block_size, L1)        
            right_start= L0 + L1 + (left_end - L0)

            i_start = bi * block_size
            i_end   = min((bi + 1) * block_size, L1)              # no i_end

            block_rows = rows[i_start:i_end]                    
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end]   = 1
            bias[:, :, block_rows.unsqueeze(-1), right_start:(right_start + block_size)] = 1

            block_rows = rows_token[i_start:i_end]
            left_end   = L0 + min((bi + 1) * block_size, L1)
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end]   = 1
        
        if L0 > 0:
            num_blocks_pre = (L0 + block_size - 1) // block_size
            for bi in range(num_blocks_pre):
                # row interval [row_start, row_end)
                row_end   = max(L0 - bi * block_size, 0)
                row_start = max(L0 - (bi + 1) * block_size, 0)
                if row_end > row_start:
                    block_rows = torch.arange(row_start, row_end)
                    bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = 1
        
        return bias        # (B,1,N,N)
    
    
    

    basic_block_attention = make_basic_block_attention(L0 + 2 * L1, start_pos, config.training.block_size)
    basic_block_attention = basic_block_attention.cpu()


    def process_pad(attn, input_ids):
        N = L0 + 2 * L1
        device = input_ids.device

        cols = torch.arange(N, device=device)                  # (N,)
        key_mask = (cols < start_pos).unsqueeze(0) & (input_ids == pad_id)  # (B, N)

        # set -inf
        attn.masked_fill_(key_mask[:, None, None, :], 0)

        # aviod +-inf or none in forward
        A = attn[:, 0]  # (B, N, N)
        bad = (A.sum(dim=-1) == 0) & (torch.arange(A.size(1), device=A.device).unsqueeze(0) < start_pos)
        b, r = bad.nonzero(as_tuple=True)
        A[b, r, :] = 0; A[b, r, r] = 1  

        return attn


    dataset_lm = TrainDataset(extended_input_ids, p_mask, tok_idx_ext, labels, adv)

    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0
    )





    

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )





    import torch.nn.functional as F


    @torch.no_grad()
    def compute_logp_old_tok_parallel(
            accelerator,
            dataset,
            train_dataloader_lm,
            start_pos, pad_id,
            batch_size):

        model.eval()

        dl = train_dataloader_lm

        for batch in dl:
            ids        = batch["ids"]                       # (b,)
            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)

            B, L = p_mask.shape
            L0    = start_pos
            L1    = L - L0
            device = extended_input_ids.device

            attention_mask = basic_block_attention.clone()
            attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
            attention_mask = process_pad(attention_mask, extended_input_ids)

            logits = model(input_ids = extended_input_ids, attention_mask = attention_mask, position_ids = tok_idx_ext).logits
            logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1 :, :]], dim=1)  # (B, L0+L1, V)

            log_probs = F.log_softmax(logits, dim=-1)
            logp_tok  = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            dataset.logp_old_tok[ids] = logp_tok.float().cpu()

        accelerator.wait_for_everyone()

        model.train()


    #################################
    #             Inference         #
    #################################
    logger.info("***** Running inference *****")

    compute_logp_old_tok_parallel(
        accelerator,
        dataset_lm,
        train_dataloader_lm,
        start_pos=start_pos,
        pad_id=pad_id,
        batch_size=config.training.batch_size_lm,
    )






    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {p_mask.shape[0]}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()

    


    

    def forward_process(extended_input_ids, p_mask, tok_idx_ext, labels, adv, logp_old_tok):

        B, L = p_mask.shape
        L0    = start_pos
        L1    = L - L0
        device = extended_input_ids.device

        attention_mask = basic_block_attention.clone()
        attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
        attention_mask = process_pad(attention_mask, extended_input_ids)

        logits = model(input_ids = extended_input_ids, attention_mask = attention_mask, position_ids = tok_idx_ext).logits
        logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1 :, :]], dim=1)  # (B, L0+L1, V)

        log_probs = F.log_softmax(logits, dim=-1)
        
        logp_new_tok  = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)     # (B, T)

        ratio   = logp_new_tok - logp_old_tok
        ratio = torch.where(p_mask, ratio, torch.zeros_like(ratio)).clamp(-10.0, 10.0) # for stablability
        ratio   = torch.exp(ratio)          # (B, T)
        clipped = torch.clamp(ratio, 1 - config.training.eps, 1 + config.training.eps)            # (B, T)

        surrogate_tok = torch.min(ratio * adv, clipped * adv)  # (B, T)
        surrogate_tok = surrogate_tok * p_mask

        num_mask = torch.clamp(p_mask.sum(dim=1), min=1)
        surrogate_tok = surrogate_tok.sum(dim=1) / L1

        policy_loss = - (surrogate_tok.sum() / B)

        # ---------- 6. KL penalty（可选） ----------
        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        if config.training.beta > 0:
            kl_seq = logp_new_tok - logp_old_tok
            kl_seq = torch.where(p_mask, kl_seq, torch.zeros_like(kl_seq))
            if config.training.use_kl_estimator_k3:
                t = (-kl_seq).clamp(-10.0, 10.0)
                kl_seq = t.exp() - 1.0 + kl_seq
            kl_seq = (kl_seq * p_mask).sum(dim=1) / L1
            kl_loss = config.training.beta * kl_seq.sum() / B
            total_loss = policy_loss + kl_loss
        else:
            total_loss = policy_loss

        return total_loss






    from tqdm.auto import tqdm

    for epoch in range(first_epoch, num_train_epochs):
        
        model.train()
        
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,         
            leave=True              
        )
        
        

        for step, batch in enumerate(progress_bar, start=1):
            
            # for loss calculation

            data_time_m.update(time.time() - end)

            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            adv = batch["adv"].to(accelerator.device)
            old_lp = dataset_lm.logp_old_tok[batch["ids"].cpu()].to(accelerator.device)

            if torch.isneginf(old_lp).any().item():
                print(old_lp)

            loss_lm = forward_process(
                    extended_input_ids=extended_input_ids,
                    p_mask=p_mask,
                    tok_idx_ext=tok_idx_ext,
                    labels=labels,
                    adv=adv,
                    logp_old_tok=old_lp
                )
            loss_lm = loss_lm / accelerator.gradient_accumulation_steps


            if step < 10:
                print(loss_lm)
            accelerator.backward(loss_lm)

            if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                torch.cuda.empty_cache()
            

                


    accelerator.wait_for_everyone()

    # save checkpoint at the end of training
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)
    if config.experiment.current_epoch % config.experiment.save_every == 0:
        save_checkpoint(model, tokenizer, config, accelerator, f"epoch-{config.experiment.current_epoch}-policy")

    accelerator.end_training()






def save_checkpoint(model, tokenizer, config, accelerator, name):
    from pathlib import Path
    import time, json, shutil, os, glob, importlib, inspect

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
                    src_file = inspect.getsourcefile(mod)  # e.g. .../modeling_sdar.py
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

            logger.info(f"Copied {copied} custom module files into {dst_dir}")

        _copy_dynamic_modules(str(save_dir), model_to_save, tokenizer)

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_dir}")

    















if __name__ == "__main__":
    main()
