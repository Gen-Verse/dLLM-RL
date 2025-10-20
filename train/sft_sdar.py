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
from transformers import AutoTokenizer
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
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx]
        )


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    project_name = config.experiment.project
    pretrained_model = config.model.pretrained_model

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
            lr=optimizer_config.learning_rate,
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
        extended_input_ids, p_mask, tok_idx_ext, labels = zip(*batch)     # Tensor(L)
        return {
            "extended_input_ids":  torch.stack(extended_input_ids),
            "p_mask":  torch.stack(p_mask),
            "tok_idx_ext":  torch.stack(tok_idx_ext),
            "labels":  torch.stack(labels)
        }
    


    
    with open("./data/" + config.dataset.optimization_data + ".json", 'r') as f:
        dataset_load = json.load(f)
    #dataset_load = dataset_load[10000:20000]

    prompt_list = []
    response_list = []
    step_map_list = []
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
    
    input_ids_lm, _, start_pos, drop_num = uni_prompting((prompt_list, response_list))

    _, L = input_ids_lm.shape
    L0    = start_pos
    L1    = L - L0
    post_num = config.training.post_num

    for x in dataset_load:
        if "step_map" not in x.keys():
            step_map_list.append([j for j in range(L1)])
        else:
            step_map_i = x["step_map"]
            if len(step_map_i) > L1:
                step_map_i = step_map_i[:L1]
            else:
                step_map_i = step_map_i + [max(step_map_i) + 1] * (L1 - len(step_map_i))
            step_map_list.append(step_map_i)

    
    
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





    def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
        """
        Perform a single "round" on one sample b:
        - For each block, take the minimum non -1 value in step_map.
        - Create pmask (positions equal to the block minimum).
        - Create a noise mask for the extended segment (positions >= block minimum).
        - Mark the chosen minimum positions in step_map as -1 for the next round.

        Returns:
        extended_input_ids_b : Tensor with duplicated + masked response segment
        pmask_b              : Boolean mask for tokens selected in this round
        new_step_map_b       : Updated step_map (selected positions set to -1)
        has_any              : Whether any position was selected in this round
        """
        device = input_ids_b.device
        NB = (L1 + block_size - 1) // block_size
        pad_len = NB * block_size - L1

        # Reshape step_map into [NB, block_size], fill last incomplete block with -1
        step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
        step_pad[:L1] = step_map_b
        step_blk = step_pad.view(NB, block_size)                      # [NB, Bk]

        valid = step_blk.ge(0)                                        # Valid positions (not -1)
        big = torch.iinfo(step_blk.dtype).max
        tmp = step_blk.masked_fill(~valid, big)                       # Fill invalid positions with a large value
        min_vals, _ = tmp.min(dim=1, keepdim=True)                    # Current minimum for each block

        # Select positions equal to block minimum (only valid positions)
        pmask_blk = step_blk.eq(min_vals) & valid                     
        if not pmask_blk.any():
            # No positions left to select in this round
            return None, None, step_map_b, False

        # Noise mask for extended segment: mark positions >= block minimum
        ge_mask_blk = step_blk.ge(min_vals) & valid                   # [NB, Bk]

        # Flatten back to length L1 (discard padding)
        pmask_tail = pmask_blk.view(-1)[:L1]                          # [L1]
        ge_mask_tail = ge_mask_blk.view(-1)[:L1]                      # [L1]

        # Construct pmask_b: [0:L0] = False, [L0:] = pmask_tail
        pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
        pmask_b[L0:] = pmask_tail

        # Build extended segment: duplicate response and replace noise positions with mask_id
        tail = input_ids_b[L0:L0+L1].clone()
        tail[ge_mask_tail] = mask_id

        
        extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
        extended_input_ids_b[:L0+L1] = input_ids_b
        extended_input_ids_b[L0+L1:] = tail

        # Update step_map: mark selected minimum positions as -1 for the next round
        new_step_map_b = step_map_b.clone()
        new_step_map_b[pmask_tail] = -1

        return extended_input_ids_b, pmask_b, new_step_map_b, True




    def collect_training_data(input_ids, step_map_list):

        B, L = input_ids.shape
        L0    = start_pos
        L1    = L - L0
        block_size = config.training.block_size

        lower = config.training.lower_p
        upper = config.training.upper_p


        '''
        if config.training.method == "semi-ar":

            extended_input_ids_list, pmask_list = [], []
            
            
            for b in range(B):

                extended_input_ids_b = input_ids[b]
                pmask_b = torch.zeros(start_pos, dtype=torch.bool)
                
                for j in range(int((L1 - 1) / block_size) + 1):

                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)

                    pmask_b_j = torch.rand(end - start) <= torch.empty(end - start).uniform_(lower, upper)
                    #pmask_b_j = torch.rand(end - start) <= torch.linspace(lower, upper, steps=end - start)
                    pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)

                    noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                    noise_b_j = noise_b_j.masked_fill_(pmask_b_j, mask_id)

                    extended_input_ids_b = torch.cat([extended_input_ids_b, noise_b_j], dim=0)
                
                extended_input_ids_list.append(extended_input_ids_b)
                pmask_list.append(pmask_b)'''
        
        if config.training.method == "semi-ar":

            extended_input_ids_list, pmask_list = [], []

            # Pre-compute the global linear probability
            

            for b in range(B):

                prob_ramp = torch.empty(L1).uniform_(lower, upper)
                
                # 1) Sample random numbers once for each position in the tail segment of this sample
                rand_tail = torch.rand(L1)
                pmask_tail = rand_tail <= prob_ramp  # [L1], True means to mask

                # 2) Construct the pmask for this sample (prefix all False, tail uses pmask_tail)
                pmask_b = torch.cat([
                    torch.zeros(L0, dtype=torch.bool),
                    pmask_tail
                ], dim=0)  # [L]

                # 3) Construct the noisy tail and append it to the original sequence to get the extended sequence
                noise_tail = input_ids[b, L0:].clone()            # [L1]
                noise_tail.masked_fill_(pmask_tail, mask_id)      # Replace masked positions
                extended_b = torch.cat([input_ids[b], noise_tail], dim=0)  # [L + L1]

                extended_input_ids_list.append(extended_b)
                pmask_list.append(pmask_b)

            
        elif config.training.method == "trace":

            for b in range(B):
                step_map_i = step_map_list[b]

                for j in range(int((L1 - 1) / block_size) + 1):
                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)
                    step_map_list[b][start:end] = collapse_k_unique(step_map_i[start:end], config.training.shrink)
            
            step_map = torch.as_tensor(step_map_list, dtype=torch.long)

            assert step_map.shape[1] == L1

            extended_input_ids_list, pmask_list = [], []

            for b in range(B):
                step_b = step_map[b]
                while True:
                    out = one_round_vectorized(
                        input_ids_b=input_ids[b],
                        step_map_b=step_b,
                        L0=L0,
                        L1=L1,
                        block_size=block_size,
                        mask_id=mask_id,
                    )
                    extended_b, pmask_b, step_b, has_any = out
                    if not has_any:
                        break

                    extended_input_ids_list.append(extended_b)
                    pmask_list.append(pmask_b)

        extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
        p_mask =  torch.stack(pmask_list, dim=0).to(torch.bool)
        
        pad_resp = (extended_input_ids[:, :L] == pad_id) & p_mask        
        if post_num is not None:
            cum_pad = torch.cumsum(pad_resp.int(), dim=1)
            p_mask &= ~(pad_resp & (cum_pad > post_num))
        
        labels = extended_input_ids[:, :L].clone()

        idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
        valid = (idx >= start_pos) | extended_input_ids[:, :L].ne(pad_id)      
        tok_idx = valid.long().cumsum(dim=-1) - 1         
        tok_idx = tok_idx.masked_fill(~valid, 1)
        tok_idx_resp = tok_idx[:, start_pos:]  
        tok_idx_ext  = torch.cat([tok_idx, tok_idx_resp], dim=1)

        keep = p_mask.view(p_mask.size(0), -1).any(dim=1)

        extended_input_ids = extended_input_ids[keep]
        p_mask            = p_mask[keep]
        tok_idx_ext       = tok_idx_ext[keep]
        labels            = labels[keep]

        return extended_input_ids, p_mask, tok_idx_ext, labels
        

    
    extended_input_ids, p_mask, tok_idx_ext, labels = collect_training_data(input_ids_lm, step_map_list)




    dataset_lm = TrainDataset(extended_input_ids, p_mask, tok_idx_ext, labels)

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



    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {extended_input_ids.shape[0]}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()

    import torch.nn.functional as F






    

    ''' To be updated
    def forward_process(extended_input_ids, p_mask, tok_idx_ext, labels):

        B, L = p_mask.shape
        L0    = start_pos
        L1    = L - L0
        device = extended_input_ids.device

        attention_mask = basic_block_attention.clone()
        attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
        attention_mask = process_pad(attention_mask, extended_input_ids)

        labels_trim = labels[:, L0:].clone()                             # (B, L1)
        resp_keep   = p_mask[:, L0:]                                        # (B, L1) 
        labels_trim[~resp_keep] = -100   

        out = model(
            input_ids      = extended_input_ids,
            attention_mask = attention_mask,
            position_ids   = tok_idx_ext,
            labels         = labels_trim,    
            logits_to_keep = L1,            
            return_dict    = True,
            use_cache      = False,
        )

        loss_lm = out.loss


        tokens_kept = (labels_trim != -100).sum()
        if tokens_kept > 0:
            loss_lm = loss_lm * (tokens_kept.to(loss_lm.dtype) / B)

        
        return loss_lm
    '''
    

    def forward_process(extended_input_ids, p_mask, tok_idx_ext, labels):

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
        logp_tok  = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)     # (B, T)
        loss_lm = - (logp_tok * p_mask).sum(dim=1) / L1

        loss_lm = loss_lm.sum() / B
        
        return loss_lm






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

            data_time_m.update(time.time() - end)


            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)

            loss_lm = forward_process(
                    extended_input_ids=extended_input_ids,
                    p_mask=p_mask,
                    tok_idx_ext=tok_idx_ext,
                    labels=labels
                )
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
