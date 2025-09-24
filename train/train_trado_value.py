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
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
import torch.nn as nn

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed



from transformers import AutoModelForCausalLM, AutoTokenizer
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
    def __init__(
        self,
        extended_input_ids, p_mask, tok_idx_ext, labels, reward,
        *,
        seq_ids,                  # (N_rows,)  The original sequence id (i.e., batch index b) for this row
        L0, L1,                   # Prompt length, response length
        step_map_all,             # (B, L1)    Step map for each original sequence (already shrunk)
        resp_input_ids_all,       # (B, L1)    Response token ids for each original sequence (used for pad/post_num)
        per_seq_reward,           # (B,)       Scalar reward for each original sequence
        pad_id, post_num          # Pad token id and post_num
    ):
        self.extended_input_ids = extended_input_ids  # (N_rows, L_ext)
        self.p_mask  = p_mask                         # (N_rows, L=L0+L1), only valid for the first L positions
        self.tok_idx_ext = tok_idx_ext               # (N_rows, L)
        self.labels  = labels                        # (N_rows, L)
        self.Return  = reward                        # (N_rows, L) Placeholder, will be overwritten with "return" later
        # --- Extra fields for aggregation ---
        self.seq_ids = torch.as_tensor(seq_ids, dtype=torch.long)        # (N_rows,)
        self.L0 = int(L0); self.L1 = int(L1)
        self.step_map_all = step_map_all.clone().cpu()                   # (B, L1)
        self.resp_input_ids_all = resp_input_ids_all.clone().cpu()       # (B, L1)
        self.per_seq_reward = torch.as_tensor(per_seq_reward, dtype=torch.float32)  # (B,)
        self.pad_id = int(pad_id); self.post_num = int(post_num) if post_num is not None else None

        # Old values (predicted by the model during inference)
        self.old_values = torch.full((len(extended_input_ids), p_mask.shape[1]), 0.0)
        # Advantage placeholder (to be filled later)
        self.adv = torch.full((len(extended_input_ids), p_mask.shape[1]), 0.0)

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            idx,
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
            self.Return[idx],   
        )





def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    project_name = config.experiment.project
    pretrained_model = "./" + project_name + "/ckpt/" + config.model.optimized_value_name

    from transformers import AutoConfig
    from train.init_trado_value_model import _get_value_model
    from models import SDARForCausalLM
    value_model_class = _get_value_model(SDARForCausalLM, "value_head")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
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

    
    
    
    
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)
    

    # calculate loss ourselves, needs logits，so aviod fuse CE
    if hasattr(value_model, "config"):
        value_model.config.fuse_cross_entropy = False   
    

    if config.training.gradient_checkpointing_enable:
        value_model.gradient_checkpointing_enable()
        if hasattr(value_model, "config"):
            value_model.config.use_cache = False
    else:
        value_model = value_model.to(accelerator.device)

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

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.value_learning_rate,
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
        idx, extended_input_ids, p_mask, tok_idx_ext, labels, Return = zip(*batch)     #Tensor(L)
        return {
            "ids":        torch.tensor(idx),
            "extended_input_ids":  torch.stack(extended_input_ids),
            "p_mask":  torch.stack(p_mask),
            "tok_idx_ext":  torch.stack(tok_idx_ext),
            "labels":  torch.stack(labels),
            "Return":     torch.stack(Return),
        }
    


    
    with open("./" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json", 'r') as f:
        dataset_load = json.load(f)
    #dataset_load = dataset_load[:2000]



    prompt_list = []
    response_list = []
    step_map_list = []
    reward_list = []
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
        reward_list.append(x["reward"])
    
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
            i_end   = min((bi + 1) * block_size, L1)         

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

        attn.masked_fill_(key_mask[:, None, None, :], 0)

        # aviod +-inf or none in forward
        A = attn[:, 0]  # (B, N, N)
        bad = (A.sum(dim=-1) == 0) & (torch.arange(A.size(1), device=A.device).unsqueeze(0) < start_pos)
        b, r = bad.nonzero(as_tuple=True)
        A[b, r, :] = 0; A[b, r, r] = 1 

        return attn
    

    

    def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
        """
        Vectorized selection for a single sample in one round.

        Purpose:
          - Choose, within each response block, the tokens whose step id equals
            the current minimum step id of that block.
          - Build:
              * pmask_b: a boolean mask over the first (L0+L1) tokens marking the
                selected response positions for this round.
              * extended_input_ids_b: the original sequence concatenated with a
                masked copy of the response segment where all positions with
                step_id >= block-min are set to `mask_id`.
              * new_step_map_b: the updated step_map where positions selected
                in this round are set to -1 (so they won't be selected again).
              * sel_step_tail: length-L1 tensor storing the step id for the
                positions selected in this round (all other positions are -1).

        Returns:
          (extended_input_ids_b, pmask_b, new_step_map_b, sel_step_tail, has_any)
          where has_any is True if any position was selected in this round,
          otherwise returns (None, None, step_map_b, None, False).
        """
        device = input_ids_b.device
        NB = (L1 + block_size - 1) // block_size # number of response blocks

        # Pad step_map to a multiple of block_size with -1 so we can view as [NB, block_size]
        step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
        step_pad[:L1] = step_map_b
        step_blk = step_pad.view(NB, block_size)

        # valid marks entries that belong to the real response (>= 0 means not yet selected)
        valid = step_blk.ge(0)
        big = torch.iinfo(step_blk.dtype).max # sentinel for masked invalid entries
        tmp = step_blk.masked_fill(~valid, big)
        min_vals, _ = tmp.min(dim=1, keepdim=True) # per-block current minimum step id

        # pmask_blk selects entries that are valid and equal to the block-wise minimum
        pmask_blk = step_blk.eq(min_vals) & valid
        if not pmask_blk.any():
            # Nothing left to select in any block: end this round
            return None, None, step_map_b, None, False

        # ge_mask_blk marks valid entries with step_id >= block minimum (used to mask in the tail copy)
        ge_mask_blk = step_blk.ge(min_vals) & valid

        # Flatten back to length-L1 tails
        pmask_tail = pmask_blk.view(-1)[:L1]
        ge_mask_tail = ge_mask_blk.view(-1)[:L1]

        # pmask over the first L0+L1 tokens (prompt + response); only response part is ever True
        pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
        pmask_b[L0:] = pmask_tail

        # Make a copy of the response tail and mask all positions with step_id >= block minimum
        tail = input_ids_b[L0:L0+L1].clone()
        tail[ge_mask_tail] = mask_id

        # Concatenate original (prompt+response) with the masked tail copy to form the extended input
        extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
        extended_input_ids_b[:L0+L1] = input_ids_b
        extended_input_ids_b[L0+L1:] = tail

        # sel_step_tail stores the step id for positions selected in this round; others are -1
        sel_step_tail = torch.full((L1,), -1, dtype=torch.long, device=device)
        sel_step_tail[pmask_tail] = step_map_b[pmask_tail]

        # Update step_map: mark selected positions as -1 so they are excluded in subsequent rounds
        new_step_map_b = step_map_b.clone()
        new_step_map_b[pmask_tail] = -1

        return extended_input_ids_b, pmask_b, new_step_map_b, sel_step_tail, True
    

    def collect_training_data(input_ids, step_map_list, reward):
        B, L = input_ids.shape
        L0 = start_pos
        L1 = L - L0
        block_size = config.training.block_size

        # shrink (in-place modification of step_map_list)
        for b in range(B):
            step_map_i = step_map_list[b]
            for j in range(int((L1 - 1) / block_size) + 1):
                s = j * block_size
                e = min(L1, (j + 1) * block_size)
                step_map_list[b][s:e] = collapse_k_unique(step_map_i[s:e], config.training.shrink)

        step_map = torch.as_tensor(step_map_list, dtype=torch.long)  # (B, L1)
        assert step_map.shape[1] == L1

        extended_input_ids_list, pmask_list = [], []
        reward_list_rows = []   # placeholder
        seq_ids_rows = []
        sel_step_tail_rows = [] # (row, L1) step ids selected in this round; others set to -1

        for b in range(B):
            step_b = step_map[b].clone()
            while True:
                out = one_round_vectorized(
                    input_ids_b=input_ids[b],
                    step_map_b=step_b,
                    L0=L0, L1=L1,
                    block_size=block_size,
                    mask_id=mask_id,
                )
                extended_b, pmask_b, step_b, sel_step_tail, has_any = out
                if not has_any:
                    break
                extended_input_ids_list.append(extended_b)
                pmask_list.append(pmask_b)
                reward_list_rows.append(reward[b])  # scalar, row-level; actual token assignment is done later
                seq_ids_rows.append(b)
                sel_step_tail_rows.append(sel_step_tail.cpu())  # only response segment (L1)

        extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
        p_mask = torch.stack(pmask_list, dim=0).to(torch.bool)
        seq_ids_rows = torch.as_tensor(seq_ids_rows, dtype=torch.long)
        sel_step_tail_rows = torch.stack(sel_step_tail_rows, dim=0)  # (N_rows, L1)

        #  post_num 
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
        tok_idx_ext = torch.cat([tok_idx, tok_idx_resp], dim=1)

        # Filter out rows where nothing was selected
        keep = p_mask.view(p_mask.size(0), -1).any(dim=1)
        idx_keep = keep.nonzero(as_tuple=True)[0]

        extended_input_ids = extended_input_ids[idx_keep]
        p_mask            = p_mask[idx_keep]
        tok_idx_ext       = tok_idx_ext[idx_keep]
        labels            = labels[idx_keep]
        seq_ids_rows      = seq_ids_rows[idx_keep]
        sel_step_tail_rows= sel_step_tail_rows[idx_keep]
        reward_rows       = [reward_list_rows[i] for i in idx_keep.tolist()]  

        # Initialize row-level “reward_mat” with all zeros
        reward_vec = torch.as_tensor(reward_rows, dtype=torch.float32, device=p_mask.device)
        reward_mat = torch.zeros_like(p_mask, dtype=torch.float32)

        # Extra return: for later aggregation
        resp_input_ids_all = input_ids[:, L0:L0+L1].clone().cpu()  # (B,L1)

        return (
            extended_input_ids, p_mask, tok_idx_ext, labels, reward_mat,
            seq_ids_rows, sel_step_tail_rows, step_map, resp_input_ids_all
        )

        

    
    (extended_input_ids, p_mask, tok_idx_ext, labels, rewards,        # rewards all 0, as place-holder
        seq_ids_rows, sel_step_tail_rows, step_map_all, resp_input_ids_all) = collect_training_data(input_ids_lm, step_map_list, reward_list)


    
    dataset_lm = TrainDataset(
        extended_input_ids, p_mask, tok_idx_ext, labels, rewards,
        seq_ids=seq_ids_rows,
        L0=start_pos, L1=(labels.shape[1]-start_pos),
        step_map_all=step_map_all,
        resp_input_ids_all=resp_input_ids_all,
        per_seq_reward=torch.as_tensor(reward_list, dtype=torch.float32),
        pad_id=pad_id, post_num=post_num
    )


    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
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
    value_model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        value_model, optimizer, lr_scheduler, train_dataloader_lm
    )


    import torch.nn.functional as F






    @torch.no_grad()
    def compute_old_value_parallel(
            accelerator,
            dataset,
            train_dataloader_lm,
            start_pos, pad_id,
            batch_size):

        value_model.eval()
        dl = train_dataloader_lm

        for batch in dl:
            ids        = batch["ids"]  # (b,)
            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)

            B, L = p_mask.shape
            L0 = start_pos
            L1 = L - L0
            device = extended_input_ids.device

            attention_mask = basic_block_attention.clone()
            attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
            attention_mask = process_pad(attention_mask, extended_input_ids)

            values = value_model(
                input_ids=extended_input_ids,
                attention_mask=attention_mask,
                position_ids=tok_idx_ext
            )
            values = torch.cat([values[:, :L0], values[:, L0 + L1 :]], dim=1)  # (B, L0+L1)
            values = torch.where(p_mask, values, torch.zeros_like(values))

            if accelerator.num_processes > 1:
                ids_dev   = ids.to(accelerator.device)
                ids_pad    = accelerator.pad_across_processes(ids_dev, dim=0, pad_index=-1)
                values_pad = accelerator.pad_across_processes(values,  dim=0)  

                ids_all    = accelerator.gather(ids_pad)
                values_all = accelerator.gather(values_pad)

                valid = ids_all.ne(-1)
                idx_cpu = ids_all[valid].long().cpu()
                vals_cpu= values_all[valid].float().cpu()

                dataset.old_values[idx_cpu] = vals_cpu
            else:
                dataset.old_values[ids] = values.float().cpu()

        accelerator.wait_for_everyone()
        value_model.train()



    #################################
    #             Inference         #
    #################################
    logger.info("***** Running inference *****")

    compute_old_value_parallel(
        accelerator,
        dataset_lm,
        train_dataloader_lm,
        start_pos=start_pos,
        pad_id=pad_id,
        batch_size=config.training.batch_size_lm,
    )






    #################################
    #             Inference         #
    #################################
    logger.info("***** Calculate Advantage and Return *****")



    def compute_returns_and_advantages_from_fragments(
        dataset: TrainDataset,
        gamma: float,
        lam: float,
        *,
        atol: float = 1e-5
    ):
        """
        Read dataset.old_values (nonzero only at p_mask positions),
        aggregate fragments from the same original sequence to reconstruct the
        full token-level V^{old}, compute token-level R_j / A_j based on the
        per-sequence step_map / RLVR reward assignment, and finally write
        the R/A for the “currently trainable tokens” back into
        dataset.Return / dataset.adv (nonzero only at p_mask positions).
        """
        L0, L1 = dataset.L0, dataset.L1
        B = dataset.step_map_all.size(0)
        N_rows = dataset.p_mask.size(0)

        # Pre-group the row indices belonging to each original sequence
        rows_by_seq = [[] for _ in range(B)]
        for row_idx in range(N_rows):
            s = int(dataset.seq_ids[row_idx].item())
            rows_by_seq[s].append(row_idx)

        # Clones for writing back later
        Return_mat = dataset.Return.clone()     # (N_rows, L)
        adv_mat    = dataset.adv.clone()
        old_vals   = dataset.old_values.clone() # (N_rows, L)

        # Process each original sequence
        for s in range(B):
            rows = rows_by_seq[s]
            if not rows:
                continue

            step_map_s = dataset.step_map_all[s].clone()          # (L1,)
            resp_ids_s = dataset.resp_input_ids_all[s].clone()    # (L1,)

            # Compute trainable mask (response segment):
            # not pad OR (pad but pad count <= post_num)
            is_pad = resp_ids_s.eq(dataset.pad_id)
            if dataset.post_num is None:
                trainable_mask = ~is_pad
            else:
                cum_pad = torch.cumsum(is_pad.int(), dim=0)
                trainable_mask = (~is_pad) | (is_pad & (cum_pad <= dataset.post_num))

            # RLVR reward is only given to tokens in the “last trainable trace step”
            # Find the maximum step id in the trainable region
            valid_steps = step_map_s[trainable_mask]
            assert valid_steps.numel() > 0, f"sequence {s}: no trainable tokens"
            last_step_id = int(valid_steps.max().item())

            # Token-level immediate reward r_j
            r_resp = torch.zeros(L1, dtype=torch.float32)
            r_resp[(step_map_s == last_step_id) & trainable_mask] = dataset.per_seq_reward[s].item()

            # Aggregate token-level V^{old} from fragment rows
            V_resp = torch.zeros(L1, dtype=torch.float32)
            filled = torch.zeros(L1, dtype=torch.bool)
            union_mask_resp = torch.zeros(L1, dtype=torch.bool)

            for row in rows:
                pm = dataset.p_mask[row]  # (L0+L1,)
                # p_mask is always False in the first L0
                tail_mask = pm[L0:]       # (L1,)
                if not tail_mask.any():
                    continue
                vals_row = old_vals[row, L0:L0+L1]  # (L1,)
                # Each position should only be selected once
                assert not filled[tail_mask].any(), \
                    f"sequence {s}: duplicated selection in fragments"
                V_resp[tail_mask] = vals_row[tail_mask]
                filled[tail_mask] = True
                union_mask_resp |= tail_mask

            # All trainable positions must be covered
            assert torch.all(union_mask_resp[trainable_mask]), \
                f"sequence {s}: some trainable tokens lack value predictions"

            # Build an ordered list of step ids (trainable only)
            uniq_steps = torch.unique(step_map_s[trainable_mask], sorted=True)
            S = uniq_steps.numel()
            step_to_rank = {int(uniq_steps[i].item()): i for i in range(S)}

            # Per-step r_t^* / V_t^{*,old}
            r_star = torch.zeros(S, dtype=torch.float32)
            V_star = torch.zeros(S, dtype=torch.float32)

            for sid in uniq_steps.tolist():
                sid = int(sid)
                mask = (step_map_s == sid) & trainable_mask
                r_star[step_to_rank[sid]] = r_resp[mask].mean() if mask.any() else 0.0
                V_star[step_to_rank[sid]] = V_resp[mask].mean() if mask.any() else 0.0

            # Backward recursion for R_t^*
            R_star = torch.zeros(S, dtype=torch.float32)
            for i in range(S-1, -1, -1):
                R_star[i] = r_star[i] + (gamma * R_star[i+1] if i+1 < S else 0.0)

            # TD residual and step-level GAE
            delta_star = torch.zeros(S, dtype=torch.float32)
            for i in range(S):
                v_next = V_star[i+1] if i+1 < S else 0.0
                delta_star[i] = r_star[i] - V_star[i] + gamma * v_next

            A_star = torch.zeros(S, dtype=torch.float32)
            for i in range(S-1, -1, -1):
                A_star[i] = delta_star[i] + (gamma * lam * A_star[i+1] if i+1 < S else 0.0)

            # Map back to tokens: R_j, A_j
            R_resp = torch.zeros(L1, dtype=torch.float32)
            A_resp = torch.zeros(L1, dtype=torch.float32)
            for pos in torch.nonzero(trainable_mask, as_tuple=False).flatten().tolist():
                sid = int(step_map_s[pos].item())
                i = step_to_rank[sid]
                rj = r_resp[pos]
                R_next = R_star[i+1] if i+1 < S else 0.0
                V_next = V_star[i+1] if i+1 < S else 0.0
                A_next = A_star[i+1] if i+1 < S else 0.0

                R_resp[pos] = rj + gamma * R_next
                A_resp[pos] = (rj - V_resp[pos]) + gamma * V_next + gamma * lam * A_next

            # Write back into fragment rows (nonzero only at p_mask positions)
            R_full = torch.zeros(L0+L1, dtype=torch.float32)
            A_full = torch.zeros(L0+L1, dtype=torch.float32)
            R_full[L0:] = R_resp
            A_full[L0:] = A_resp

            for row in rows:
                pm = dataset.p_mask[row]
                Return_mat[row][pm] = R_full[pm]
                adv_mat[row][pm]    = A_full[pm]

            # Assertion 1: when gamma=lambda=1,
            # R_j equals the sequence reward, and A_j = R_j - V_j^{old}
            if abs(gamma - 1.0) < 1e-8 and abs(lam - 1.0) < 1e-8:
                expected = dataset.per_seq_reward[s].item()
                for row in rows:
                    pm = dataset.p_mask[row]
                    R_row = Return_mat[row][pm]
                    V_row = old_vals[row][pm]
                    A_row = adv_mat[row][pm]
                    assert torch.allclose(R_row, torch.full_like(R_row, expected), atol=atol), \
                        f"gamma=lambda=1 check failed (R) at seq {s}, row {row}"
                    assert torch.allclose(A_row, R_row - V_row, atol=atol), \
                        f"gamma=lambda=1 check failed (A=R-V) at seq {s}, row {row}"
            
            # Assertion 2: when gamma=1, lambda=0,
            # A_j = r_j - V_j^{old} + V_{t_j+1}^{*,old}
            if abs(gamma - 1.0) < 1e-8 and abs(lam - 0.0) < 1e-8:
                # For each trainable position, construct “next-step mean V_next^{*,old}”
                V_next_resp = torch.zeros(L1, dtype=torch.float32)
                for sid in uniq_steps.tolist():
                    i = step_to_rank[int(sid)]
                    mask = (step_map_s == int(sid)) & trainable_mask
                    if i + 1 < S:
                        V_next_resp[mask] = V_star[i + 1]
                    # else remains 0 (for the last step)

                expected_A_resp = (r_resp - V_resp) + V_next_resp
                A_expected_full = torch.zeros(L0 + L1, dtype=torch.float32)
                A_expected_full[L0:] = expected_A_resp

                for row in rows:
                    pm = dataset.p_mask[row]
                    A_row = adv_mat[row][pm]
                    A_row_expected = A_expected_full[pm]
                    assert torch.allclose(A_row, A_row_expected, atol=atol), \
                        f"gamma=1, lambda=0 check failed (A=r-V+V_next*) at seq {s}, row {row}"

        # write back to data
        dataset.Return = Return_mat
        dataset.adv    = adv_mat
    

    gam = config.training.gam
    lam = config.training.lam
    compute_returns_and_advantages_from_fragments(dataset_lm, gam, lam, atol=1e-5)





    def save_dataset_tensors(dataset_lm, save_dir, name, accelerator, *,
                         start_pos: int, drop_num: int):
        from pathlib import Path, PurePath
        import time
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "extended_input_ids": dataset_lm.extended_input_ids,  # (N, L_ext)
            "p_mask":            dataset_lm.p_mask,               # (N, L)
            "tok_idx_ext":       dataset_lm.tok_idx_ext,          # (N, L)
            "labels":            dataset_lm.labels,               # (N, L)
            "adv":               dataset_lm.adv,                  # (N, L)
            "meta": {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "start_pos": int(start_pos),
                "drop_num":  int(drop_num),
            },
        }

        if accelerator.is_main_process:
            torch.save(payload, save_dir / f"{name}.pt")


    save_dataset_tensors(
        dataset_lm,                      #  extended_input_ids / p_mask / tok_idx_ext / labels / adv
        save_dir=Path(config.experiment.project) / "temp_data",
        name=f"{config.dataset.optimization_data}",  
        accelerator=accelerator,
        start_pos = start_pos,
        drop_num = drop_num
    )



    if config.experiment.current_epoch % config.experiment.train_value_every != 0:
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return




    #################################
    #             Training          #
    #################################
    

    

    
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training / inference data = {input_ids_lm.shape[0]}")

    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()







    def forward_process(extended_input_ids, p_mask, tok_idx_ext, Return, old_values):

        B, L = p_mask.shape
        L0    = start_pos
        L1    = L - L0
        device = extended_input_ids.device

        attention_mask = basic_block_attention.clone()
        attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
        attention_mask = process_pad(attention_mask, extended_input_ids)

        values = value_model(input_ids = extended_input_ids, attention_mask = attention_mask, position_ids = tok_idx_ext)
        values = torch.cat([values[:, :L0], values[:, L0 + L1 :]], dim=1)
        values = torch.where(p_mask, values, torch.zeros_like(values))

        v_clipped = old_values + (values - old_values).clamp(-config.training.eps, config.training.eps)
        loss_unclipped = (values - Return) ** 2
        loss_clipped   = (v_clipped - Return) ** 2

        loss = 0.5 * torch.maximum(loss_unclipped, loss_clipped) * p_mask

        loss = loss.sum(dim=1) / L1
        loss = loss.sum() / B

        return loss







    from tqdm.auto import tqdm

    loss_list = []

    for epoch in range(first_epoch, num_train_epochs):
        
        value_model.train()
        
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
            Return = batch["Return"].to(accelerator.device)
            old_values = dataset_lm.old_values[batch["ids"].cpu()].to(accelerator.device)

            loss_lm = forward_process(
                    extended_input_ids=extended_input_ids,
                    p_mask=p_mask,
                    tok_idx_ext=tok_idx_ext,
                    Return=Return,
                    old_values=old_values
                )
            loss_lm = loss_lm / accelerator.gradient_accumulation_steps
            

            if step < 10:
                print(loss_lm)


            accelerator.backward(loss_lm)

            if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(value_model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                torch.cuda.empty_cache()

            loss_list.append(loss_lm.detach().float().cpu().item())
            

                


    accelerator.wait_for_everyone()

    # save checkpoint at the end of training
    save_checkpoint(value_model, tokenizer, config, accelerator, config.model.optimized_value_name)
    if config.experiment.current_epoch % config.experiment.save_every == 0:
        save_checkpoint(value_model, tokenizer, config, accelerator, f"epoch-{config.experiment.current_epoch}-value")

    accelerator.end_training()

    from termcolor import cprint



    if accelerator.is_main_process:

        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + config.dataset.train_dataset

        def _mean(x): 
            return float(sum(x) / max(1, len(x))) 

        temp_len = 50
        first = loss_list[:temp_len]
        last  = loss_list[-temp_len:] if len(loss_list) >= temp_len else loss_list

        first_few_avg_loss = _mean(first)
        last_few_avg_loss  = _mean(last)
        avg_loss           = _mean(loss_list)

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




    
    


    
