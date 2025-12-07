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
from accelerate.utils import set_seed


from train.utils import get_config, flatten_omega_conf, AverageMeter

from models import MMadaConfig, MMadaModelLM
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import Dataset, DataLoader




try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")




class TrainDataset(Dataset):
    def __init__(self, inputs, labels, pmasks, adv):
        self.inputs   = inputs
        self.labels   = labels
        self.pmasks   = pmasks.to(torch.bool)
        self.adv      = adv
        L_raw      = inputs.shape[1]
        self.logp_old_tok = torch.full(
            (len(inputs), L_raw), 
            float('-inf')
        )
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return (
            idx,                         
            self.inputs[idx],
            self.labels[idx],
            self.pmasks[idx],
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

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    SOI_ID = 126084
    EOI_ID = 126085
    MMU_ID = 126089
    IPAD_ID = 126093
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)
    
    model = MMadaModelLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)
    model = model.to(accelerator.device)

    mask_id = tokenizer.encode('<|mdm_mask|>')[0]
    pad_id = tokenizer.encode('<|endoftext|>')[0]

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

    

    @torch.no_grad()
    def prepare_inputs_and_labels_for_text(
        prompt, response, step_map, reward, eps=1e-3, mask_id=mask_id
    ):
        input_ids_lm, labels_lm, start_pos, drop_num = uni_prompting((prompt, response))
        
        B, L = input_ids_lm.shape
        max_gen_len = config.training.max_gen_length
        if max_gen_len + start_pos < L:
            L_after = start_pos + max_gen_len
        else:
            L_after = L
        input_ids_lm = input_ids_lm[:, :L_after]
        labels_lm = labels_lm[:, :L_after]
    
        
        lower = config.training.lower_p
        upper = config.training.upper_p


        if config.training.method == "TraceRL":
            noisy_list, label_list, pmask_list, reward_list = [], [], [], []

            device = input_ids_lm.device
            B, L   = input_ids_lm.shape

            for b in range(B):
                
                order_list = list(step_map[b])
                order_list = collapse_k_unique(order_list, config.training.shrink)
                order = torch.as_tensor(order_list, device=device)
                order_full = torch.full((L_after,), -1, device=device)
                order_full[start_pos:] = order[: L_after - start_pos]
                uniq_steps = torch.unique(order_full[start_pos:], sorted=True)

                base_ids = input_ids_lm[b]

                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id)
                    pad_mask_b[:start_pos] = False
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L, dtype=torch.bool, device=device)
                    tail_pad_b       = torch.zeros(L, dtype=torch.bool, device=device)

                for step_val in uniq_steps:
                    tgt_mask = (order_full == step_val)
                    pmask_this = tgt_mask & ~tail_pad_b

                    if not pmask_this.any():
                        continue

                    noisy_ids = base_ids.clone()
                    mask_pos  = (order_full >= step_val)
                    noisy_ids[mask_pos] = mask_id

                    noisy_list.append(noisy_ids)
                    label_list.append(labels_lm[b])
                    pmask_list.append(pmask_this)
                    reward_list.append(reward[b])

            noisy_batch = torch.stack(noisy_list)
            labels_lm   = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)



        
        


            
        elif config.training.method == "random_masking":
            m = config.training.mask_times_per_sample
            B, L = input_ids_lm.shape
            device = input_ids_lm.device

            noisy_list, label_list, pmask_list, reward_list = [], [], [], []
            for b in range(B):
                base_ids  = input_ids_lm[b]
                label_ids = labels_lm[b]
                rwd       = reward[b]

                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id)
                    pad_mask_b[:start_pos] = False    
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L, dtype=torch.bool, device=device)
                    tail_pad_b       = torch.zeros(L, dtype=torch.bool, device=device)

                for _ in range(m):
                    t = (upper - lower) * torch.rand(1, device=device) + lower
                    rand_mask = torch.rand(L, device=device) < t
                    rand_mask[:start_pos] = False
                    rand_mask = rand_mask & ~tail_pad_b

                    if not rand_mask.any():
                        continue

                    noisy_ids = base_ids.clone()
                    noisy_ids[rand_mask]  = mask_id
                    noisy_ids[tail_pad_b] = mask_id  

                    noisy_list.append(noisy_ids)
                    label_list.append(label_ids)
                    pmask_list.append(rand_mask)
                    reward_list.append(rwd)

            noisy_batch = torch.stack(noisy_list)
            labels_lm   = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)
        




        elif config.training.method == "coupled":
            m      = config.training.mask_times_per_sample
            B, L   = input_ids_lm.shape
            device = input_ids_lm.device

            noisy_list, label_list, pmask_list, reward_list = [], [], [], []
            for b in range(B):
                base_ids  = input_ids_lm[b]
                label_ids = labels_lm[b]
                rwd       = reward[b]

                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id)
                    pad_mask_b[:start_pos] = False
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L, dtype=torch.bool, device=device)
                    tail_pad_b       = torch.zeros(L, dtype=torch.bool, device=device)

                for _ in range(m):
                    t = (upper - lower) * torch.rand(1, device=device) + lower
                    rand_mask = torch.rand(L, device=device) < t
                    rand_mask[:start_pos] = False

                    comp_mask = torch.zeros(L, device=device, dtype=torch.bool)
                    comp_mask[start_pos:] = ~rand_mask[start_pos:]

                    rand_mask  = rand_mask  & ~tail_pad_b
                    comp_mask  = comp_mask  & ~tail_pad_b

                    if rand_mask.any():
                        noisy_rand = base_ids.clone()
                        noisy_rand[rand_mask] = mask_id
                        noisy_rand[tail_pad_b] = mask_id
                        noisy_list.append(noisy_rand)
                        label_list.append(label_ids)
                        pmask_list.append(rand_mask)
                        reward_list.append(rwd)

                    if comp_mask.any():
                        noisy_comp = base_ids.clone()
                        noisy_comp[comp_mask] = mask_id
                        noisy_comp[tail_pad_b] = mask_id
                        noisy_list.append(noisy_comp)
                        label_list.append(label_ids)
                        pmask_list.append(comp_mask)
                        reward_list.append(rwd)

            noisy_batch = torch.stack(noisy_list)
            labels_lm   = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)
        

        valid_rows = p_mask.any(dim=1)
        noisy_batch = noisy_batch[valid_rows]
        labels_lm   = labels_lm[valid_rows]
        p_mask      = p_mask[valid_rows]
        keep_idx = torch.where(valid_rows)[0].tolist()
        reward_list = [reward_list[i] for i in keep_idx]

            
        
        return noisy_batch, labels_lm, p_mask, reward_list, start_pos, drop_num
    
    @torch.no_grad()
    def prepare_inputs_and_labels_for_mmu(
        prompt_list, response_list, step_map_list, reward_list, image_token_ids_list, eps=1e-3, mask_id=mask_id
    ):
        input_ids_batch = []
        labels_batch = []
        start_pos_raw_list = []
        orig_len_list = []
        drop_num = 0
        end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        pad_id_local = tokenizer.eos_token_id
        for p_str, resp_str, step_map, img_ids in zip(prompt_list, response_list, step_map_list, image_token_ids_list):
            chat_ids = tokenizer([p_str], add_special_tokens=False)["input_ids"][0]
            chat_ids = torch.tensor(chat_ids, dtype=torch.long)
            img_ids_t = torch.tensor(img_ids, dtype=torch.long)
            mmu = torch.tensor([MMU_ID, SOI_ID], dtype=torch.long)
            eoi = torch.tensor([EOI_ID], dtype=torch.long)
            resp_ids = tokenizer([resp_str], add_special_tokens=False)["input_ids"][0]
            resp_ids = torch.tensor(resp_ids, dtype=torch.long)
            full = torch.cat([mmu, img_ids_t, eoi, chat_ids, resp_ids], dim=0)  # [T]
            chat = torch.tensor(chat_ids, dtype=torch.long)
            pos = (chat == end_header_id).nonzero(as_tuple=False)
            if pos.numel() == 0:
                s_pos = (2 + img_ids_t.size(0) + 1 + chat.size(0))
            else:
                last = pos[-1].item()
                s_pos = 2 + img_ids_t.size(0) + 1 + (last + 1)
            labels = full.clone()
            labels[:s_pos] = -100
            input_ids_batch.append(full)
            labels_batch.append(labels)
            start_pos_raw_list.append(s_pos)
            orig_len_list.append(full.size(0))
        max_len = max(t.size(0) for t in input_ids_batch)
        def left_pad_to(x, L, pad_id):
            out = torch.full((L,), pad_id, dtype=torch.long)
            out[-x.size(0):] = x
            return out
        input_ids_lm = torch.stack([left_pad_to(t, max_len, pad_id_local) for t in input_ids_batch], dim=0)
        labels_lm    = torch.stack([left_pad_to(t, max_len, -100) for t in labels_batch], dim=0)
        # adjust start_pos for left padding
        pad_shifts = [max_len - Lorig for Lorig in orig_len_list]
        start_pos_list = [int(sp + sh) for sp, sh in zip(start_pos_raw_list, pad_shifts)]
        B, L = input_ids_lm.shape
        lower = config.training.lower_p
        upper = config.training.upper_p
        if config.training.method == "TraceRL":
            noisy_list, label_list, pmask_list, reward_list_new = [], [], [], []
            for b in range(B):
                base_ids = input_ids_lm[b]
                s_pos_b = start_pos_list[b]
                order_list = list(step_map_list[b])
                order_list = collapse_k_unique(order_list, config.training.shrink)
                order = torch.as_tensor(order_list)
                order_full = torch.full((L,), -1, dtype=torch.long)
                order_full[s_pos_b : s_pos_b + len(order)] = order[: L - s_pos_b]
                uniq_steps = torch.unique(order_full[s_pos_b : s_pos_b + len(order)], sorted=True)
                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id_local)
                    pad_mask_b[:s_pos_b] = False
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L, dtype=torch.bool)
                    tail_pad_b       = torch.zeros(L, dtype=torch.bool)

                start_len = len(noisy_list)
                dbg_cnt = 0
                dbg_written_clean = False
                for step_val in uniq_steps:
                    tgt_mask = (order_full == step_val)
                    pmask_this = tgt_mask & ~tail_pad_b
                    if not pmask_this.any():
                        continue
                    noisy_ids = base_ids.clone()
                    mask_pos  = (order_full >= step_val)
                    noisy_ids[mask_pos] = mask_id

                    # debug write: save clean and noisy token ids to txt (one token id per line)
                    if b < 2 and dbg_cnt < 2:
                        os.makedirs(project_name, exist_ok=True)
                        # clean (original, without padding) token ids for this sample's assistant portion
                        clean_ids = input_ids_batch[b][start_pos_raw_list[b]:].tolist()
                        clean_path = os.path.join(project_name, f"debug_sample{b}_clean.txt")
                        with open(clean_path, "w", encoding="utf-8") as cf:
                            for tok in clean_ids:
                                cf.write(str(int(tok)) + "\n")
                        # noisy: write masked sequence tail (assistant portion, after left-padding adjustment)
                        noisy_ids_tail = noisy_ids[start_pos_list[b]:].tolist()
                        noisy_path = os.path.join(project_name, f"debug_sample{b}_noisy_{dbg_cnt}.txt")
                        with open(noisy_path, "w", encoding="utf-8") as nf:
                            for tok in noisy_ids_tail:
                                nf.write(str(int(tok)) + "\n")
                        dbg_cnt += 1

                    noisy_list.append(noisy_ids)
                    label_list.append(labels_lm[b])
                    pmask_list.append(pmask_this)
                    reward_list_new.append(reward_list[b])

                if len(noisy_list) == start_len:
                    valid = (~tail_pad_b).clone()
                    valid[:s_pos_b] = False
                    if valid.any():
                        first_idx = torch.nonzero(valid, as_tuple=False)[0, 0]
                        noisy_ids = base_ids.clone()
                        noisy_ids[first_idx] = mask_id
                        pmask_this = torch.zeros(L, dtype=torch.bool)
                        pmask_this[first_idx] = True

                        noisy_list.append(noisy_ids)
                        label_list.append(labels_lm[b])
                        pmask_list.append(pmask_this)
                        reward_list_new.append(reward_list[b])

            noisy_batch = torch.stack(noisy_list)
            labels_out  = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)
            reward_out  = reward_list_new
        elif config.training.method == "random_masking":
            m = config.training.mask_times_per_sample
            noisy_list, label_list, pmask_list, reward_list_new = [], [], [], []
            for b in range(B):
                base_ids = input_ids_lm[b]
                label_ids = labels_lm[b]
                s_pos_b   = start_pos_list[b]
                rwd = reward_list[b]
                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id_local)
                    pad_mask_b[:s_pos_b] = False
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L, dtype=torch.bool)
                    tail_pad_b       = torch.zeros(L, dtype=torch.bool)
                for _ in range(m):
                    t = (upper - lower) * torch.rand(1) + lower
                    rand_mask = torch.rand(L) < t
                    rand_mask[:s_pos_b] = False
                    rand_mask = rand_mask & ~tail_pad_b

                    if not rand_mask.any():
                        valid = (~tail_pad_b).clone()
                        valid[:s_pos_b] = False
                        if valid.any():
                            first_idx = torch.nonzero(valid, as_tuple=False)[0, 0]
                            rand_mask[first_idx] = True
                        else:
                            continue

                    noisy_ids = base_ids.clone()
                    noisy_ids[rand_mask]  = mask_id
                    noisy_ids[tail_pad_b] = mask_id
                    noisy_list.append(noisy_ids)
                    label_list.append(label_ids)
                    pmask_list.append(rand_mask)
                    reward_list_new.append(rwd)

            noisy_batch = torch.stack(noisy_list)
            labels_out  = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)
            reward_out  = reward_list_new
        elif config.training.method == "coupled":
            m = config.training.mask_times_per_sample
            noisy_list, label_list, pmask_list, reward_list_new = [], [], [], []
            for b in range(B):
                base_ids = input_ids_lm[b]
                label_ids = labels_lm[b]
                s_pos_b   = start_pos_list[b]
                rwd = reward_list[b]
                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id_local)
                    pad_mask_b[:s_pos_b] = False
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L, dtype=torch.bool)
                    tail_pad_b       = torch.zeros(L, dtype=torch.bool)
                for _ in range(m):
                    t = (upper - lower) * torch.rand(1) + lower
                    rand_mask = torch.rand(L) < t
                    rand_mask[:s_pos_b] = False

                    comp_mask = torch.zeros(L, dtype=torch.bool)
                    comp_mask[s_pos_b:] = ~rand_mask[s_pos_b:]

                    rand_mask  = rand_mask  & ~tail_pad_b
                    comp_mask  = comp_mask  & ~tail_pad_b

                    if (not rand_mask.any()) and (not comp_mask.any()):
                        valid = (~tail_pad_b).clone()
                        valid[:s_pos_b] = False
                        if valid.any():
                            first_idx = torch.nonzero(valid, as_tuple=False)[0, 0]
                            rand_mask[first_idx] = True
                            comp_mask[first_idx] = False
                        else:
                            continue

                    if rand_mask.any():
                        noisy_rand = base_ids.clone()
                        noisy_rand[rand_mask] = mask_id
                        noisy_rand[tail_pad_b] = mask_id
                        noisy_list.append(noisy_rand)
                        label_list.append(label_ids)
                        pmask_list.append(rand_mask)
                        reward_list_new.append(rwd)

                    if comp_mask.any():
                        noisy_comp = base_ids.clone()
                        noisy_comp[comp_mask] = mask_id
                        noisy_comp[tail_pad_b] = mask_id
                        noisy_list.append(noisy_comp)
                        label_list.append(label_ids)
                        pmask_list.append(comp_mask)
                        reward_list_new.append(rwd)

            noisy_batch = torch.stack(noisy_list)
            labels_out  = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)
            reward_out  = reward_list_new
        valid_rows = p_mask.any(dim=1)
        noisy_batch = noisy_batch[valid_rows]
        labels_out  = labels_out[valid_rows]
        p_mask      = p_mask[valid_rows]
        keep_idx    = torch.where(valid_rows)[0].tolist()
        reward_out  = [reward_out[i] for i in keep_idx]
        return noisy_batch, labels_out, p_mask, reward_out, start_pos_list[0], drop_num


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
            input_ids  = batch["input_ids"].to(accelerator.device)
            labels     = batch["labels"].to(accelerator.device)

            L = labels.shape[1]
            logits = model(input_ids).logits                       # (B, L_ext, V)
            log_probs = F.log_softmax(logits[:, :L, :], dim=-1)    
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            tok_lp  = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, L)
            dataset.logp_old_tok[ids, :L] = tok_lp.float().cpu()
        
        accelerator.wait_for_everyone()

        model.train()
    

    
    def simple_collate(batch):
        idx, inp, lbl, msk, adv = zip(*batch)
        return {
            "ids":        torch.tensor(idx),
            "input_ids":  torch.stack(inp),
            "labels":     torch.stack(lbl),
            "p_mask_lm":  torch.stack(msk),
            "adv":        torch.stack(adv),
        }
    




    ##################################
    #       Preprocess data          #
    #################################
    logger.info("Preprocessing Data")



    dataset_pt = torch.load(Path(project_name) / "temp_data" / f"{config.dataset.optimization_data}-step{config.experiment.current_epoch}.pt", map_location="cpu")

    input_ids  = dataset_pt["extended_input_ids"]   # (N_rows, L_ext)
    p_mask_lm  = dataset_pt["p_mask"]               # (N_rows, L)
    labels     = dataset_pt["labels"]               # (N_rows, L)
    adv_t      = dataset_pt["adv"]                  # (N_rows, L)
    start_pos  = dataset_pt["meta"]["start_pos"]
    drop_num   = dataset_pt["meta"]["drop_num"]
    dataset_lm = TrainDataset(input_ids, labels, p_mask_lm, adv_t)








    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")


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
    
    
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )







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





    

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_lm)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {input_ids.shape[0]}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    
    
    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()

    

    def forward_process(input_ids, labels, p_mask_lm, adv, logp_old_tok):

        logits = model(input_ids).logits
        B, T, V = logits.shape
        adv = torch.as_tensor(adv, device=input_ids.device).detach()  # (B, L)
        L = labels.shape[1]
        log_probs = F.log_softmax(logits[:, :L, :], dim=-1) 
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0
        logp_new_tok  = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)     # (B, L)
        ratio   = torch.exp((logp_new_tok - logp_old_tok).clamp(-10.0, 10.0))   # (B, L)
        clipped = torch.clamp(ratio, 1 - config.training.eps, 1 + config.training.eps)            # (B, L)
        surrogate_tok = torch.min(ratio * adv, clipped * adv)  # (B, L)
        surrogate_tok = surrogate_tok * p_mask_lm
        num_mask = torch.clamp(p_mask_lm.sum(dim=1), min=1)
        surrogate_tok = surrogate_tok.sum(dim=1) / num_mask
        policy_loss = - (surrogate_tok.sum() / B)
        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        if config.training.beta > 0:
            kl_seq = (logp_new_tok - logp_old_tok)            # (B, L)
            kl_seq = torch.where(p_mask_lm, kl_seq, torch.zeros_like(kl_seq))
            if config.training.use_kl_estimator_k3:
                t = (-kl_seq).clamp(-10.0, 10.0)
                kl_seq = t.exp() - 1.0 + kl_seq
            kl_seq = (kl_seq * p_mask_lm).sum(dim=1) / num_mask
            kl_loss = config.training.beta * kl_seq.sum() / B
        total_loss = policy_loss + kl_loss
        return total_loss


        # KL penalty
        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        if config.training.beta > 0:
            kl_seq = logp_new_tok - logp_old_tok
            if config.training.use_kl_estimator_k3:
                kl_seq = (-kl_seq).exp() - 1.0 + kl_seq
            kl_seq = (kl_seq * p_mask_lm).sum(dim=1)
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
        grad_accum = config.training.gradient_accumulation_steps
        step = 0

        
        # assert accelerator.gradient_accumulation_steps == len(progress_bar)

        for step, batch in enumerate(progress_bar, start=1):
            # for loss calculation

            data_time_m.update(time.time() - end)

            input_ids = batch["input_ids"].to(accelerator.device)
            labels    = batch["labels"].to(accelerator.device)
            p_mask_lm = batch["p_mask_lm"].to(accelerator.device)
            adv = batch["adv"].to(accelerator.device)  # (B, L)
            old_lp = dataset_lm.logp_old_tok[batch["ids"].cpu()].to(accelerator.device)[:, :labels.size(1)]  # (B, L)
            loss_lm = forward_process(
                input_ids=input_ids,
                labels=labels,
                p_mask_lm=p_mask_lm,
                adv=adv,
                logp_old_tok=old_lp
            )
            loss_lm = loss_lm / grad_accum
            if step <= 10:
                print(loss_lm)
            accelerator.backward(loss_lm)

            if step % grad_accum == 0:
                logger.info("***** training step *****")
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(),
                                                config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                del input_ids, labels, p_mask_lm
                torch.cuda.empty_cache()
        # handle remaining gradients if any (cases: not divisible / steps < grad_accum)
        if step % grad_accum != 0:
            logger.info("***** final training step for residual grads *****")
            if config.training.max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(),
                                            config.training.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            # try to free memory similar to regular step
            try:
                del input_ids, labels, p_mask_lm
            except Exception:
                pass
            torch.cuda.empty_cache()

                


    accelerator.wait_for_everyone()

    # save checkpoint at the end of training
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)
    if config.experiment.current_epoch % config.experiment.save_every == 0:
        save_checkpoint(model, tokenizer, config, accelerator, f"epoch-{config.experiment.current_epoch}")

    accelerator.end_training()

    
    




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
        # 2) tokenizer
        tokenizer.save_pretrained(str(save_base / name))

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_base / name}")


if __name__ == "__main__":
    main()
