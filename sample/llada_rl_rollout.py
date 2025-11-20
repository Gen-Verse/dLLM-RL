from __future__ import annotations
import math, json, os, time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from jinja2 import Template
import torch
from termcolor import cprint
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from llada.modeling_llada import LLaDAModelLM
import multiprocessing as mp

from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    noise = (- torch.log(noise)) ** temperature
    return logits.exp() / noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens



# ──────────────────────────── return type ────────────────────────────────
@dataclass
class DiffusionOutput:
    sequences: torch.Tensor               # final result  (B, L_total)  (GPU)
    history:   List[torch.Tensor]         # all intermediate x (CPU)
    nfe:       int






@torch.no_grad()
def generate_with_prefix_cache(
        model, prompt,
        steps, gen_length, block_length, temperature,
        target, mask_id, further_horizon, use_cache, unmask_threshold
    ) -> DiffusionOutput:

    cgws = further_horizon
    B, L0 = prompt.shape
    x = torch.full((B, L0 + gen_length), mask_id, dtype=torch.long, device=prompt.device)
    max_length = L0 + gen_length
    x[:, :L0] = prompt
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (i < rem) for i in range(num_blocks)]

    nfe = 0
    hist: List[torch.Tensor] = []

    for blk in range(num_blocks):
        s, e = L0 + blk * block_length, L0 + (blk + 1) * block_length

        if cgws is not None:
            window_end  = max_length if cgws is None else min(e + cgws, max_length)
            window_slice = slice(s, window_end)
        
        cur_steps = steps_per_block[blk]
        num_transfer = get_num_transfer_tokens((x[:, s:e] == mask_id), cur_steps)

        # first full forward to build prefix cache
        if use_cache:
            out = model(x, use_cache=True)
            pkv = out.past_key_values
            # chop prefix out of past_kv to keep cache small
            new_pkv = tuple(
                tuple(t[:, :, :s] for t in layer) for layer in pkv
            )
            pkv = new_pkv
        else:
            out = model(x, use_cache=False)
        
        mask_all = (x == mask_id)
        mask_all[:, e:] = 0

        x0, tr_idx = get_transfer_index(
            out.logits, temperature, target, mask_all,
            x, num_transfer[:, 0], unmask_threshold)
        x[tr_idx] = x0[tr_idx]
        hist.append(x.clone().cpu())
        nfe += 1

        i = 1
        while True:
            nfe += 1
            if cgws is not None:
                mask_blk = (x[:, window_slice] == mask_id)
            else:
                mask_blk = (x[:, s:] == mask_id)
            mask_blk[:, block_length:] = 0

            if use_cache:
                if cgws is not None:
                    logits = model(x[:, window_slice], past_key_values=pkv, use_cache=True).logits
                    x0, tr_idx = get_transfer_index(
                        logits, temperature, target,
                        mask_blk, x[:, window_slice], num_transfer[:, i], unmask_threshold)
                    x[:, window_slice][tr_idx] = x0[tr_idx]
                else:
                    logits = model(x[:, s:], past_key_values=pkv, use_cache=True).logits
                    x0, tr_idx = get_transfer_index(
                        logits, temperature, target,
                        mask_blk, x[:, s:], num_transfer[:, i], unmask_threshold)
                    x[:, s:][tr_idx] = x0[tr_idx]
            else:
                logits = model(x, use_cache=False).logits
                logits = logits[:, s:]
                x0, tr_idx = get_transfer_index(
                    logits, temperature, target,
                    mask_blk, x[:, s:], num_transfer[:, i], unmask_threshold)
                x[:, s:][tr_idx] = x0[tr_idx]
            
            hist.append(x.clone().cpu())

            if (x[:, s:e] == mask_id).sum() == 0:
                break
            i += 1

    return DiffusionOutput(sequences=x, history=hist, nfe=nfe)




def get_transfer_index(logits, temperature, target, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if target == 'confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif target == 'margin_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        top2 = torch.topk(p, 2, dim=-1).values            # (b, l, 2)
        x0_p = top2[..., 0] - top2[..., 1]                # Δ(top1, top2)
    elif target == 'neg_entropy':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = -torch.sum(p * torch.log(p + 1e-10), dim=-1)  # –entropy
    elif target == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(target)
    
    x0 = torch.where(mask_index, x0, x)
    
    if threshold is not None:
        selected = mask_index & (x0_p >= threshold)  # (B, T)

        has_mask = mask_index.any(dim=-1)               # (B,)
        none_sel = (~selected.any(dim=-1)) & has_mask   # (B,)
        if none_sel.any():
            masked_scores = x0_p.masked_fill(~mask_index, float("-inf"))
            best_idx = masked_scores.argmax(dim=-1)     # (B,)
            rows = torch.nonzero(none_sel, as_tuple=False).squeeze(-1)
            selected[rows, best_idx[rows]] = True

        return x0, selected

    confidence = x0_p.masked_fill(~mask_index, float("-inf"))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        k = int(num_transfer_tokens[j].item() if torch.is_tensor(num_transfer_tokens[j]) else num_transfer_tokens[j])
        if k <= 0:
            continue
        _, sel = torch.topk(confidence[j], k=k)
        transfer_index[j, sel] = True
    return x0, transfer_index


import random 
def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list


# obtain prompt
def get_prompt(data_i):
    return Template(system_prompts, keep_trailing_newline=True).render(problem = data_i["question"])


def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)          # last \boxed{
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1                    # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:       # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def denoise_step_map(history, mask_id: int, sample_idx: int = 0):
    L = history[0].shape[1]        
    step_map = torch.zeros(L, dtype=torch.long)
    prev = torch.full((L,), mask_id, dtype=torch.long)

    for t, snap in enumerate(history, start=0): 
        cur = snap[sample_idx]        
        changed = (prev == mask_id) & (cur != mask_id)
        step_map[changed] = t
        prev = cur
        if (step_map == 0).sum() == 0:     
            break
    return step_map



from tqdm import tqdm

def worker(pretrained_model, rank, prompts, orig_idx, seq_dict, step_dict, batch_size, config):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # load model once
    model_gpu = (LLaDAModelLM
                 .from_pretrained(pretrained_model,
                                  trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)
                 .to(device)
                 .eval())

    tokenizer_gpu = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    # process in chunks of `batch_size`
    for start in tqdm(range(0, len(prompts), batch_size),
                      desc=f"GPU {rank}", position=rank, leave=True):
        batch_prompts = prompts[start:start+batch_size]
        batch_idxs    = orig_idx[start:start+batch_size]

        # tokenize & move to GPU
        enc = tokenizer_gpu(batch_prompts,
                            padding=True, #truncation=True,
                            return_tensors="pt", padding_side="left")
        input_ids = enc["input_ids"].to(device)

        mask_id = tokenizer_gpu.encode('<|mdm_mask|>')[0]

        if config.rollout.use_cache == False:
            config.rollout.further_horizon = None
        
        if config.rollout.remasking_strategy == "low_confidence_static":
            unmask_threshold = None
        else:
            unmask_threshold = config.rollout.dynamic_threshold

        # generate_with_prefix_cache
        out = generate_with_prefix_cache(
            model_gpu, input_ids,
            steps=config.rollout.steps, gen_length=config.rollout.max_gen_length,
            block_length=config.rollout.block_size, temperature=config.rollout.temperature,
            target=config.rollout.target, mask_id=mask_id, further_horizon=config.rollout.further_horizon,
            use_cache=config.rollout.use_cache, unmask_threshold = unmask_threshold
        )
        out.sequences = out.sequences.cpu()
        torch.cuda.empty_cache()

        # decode
        seq_ids = out.sequences[:, input_ids.shape[1]:].tolist()
        texts  = tokenizer_gpu.batch_decode(
            seq_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        
        
        
        # compute and store step maps
        for i, idx in enumerate(batch_idxs):
            # extract step map for sample i in this batch
            m = denoise_step_map(out.history, mask_id=mask_id, sample_idx=i)
            step_map = m[input_ids.shape[1]:].tolist()
            seq_dict[idx]  = texts[i]
            step_dict[idx] = step_map

        # free unused GPU cache
        torch.cuda.empty_cache()


def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node 
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]



def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output


if __name__ == "__main__":

    config = get_config()

    mp.set_start_method("spawn", force=True)

    k_sample = config.rollout.num_response_per_task
    batch_size = config.rollout.batch_size
    
    project_name = config.experiment.project
    
    system_prompts = """<|startoftext|><|start_header_id|>user<|end_header_id|>You need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
    
    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name




    code_task = False
    if config.experiment.function == "train":
        dataset = config.dataset.train_dataset
        k_sample = config.rollout.num_response_per_task
        batch_size = config.rollout.batch_size

        if config.dataset.data_type == "code":
            code_task = True
            system_prompts_function = '''<|startoftext|><|start_header_id|>user<|end_header_id|>{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n'''
            system_prompts_stdio = '''<|startoftext|><|start_header_id|>user<|end_header_id|>This is the problem:\n{{problem}}\n You should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n'''


        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
        
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        if config.evaluation.data_type == "code":
            code_task = True
            system_prompts_function = '''<|startoftext|><|start_header_id|>user<|end_header_id|>{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n'''
            system_prompts_stdio = '''<|startoftext|><|start_header_id|>user<|end_header_id|>This is the problem:\n{{problem}}\n You should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n'''
        
        k_sample = config.evaluation.num_response_per_task
        batch_size = config.evaluation.batch_size

        config.rollout.steps = config.evaluation.steps
        config.rollout.temperature = config.evaluation.temperature
        config.rollout.target = config.evaluation.target
        config.rollout.block_size = config.evaluation.block_size
        config.rollout.use_cache = config.evaluation.use_cache
        config.rollout.further_horizon = config.evaluation.further_horizon
        config.rollout.remasking_strategy = config.evaluation.remasking_strategy
        config.rollout.dynamic_threshold = config.evaluation.dynamic_threshold
        config.rollout.target = config.evaluation.target

        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset



    with open("../data/" + dataset + ".json", 'r') as f:
        data = json.load(f)
    #data = [data[i] for i in range(8)]
    
    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    if num_node > 1:
        if config.experiment.function == "train":
            random.shuffle(data)
        data = get_data_chunk(data, num_node, node_index)
    

    if config.experiment.function == "train":
        random_select_num = config.rollout.num_task_per_step
        random_select_num = int(random_select_num / num_node)
        random_select_num = min(random_select_num, len(data))
        data = random_select(data, random_select_num)
    num = len(data)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)




    # initialization
    generation_prompts = []
    prefix_list = []
    index_list = []
    for i in range(num):
        # preprocess
        if code_task:
            if data[i]["test_method"] == "stdio":
                system_prompts = system_prompts_stdio
                prefix_list = prefix_list + [None] * k_sample
            else:
                system_prompts = system_prompts_function + data[i]["prefix"]
                prefix_list = prefix_list + [data[i]["prefix"]] * k_sample
        generation_prompts = generation_prompts + [get_prompt(data[i])] * k_sample
        index_list = index_list + [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = get_prompt(data[i])





    # --------------------------- 1. shuffle --------------------------
    cprint("start generation...", "green")

    all_prompts = generation_prompts
    N = len(all_prompts)

    shuffled_idx     = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]

    # --------------------- 2. split to each GPU ----------------------
    n_gpu = torch.cuda.device_count()
    assert n_gpu > 1, "need >=2 GPUs for parallel inference"

    def split_even(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

    prompt_chunks = split_even(shuffled_prompts, n_gpu)
    idx_chunks    = split_even(shuffled_idx,     n_gpu)

    

    # ------------------- 4. launch all workers -----------------------
    manager    = mp.Manager()
    seq_dict   = manager.dict()   # {shuffled_pos: text}
    step_dict  = manager.dict()   # {shuffled_pos: step_map}
    procs = []

    for rk in range(n_gpu):
        p = mp.Process(target=worker,
                    args=(pretrained_model, rk,
                            prompt_chunks[rk],
                            idx_chunks[rk],
                            seq_dict,
                            step_dict,
                            batch_size,
                            config))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # ------------------- 5. restore original order -------------------
    restored_outputs    = [seq_dict[i]  for i in range(N)]
    restored_step_maps  = [step_dict[i] for i in range(N)]

    cprint("generation job done!", "green")














    import re

    # calculate the response length (ignoring repeated <|endoftext|> tokens)
    def get_token_lengths(strings, tokenizer):
        pad_token = tokenizer.pad_token

        escaped = re.escape(pad_token)
        pattern = rf"(?:{escaped})+"
        remove_pattern = escaped

        collapse_re = re.compile(pattern)

        lengths = []
        for s in strings:
            s_clean = collapse_re.sub(lambda _: pad_token if isinstance(pad_token, str) else '', s)
            s_clean = re.sub(remove_pattern, '', s_clean)
            lengths.append(len(tokenizer.encode(s_clean, add_special_tokens=False)))
        return lengths

    response_length = get_token_lengths(restored_outputs, tokenizer)
    mean_response_length = sum(response_length) / len(response_length)




    # process generated codes
    i = 0
    for full_output in restored_outputs:
        if code_task:
            if data[int(i/k_sample)]["test_method"] == "function":
                extracted_output = extract_code(prefix_list[i] + full_output)
            elif data[int(i/k_sample)]["test_method"] == "stdio":
                extracted_output = extract_code(full_output)
        else:
            extracted_output = extract_final_boxed_answer(full_output)
        index_i = index_list[i]
        data[index_i]["full_output"].append(full_output)
        data[index_i]["step_map"].append(restored_step_maps[i])
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])
        i += 1

    # output the data
    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
