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
from vq.modeling_magvitv2 import MAGVITv2
from torchvision import transforms
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

SOI_ID = 126084
EOI_ID = 126085
MMU_ID = 126089
IPAD_ID = 126093
def default_image_transform(resolution=512):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),                           # [0,1]
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) # [-1,1]
    ])

def default_image_transform_pad(resolution=512, fill_color=(255, 255, 255)):
    def _pad_and_resize(img):
        img = img.convert('RGB')
        w, h = img.size
        if w == h:
            padded_image = img
        elif w < h:
            padding_needed = h - w
            padding_left = padding_needed // 2
            padding_right = padding_needed - padding_left
            pad_transform = transforms.Pad((padding_left, 0, padding_right, 0), fill=fill_color, padding_mode='constant')
            padded_image = pad_transform(img)
        else:
            padding_needed = w - h
            padding_top = padding_needed // 2
            padding_bottom = padding_needed - padding_top
            pad_transform = transforms.Pad((0, padding_top, 0, padding_bottom), fill=fill_color, padding_mode='constant')
            padded_image = pad_transform(img)
        return transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC)(padded_image)
    return transforms.Compose([
        transforms.Lambda(_pad_and_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

@torch.no_grad()
def encode_image_to_tokens(vq_model, image_pil, tokenizer, device, resolution=512, transform=None):
    if transform is None:
        transform = default_image_transform_pad()
    img_tensor = transform(image_pil).unsqueeze(0).to(device)     # [1,3,H,W]
    codes = vq_model.get_code(img_tensor)                         # [1, L], int
    offset = len(tokenizer)
    image_token_ids = (codes + offset).long().squeeze(0)          # [L]
    return image_token_ids


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
        while i < cur_steps:
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
    return Template(system_prompts).render(problem=data_i["question"])



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

def worker(pretrained_model, rank, prompts, orig_idx, data_idx, image_paths, seq_dict, step_dict, imgtok_dict, batch_size, config):
    from PIL import Image
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    # select model by base
    model_gpu = (LLaDAModelLM
                 .from_pretrained(pretrained_model,
                                  trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)
                 .to(device)
                 .eval())
    tokenizer_gpu = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    is_mmu = (config.model.model_base == "mmada" and (config.dataset.data_type == "mmu" or config.evaluation.data_type == "mmu"))
    if is_mmu:
        assert MAGVITv2 is not None, "MAGVITv2 is not installed. Please install or check PYTHONPATH."
        vq_model = MAGVITv2.from_pretrained(config.model.vq_model_path).to(device).eval()
        vq_model.requires_grad_(False)
    def left_pad_batch(tensors, pad_id):
        max_len = max(t.size(0) for t in tensors)
        out = torch.full((len(tensors), max_len), pad_id, dtype=torch.long)
        for i, t in enumerate(tensors):
            out[i, -t.size(0):] = t
        return out
    mask_id = tokenizer_gpu.encode('<|mdm_mask|>')[0]
    if config.rollout.use_cache == False:
        config.rollout.further_horizon = None
    unmask_threshold = None if config.rollout.remasking_strategy == "low_confidence_static" else config.rollout.dynamic_threshold
    # local cache to avoid repeated VQ for the same sample within this worker
    local_img_code_cache = {}
    # process in chunks
    for start in tqdm(range(0, len(prompts), batch_size),
                      desc=f"GPU {rank}", position=rank, leave=True):
        batch_prompts = prompts[start:start+batch_size]
        batch_idxs    = orig_idx[start:start+batch_size]
        batch_didx    = data_idx[start:start+batch_size]
        batch_ipaths  = image_paths[start:start+batch_size] if image_paths is not None else None
        if not is_mmu:
            enc = tokenizer_gpu(batch_prompts, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
        else:
            mmu_inputs = []
            for j, p in enumerate(batch_prompts):
                did = batch_didx[j]
                if did not in local_img_code_cache:
                    img_path = batch_ipaths[j]
                    if not os.path.isabs(img_path) and hasattr(config.dataset, "image_root") and config.dataset.image_root is not None:
                        img_path = os.path.join(config.dataset.image_root, img_path)
                    img = Image.open(img_path).convert("RGB")
                    img_codes = encode_image_to_tokens(vq_model, img, tokenizer_gpu, device, resolution=config.model.get("image_resolution", 512))
                    local_img_code_cache[did] = img_codes.detach().cpu()
                    imgtok_dict[did] = img_codes.detach().cpu().tolist()
                img_codes = local_img_code_cache[did].to(device)
                chat_ids = tokenizer_gpu([p], add_special_tokens=False)["input_ids"]
                chat_ids = torch.tensor(chat_ids[0], dtype=torch.long, device=device)  # [T_chat]
                mmu = torch.tensor([MMU_ID, SOI_ID], dtype=torch.long, device=device)   # [2]
                eoi = torch.tensor([EOI_ID], dtype=torch.long, device=device)           # [1]
                full = torch.cat([mmu, img_codes, eoi, chat_ids], dim=0)               # [T]
                mmu_inputs.append(full)
            pad_id = tokenizer_gpu.eos_token_id
            input_ids = left_pad_batch(mmu_inputs, pad_id=pad_id).to(device)
        out = generate_with_prefix_cache(
            model_gpu, input_ids,
            steps=config.rollout.steps, gen_length=config.rollout.max_gen_length,
            block_length=config.rollout.block_size, temperature=config.rollout.temperature,
            target=config.rollout.target, mask_id=mask_id, further_horizon=config.rollout.further_horizon,
            use_cache=config.rollout.use_cache, unmask_threshold=unmask_threshold
        )
        out.sequences = out.sequences.cpu()
        torch.cuda.empty_cache()
        seq_ids = out.sequences[:, input_ids.shape[1]:].tolist()
        texts  = tokenizer_gpu.batch_decode(seq_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for i, idx in enumerate(batch_idxs):
            m = denoise_step_map(out.history, mask_id=mask_id, sample_idx=i)
            step_map = m[input_ids.shape[1]:].tolist()
            seq_dict[idx]  = texts[i]
            step_dict[idx] = step_map
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

    if config.reward.answer_must_in_box:
        assert config.dataset.add_think_prompt == False, "answer_must_in_box must be False when add_think_prompt is True"
    
    if config.reward.answer_must_in_box:
        system_prompts = """<|startoftext|><|start_header_id|>user<|end_header_id|>\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
    elif config.dataset.add_think_prompt:
        system_prompts = """<|startoftext|><|start_header_id|>user<|end_header_id|>\nYou should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
    else:
        system_prompts = """<|startoftext|><|start_header_id|>user<|end_header_id|>{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
    
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


        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset + f"-step{config.experiment.current_epoch}"
        
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

        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset + f"-step{config.experiment.current_epoch}"



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

    if config.model.model_base == "mmada" and (config.dataset.data_type == "mmu" or config.evaluation.data_type == "mmu"):
        image_paths = []
        for i in range(num):
            # 每个样本重复 k_sample 次
            image_paths += [data[i]["image"]] * k_sample
    else:
        image_paths = None





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
    data_idx_full = [index_list[i] for i in shuffled_idx]
    data_idx_chunks = split_even(data_idx_full,  n_gpu)
    if image_paths is not None:
        image_paths_full = [image_paths[i] for i in shuffled_idx]
        image_chunks = split_even(image_paths_full, n_gpu)
    else:
        image_chunks = [None] * n_gpu
    

    # ------------------- 4. launch all workers -----------------------
    manager    = mp.Manager()
    seq_dict   = manager.dict()   # {shuffled_pos: text}
    step_dict  = manager.dict()   # {shuffled_pos: step_map}
    imgtok_dict = manager.dict()  # {data_idx: image_token_ids(list[int])}
    procs = []

    for rk in range(n_gpu):
        p = mp.Process(target=worker,
                    args=(pretrained_model, rk,
                            prompt_chunks[rk],
                            idx_chunks[rk],
                            data_idx_chunks[rk],
                            image_chunks[rk],
                            seq_dict,
                            step_dict,
                            imgtok_dict,
                            batch_size,
                            config))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # ------------------- 5. restore original order -------------------
    restored_outputs    = [seq_dict[i]  for i in range(N)]
    restored_step_maps  = [step_dict[i] for i in range(N)]
    # 为每个 data[index] 写回 image_token_ids（仅MMU）
    if (config.model.model_base == "mmada" and (config.dataset.data_type == "mmu" or config.evaluation.data_type == "mmu")):
        for di in range(num):
            if di in imgtok_dict:
                data[di]["image_token_ids"] = imgtok_dict[di]

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
            if config.reward.answer_must_in_box:
                extracted_output = extract_final_boxed_answer(full_output)
            else:
                if "</think>" in full_output:
                    extracted_output = full_output.split("</think>")[1]
                else:
                    extracted_output = full_output
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
