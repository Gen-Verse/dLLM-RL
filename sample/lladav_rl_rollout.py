from __future__ import annotations
import math, json, os, time
from dataclasses import dataclass, asdict
from typing import List, Tuple
import numpy as np
from jinja2 import Template
import torch
from termcolor import cprint
import torch.nn.functional as F
from transformers import AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook, unregister_fast_dllm_hook
from llava.hooks import register_cache_LLaDA_V
from torchvision import transforms
import multiprocessing as mp

from omegaconf import DictConfig, ListConfig, OmegaConf

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    L0:        int                        # length of prompt

def build_llava_prompt(question: str, conv_template: str = "llava_llada"):
    import copy
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

@torch.no_grad()
def generate_llada_v_with_history(
    model, tokenizer, input_ids, images, image_sizes,
    steps, gen_length, block_length, temperature,
    target, unmask_threshold=None, mask_id=126336
) -> DiffusionOutput:
    position_ids = None
    attention_mask = None
    inputs_embeds = None
    (_input_ids, position_ids, attention_mask, _pkv, inputs_embeds, _labels) = model.prepare_inputs_labels_for_multimodal(
        input_ids, position_ids, attention_mask, None, None, images, ["image"], image_sizes=image_sizes
    )
    assert input_ids.shape[1] < inputs_embeds.shape[1], "inputs_embeds must be longer than input_ids, because it contains image embeddings."
    device = inputs_embeds.device
    B = inputs_embeds.shape[0]
    L0 = inputs_embeds.shape[1]
    d  = inputs_embeds.shape[2]

    assert B == 1, "LLaDA-V only support batch size 1 for now"

    total_len = L0 + gen_length
    masked_embed = model.get_model().embed_tokens(torch.tensor([mask_id], device=device))  # (1, d)
    x_embeds = masked_embed.repeat(B, total_len, 1)
    x_embeds[:, :L0] = inputs_embeds 
    x_tokens = torch.full((B, total_len), mask_id, dtype=torch.long, device=device) 
    prompt_index = torch.zeros_like(x_tokens, dtype=torch.bool)
    prompt_index[:, :L0] = True

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (i < rem) for i in range(num_blocks)]

    nfe = 0
    hist: List[torch.Tensor] = []

    for blk in range(num_blocks):
        s = L0 + blk * block_length
        e = L0 + (blk + 1) * block_length

        cur_steps = steps_per_block[blk]
        
        outputs = model.get_model()(inputs_embeds=x_embeds, attention_mask=None, position_ids=None, use_cache=False, return_dict=True)
        logits = model.lm_head(outputs.last_hidden_state).float()
        
        blk_tokens = x_tokens[:, s:e]
        eos_mask = (blk_tokens == 126348)
        if eos_mask.any():
            first_eos_pos = torch.where(eos_mask[0])[0][0].item()
            valid_region = torch.zeros_like(blk_tokens, dtype=torch.bool)
            valid_region[:, :first_eos_pos] = True
        else:
            valid_region = torch.ones_like(blk_tokens, dtype=torch.bool)
        
        mask_blk = (blk_tokens == mask_id) & valid_region
        num_transfer = get_num_transfer_tokens(mask_blk, cur_steps)
        
        x0_blk, tr_idx_blk = get_transfer_index(
            logits[:, s:e], temperature, target,
            mask_blk, blk_tokens, num_transfer[:, 0],
            threshold=unmask_threshold
        )
        x_slice = x_tokens[:, s:e]
        x_slice[tr_idx_blk] = x0_blk[tr_idx_blk]
        x_tokens[:, s:e] = x_slice
        x_embeds[:, s:e] = model.get_model().embed_tokens(x_tokens[:, s:e])
        hist.append(x_tokens.clone().cpu())
        nfe += 1

        i = 1
        while i < cur_steps:
            nfe += 1
            outputs = model.get_model()(inputs_embeds=x_embeds, attention_mask=None, position_ids=None, use_cache=False, return_dict=True)
            logits = model.lm_head(outputs.last_hidden_state).float()

            blk_tokens = x_tokens[:, s:e]
            eos_mask = (blk_tokens == 126348)
            if eos_mask.any():
                first_eos_pos = torch.where(eos_mask[0])[0][0].item()
                valid_region = torch.zeros_like(blk_tokens, dtype=torch.bool)
                valid_region[:, :first_eos_pos] = True
            else:
                valid_region = torch.ones_like(blk_tokens, dtype=torch.bool)
            
            mask_blk = (x_tokens[:, s:] == mask_id)
            mask_blk[:, block_length:] = False
            mask_blk[:, :block_length] = mask_blk[:, :block_length] & valid_region
            
            x0, tr_idx = get_transfer_index(
                logits[:, s:], temperature, target,
                mask_blk, x_tokens[:, s:], num_transfer[:, i],
                threshold=unmask_threshold
            )

            if tr_idx.any():
                x_slice = x_tokens[:, s:]
                x_slice[tr_idx] = x0[tr_idx]
                x_tokens[:, s:] = x_slice
                x0_embeds = model.get_model().embed_tokens(torch.where(tr_idx, x0, x_slice))
                x_embeds[:, s:][tr_idx] = x0_embeds[tr_idx]

            hist.append(x_tokens.clone().cpu())
            
            remaining_masks = (x_tokens[:, s:e] == mask_id) & valid_region
            if remaining_masks.sum() == 0:
                break
            i += 1
        
        if (x_tokens[:, s:e] == 126348).any():
            break


    return DiffusionOutput(sequences=x_tokens, history=hist, nfe=nfe, L0=L0)




def get_transfer_index(logits, temperature, target, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    logits_with_noise[..., 126336] = -float('inf')  # 禁止生成 mask_id
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

        for j in range(selected.shape[0]):
            k = int(num_transfer_tokens[j].item() if torch.is_tensor(num_transfer_tokens[j]) else num_transfer_tokens[j])
            if k <= 0:
                continue
            
            num_selected = selected[j].sum().item()
            if num_selected < k:
                masked_scores = x0_p[j].masked_fill(~mask_index[j], float("-inf"))
                _, top_indices = torch.topk(masked_scores, k=k)
                selected[j, :] = False
                selected[j, top_indices] = True

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
    for t, snap in enumerate(history, start=1):
        cur = snap[sample_idx]
        changed = (prev == mask_id) & (cur != mask_id)
        step_map[changed] = t
        prev = cur
    unprocessed_mask = (step_map == 0)
    if unprocessed_mask.any():
        max_step = step_map.max()
        step_map[unprocessed_mask] = max_step
        
    return step_map



from tqdm import tqdm

def worker(pretrained_model, rank, prompts, orig_idx, data_idx, image_paths, seq_dict, step_dict, imgabs_dict, batch_size, config):
    from PIL import Image
    import copy
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer_gpu, model_gpu, image_processor, max_length = load_pretrained_model(
        pretrained_model, None, "llava_llada", attn_implementation="sdpa", device_map=device
    )
    model_gpu.eval()
    special_tokens = {
        "additional_special_tokens": [DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
    }
    num_new = tokenizer_gpu.add_special_tokens(special_tokens)
    if num_new > 0:
        model_gpu.resize_token_embeddings(len(tokenizer_gpu))


    use_fast_dllm = getattr(config.rollout, "use_fast_dllm", False)
    use_dllm_cache = getattr(config.rollout, "use_dllm_cache", False)
    if use_fast_dllm:
        register_fast_dllm_hook(model_gpu)
    elif use_dllm_cache:
        dLLMCache.new_instance(**asdict(dLLMCacheConfig(
            prompt_interval_steps=getattr(config.rollout, "prompt_interval_steps", 25),
            gen_interval_steps=getattr(config.rollout, "gen_interval_steps", 7),
            transfer_ratio=getattr(config.rollout, "transfer_ratio", 0.25),
        )))
        register_cache_LLaDA_V(model_gpu, "model.layers")

    mask_id = 126336
    unmask_threshold = None if config.rollout.remasking_strategy == "low_confidence_static" else config.rollout.dynamic_threshold

    for j in tqdm(range(len(prompts)), desc=f"GPU {rank}", position=rank, leave=True):
        did = data_idx[j]
        prompt_text = prompts[j]
        img_path = image_paths[j] if image_paths is not None else None

        if img_path is not None:
            if not os.path.isabs(img_path) and hasattr(config.dataset, "image_root") and config.dataset.image_root is not None:
                img_path = os.path.join(config.dataset.image_root, img_path)
            img_path = os.path.abspath(img_path)
            imgabs_dict[did] = img_path

            image = Image.open(img_path).convert("RGB")
            image_tensor = process_images([image], image_processor, model_gpu.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
            image_sizes = [image.size]
        else:
            image_tensor = None
            image_sizes = None

        conv_prompt = build_llava_prompt(prompt_text, conv_template="llava_llada")
        input_ids = tokenizer_image_token(conv_prompt, tokenizer_gpu, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        # 生成（带历史）
        out = generate_llada_v_with_history(
            model=model_gpu,
            tokenizer=tokenizer_gpu,
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            steps=config.rollout.steps,
            gen_length=config.rollout.max_gen_length,
            block_length=config.rollout.block_size,
            temperature=config.rollout.temperature,
            target=config.rollout.target,
            unmask_threshold=unmask_threshold,
            mask_id=mask_id,
        )

        out.sequences = out.sequences.cpu()
        L0 = model_gpu.prepare_inputs_labels_for_multimodal(input_ids, None, None, None, None, image_tensor, ["image"], 
        image_sizes=image_sizes)[4].shape[1]
        assert L0 == out.L0, "L0 must be equal to the length of prompt."
        seq_ids = out.sequences[:, L0:].tolist()
        text = tokenizer_gpu.batch_decode(seq_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0]

        # step_map
        m = denoise_step_map(out.history, mask_id=mask_id, sample_idx=0)
        step_map = m[L0:].tolist()

        seq_dict[orig_idx[j]] = text
        step_dict[orig_idx[j]] = step_map

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
        system_prompts = """You need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}"""
    else:
        system_prompts = """{{problem}}"""
    
    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name


    assert config.dataset.data_type != "code", "lladav is not supported for code tasks."

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

    if config.model.model_base in ["llava_llada", "llada_v", "llada-v", "lladav"] and (config.dataset.data_type == "mmu" or config.evaluation.data_type == "mmu"):
        image_paths = []
        for i in range(num):
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
    imgabs_dict = manager.dict()  # {data_idx: absolute_image_path (str)}
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
                            imgabs_dict,
                            batch_size,
                            config))

        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # ------------------- 5. restore original order -------------------
    restored_outputs    = [seq_dict[i]  for i in range(N)]
    restored_step_maps  = [step_dict[i] for i in range(N)]
    if (config.model.model_base in ["llava_llada", "llada_v", "llada-v", "lladav"] and (config.dataset.data_type == "mmu" or config.evaluation.data_type == "mmu")):
        for di in range(num):
            if di in imgabs_dict:
                data[di]["image_abs_path"] = imgabs_dict[di]


    cprint("generation job done!", "green")














    import re

    # calculate the response length (ignoring repeated <|endoftext|> tokens)
    def get_token_lengths(strings, tokenizer):
        pad_token = "<|mdm_mask|>"

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
