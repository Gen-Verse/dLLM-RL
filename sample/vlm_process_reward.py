# -*- coding: utf-8 -*-
# Path: vlm_process_reward.py

import os
import re
import json
import copy
import random
import traceback
from typing import List, Dict, Any, Tuple

import multiprocessing as mp
from jinja2 import Template
from termcolor import cprint

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from omegaconf import OmegaConf

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==================== boxed 提取 ====================

def extract_final_boxed_answer(s: str) -> str:
    tag = r'\boxed{'
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"
    i = start + len(tag)
    depth = 1
    buf = []
    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def normalize_reward_from_response(resp: str) -> int:
    boxed = extract_final_boxed_answer(resp or "")
    if boxed == "Can not extract the answer!":
        return 0
    val = boxed.strip()
    if not re.fullmatch(r'[+\-]?1', val):
        return 0
    try:
        v = int(val)
    except Exception:
        return 0
    return v if v in (1, -1) else 0



def _resolve_local_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    msgs = copy.deepcopy(messages)
    for m in msgs:
        if "content" not in m:
            continue
        for chunk in m["content"]:
            if isinstance(chunk, dict) and chunk.get("type") == "image":
                img_val = chunk.get("image")
                if isinstance(img_val, str) and os.path.exists(img_val):
                    try:
                        chunk["image"] = Image.open(img_val).convert("RGB")
                    except Exception:
                        pass
    return msgs


def prepare_inputs_for_vllm(messages: List[Dict[str, Any]],
                            processor: AutoProcessor) -> Dict[str, Any]:
    messages = _resolve_local_images(messages)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    patch_size = getattr(
        getattr(processor, "image_processor", object()),
        "patch_size",
        14,
    )
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
    except TypeError:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=patch_size,
            return_video_kwargs=True,
        )

    mm_data: Dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def build_reward_messages(abs_image_path: str,
                          text_prompt: str) -> List[Dict[str, Any]]:
    if abs_image_path:
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that scores the correctness of "
                    "response excerpts based on the provided image and context."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": abs_image_path},
                    {"type": "text", "text": text_prompt},
                ],
            },
        ]
    else:
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that scores the correctness of "
                    "response excerpts based on the provided context."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                ],
            },
        ]



def worker_fn(pretrained_model: str,
              gpu_ids: List[int],
              task_queue: mp.Queue,
              result_queue: mp.Queue,
              dtype: str,
              gpu_mem: float,
              max_model_len: int,
              temperature: float,
              top_p: float,
              max_tokens: int,
              stop_words: List[str]):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    print(f"[Worker-{gpu_ids}] Loading model...", flush=True)

    processor = AutoProcessor.from_pretrained(pretrained_model)
    llm = LLM(
        model=pretrained_model,
        trust_remote_code=True,
        dtype=dtype,
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len if max_model_len > 0 else None,
        enforce_eager=False,
    )
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_words,
    )

    while True:
        task = task_queue.get()
        if task == "STOP":
            print(f"[Worker-{gpu_ids}] Stopping...", flush=True)
            break

        try:
            task_id, msg_chunk, R = task  # msg_chunk: List[messages]
        except Exception:
            print(f"[Worker-{gpu_ids}] Invalid task: {task}", flush=True)
            continue

        if not msg_chunk:
            result_queue.put((task_id, []))
            continue

        try:
            batch_inputs = []
            for messages in msg_chunk:
                inp = prepare_inputs_for_vllm(messages, processor)
                batch_inputs.extend([inp] * R)

            outs = llm.generate(batch_inputs, sampling_params=sampling)
            texts = [o.outputs[0].text if o.outputs else "" for o in outs]
            result_queue.put((task_id, texts))
        except Exception as e:
            print(f"[Worker-{gpu_ids}] ERROR in generate: {repr(e)}", flush=True)
            traceback.print_exc()
            result_queue.put((task_id, []))


def start_workers(pretrained_model: str,
                  gpu_groups: List[List[int]],
                  dtype: str,
                  gpu_mem: float,
                  max_model_len: int,
                  temperature: float,
                  top_p: float,
                  max_tokens: int,
                  stop_words: List[str]):
    task_queues, result_queues, procs = [], [], []
    for gpu_ids in gpu_groups:
        tq = mp.Queue()
        rq = mp.Queue()
        p = mp.Process(
            target=worker_fn,
            args=(
                pretrained_model,
                gpu_ids,
                tq,
                rq,
                dtype,
                gpu_mem,
                max_model_len,
                temperature,
                top_p,
                max_tokens,
                stop_words,
            ),
        )
        p.start()
        task_queues.append(tq)
        result_queues.append(rq)
        procs.append(p)
    return task_queues, result_queues, procs


def stop_workers(task_queues: List[mp.Queue], processes: List[mp.Process]):
    for q in task_queues:
        try:
            q.put("STOP")
        except Exception:
            pass
    for p in processes:
        try:
            p.join()
        except Exception:
            pass


def split_even(items: List[Any], n: int) -> List[List[Any]]:
    if n <= 0:
        return [items]
    k, m = divmod(len(items), n)
    return [
        items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        for i in range(n)
    ]


def generate_results(messages_all: List[List[Dict[str, Any]]],
                     R: int,
                     gpu_groups: List[List[int]],
                     task_queues: List[mp.Queue],
                     result_queues: List[mp.Queue]) -> List[str]:

    num_engines = len(gpu_groups)
    chunks = split_even(messages_all, num_engines)

    jobs = []
    for i, (q, chunk) in enumerate(zip(task_queues, chunks)):
        if not chunk:
            continue
        q.put((i, chunk, R))
        jobs.append(i)

    results_by_job: Dict[int, List[str]] = {}
    remaining = set(jobs)

    idle_loops = 0
    while remaining:
        got_any = False
        for i, rq in enumerate(result_queues):
            if i not in remaining:
                continue
            try:
                task_id, result = rq.get(timeout=0.5)
            except Exception:
                continue
            results_by_job[task_id] = result
            remaining.remove(task_id)
            got_any = True

        if not got_any:
            idle_loops += 1
            if idle_loops % 10 == 0:
                print(f"[Main] Waiting for workers... remaining jobs: {sorted(remaining)}",
                      flush=True)
            if idle_loops > 600:  # 600*0.5s = 300s
                raise RuntimeError(
                    f"[Main] generate_results stuck, remaining jobs: {sorted(remaining)}"
                )

    flat: List[str] = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        flat.extend(results_by_job.get(i, []))
    return flat


# ==================== 配置 ====================

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


# ==================== 主流程 ====================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    config = get_config()

    reward_llm = config.model.process_reward_model
    
    reward_chunk_length = config.model.process_reward_chunk_size

    gpu_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    k_sample = 3
    max_model_len = 15000
    max_generation_token = 10000
    temp = 0.6
    top_p = 0.95
    stop_words = [
        "</answer>",
        "User:",
        "Human:",
        "Assistant:",
        "<|im_end|>",
        "<|endoftext|>",
    ]
    dtype = "bfloat16"
    gpu_mem = 0.85

    project_name = config.experiment.project
    dataset = config.dataset.train_dataset
    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    base_dir = config.system.base_dir

    if config.experiment.current_epoch == 1:
        policy_model = config.model.pretrained_model
    else:
        policy_model = f"../{project_name}/ckpt/{config.model.optimized_name}"

    outputs_name = "rl-" + policy_model.replace("/", ".") + "-" + dataset + f"-step{config.experiment.current_epoch}"
    if num_node > 1:
        input_path = f"../{project_name}/temp_data/outputs-{node_index}-{outputs_name}.json"
    else:
        input_path = f"../{project_name}/temp_data/outputs-{outputs_name}.json"


    print(f"[Main] Load {input_path}", flush=True)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    num = len(data)
    print(f"[Main] #items = {num}", flush=True)

    print(f"[Main] Launch {len(gpu_groups)} engines: {gpu_groups} | R={k_sample}",
          flush=True)
    task_queues, result_queues, processes = start_workers(
        pretrained_model=reward_llm,
        gpu_groups=gpu_groups,
        dtype=dtype,
        gpu_mem=gpu_mem,
        max_model_len=max_model_len,
        temperature=temp,
        top_p=top_p,
        max_tokens=max_generation_token,
        stop_words=stop_words,
    )

    system_prompts = """For the question below (with the provided image), you will be given an entire response (we have already known it is {{if_correct}}) and an excerpt (a middle part) from it. Your task is to grade the excerpt for correctness using 1 (correct) or -1 (incorrect).

Rules:
- Judge ONLY the excerpt's correctness given the question, the image content, and the whole response (which is {{if_correct}}) as context.
- The excerpt may be truncated at the beginning or end due to chunking; do NOT penalize boundary truncation.
- If the excerpt is consistent and logically justified, score 1.
- If the excerpt contains logical error, score -1.
- Ignore minor grammar issues that do not affect correctness.
- You need to think step by step first then put your final score in \\boxed{}.

This is the question:
{{question}}

This is the final ground truth answer:
{{gt_answer}}

This is the given response ({{if_correct}}):
{{whole_solution}}

This is the excerpt I need you to score:
{{excerpt}}

You need to think step by step first then put your final score in \\boxed{}.
"""

    def get_prompt_text(excerpt, question, correctness, full_output, gt_answer):
        correctness_string = "correct" if correctness else "incorrect"
        return Template(system_prompts).render(
            excerpt=excerpt,
            question=question,
            if_correct=correctness_string,
            whole_solution=full_output,
            gt_answer=gt_answer,
        )

    # tokenizer for chunking
    policy_tokenizer = AutoTokenizer.from_pretrained(
        policy_model,
        trust_remote_code=True,
    )

    messages_all: List[List[Dict[str, Any]]] = []
    index_list: List[Tuple[int, int, int]] = []

    def get_prefix_text_by_chunk_end(chunk_end: int,
                                     token_ids: List[int]) -> str:
        chunk_end = max(0, min(chunk_end, len(token_ids)))
        if chunk_end <= 0:
            return ""
        return policy_tokenizer.decode(
            token_ids[:chunk_end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    print("[Main] Build prompts...", flush=True)
    for i in range(num):
        q_i = data[i]["question"]
        n_j = len(data[i]["full_output"])

        data[i]["process_reward_list"] = []
        data[i]["reward_prompt"] = [[] for _ in range(n_j)]
        data[i]["process_reward_response"] = []

        rel_image = data[i].get("image", "")
        abs_image = os.path.join(config.dataset.image_root, rel_image) if rel_image else ""

        for j in range(n_j):
            data[i]["process_reward_list"].append([])
            data[i]["process_reward_response"].append([])

            full_j = data[i]["full_output"][j]
            length_j = data[i]["response_length"][j]
            correctness_j = data[i]["correctness"][j]

            enc = policy_tokenizer(full_j, add_special_tokens=False)
            token_ids = enc["input_ids"]

            n_k = max(1, int((length_j - 1) / reward_chunk_length + 1))
            for k in range(n_k):
                chunk_end = min(length_j, (k + 1) * reward_chunk_length)
                raw_chunk_text = get_prefix_text_by_chunk_end(chunk_end, token_ids)

                if config.model.model_base == "mmada":
                    pad_token = "<|endoftext|>"
                else:
                    pad_token = "<|mdm_mask|>"
                clean_chunk_text = raw_chunk_text.replace(pad_token, "").strip()
                full_j = full_j.replace(pad_token, "").strip()

                if not clean_chunk_text:
                    data[i]["reward_prompt"][j].append("")
                    # 直接填好 k_sample 个 0，后面不再改写这个位置
                    data[i]["process_reward_list"][j].append([0.0] * k_sample)
                    data[i]["process_reward_response"][j].append([""] * k_sample)
                    continue

                prompt_text = get_prompt_text(
                    clean_chunk_text,
                    q_i,
                    correctness_j,
                    full_j,
                    data[i]["ground_truth_answer"],
                )
                data[i]["reward_prompt"][j].append(prompt_text)

                msgs = build_reward_messages(abs_image, prompt_text)
                messages_all.append(msgs)
                index_list.append((i, j, k))

                data[i]["process_reward_list"][j].append([])
                data[i]["process_reward_response"][j].append([])

    Nq = len(messages_all)
    print(f"[Main] Total prompts: {Nq}", flush=True)

    if Nq > 0:
        idx_perm = list(range(Nq))
        random.shuffle(idx_perm)
        shuffled_msgs = [messages_all[t] for t in idx_perm]

        cprint("start VLM generation...", "green")
        flat_texts_shuffled = generate_results(
            shuffled_msgs,
            R=k_sample,
            gpu_groups=gpu_groups,
            task_queues=task_queues,
            result_queues=result_queues,
        )
        cprint("generation job done!", "green")

        expected = Nq * k_sample
        if len(flat_texts_shuffled) != expected:
            raise RuntimeError(
                f"Unexpected #outputs: {len(flat_texts_shuffled)} vs {expected}"
            )

        flat_texts: List[str] = [None] * expected
        for pos, orig_idx in enumerate(idx_perm):
            start = pos * k_sample
            flat_texts[
                orig_idx * k_sample: orig_idx * k_sample + k_sample
            ] = flat_texts_shuffled[start:start + k_sample]

        p = 0
        for t in range(Nq):
            i1, j1, k1 = index_list[t]
            scores: List[float] = []
            responses: List[str] = []  # 新增：记录原始输出
            for _ in range(k_sample):
                out = flat_texts[p]
                p += 1
                responses.append(out)
                scores.append(float(normalize_reward_from_response(out)))
            data[i1]["process_reward_list"][j1][k1] = scores
            data[i1]["process_reward_response"][j1][k1] = responses  # 新增

    
    if num_node > 1:
        output_path = f"../{project_name}/temp_data/outputs-{node_index}-{outputs_name}.json"
    else:
        output_path = f"../{project_name}/temp_data/outputs-{outputs_name}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Main] Saved to {output_path}", flush=True)

    stop_workers(task_queues, processes)
    print("[Main] Workers stopped.", flush=True)
