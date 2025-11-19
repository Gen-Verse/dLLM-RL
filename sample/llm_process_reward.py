import os
import re
import ast
import json
import random
import argparse
from jinja2 import Template
from termcolor import cprint
import multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer







os.environ["TOKENIZERS_PARALLELISM"] = "false" 





####### vllm inference #######

def worker_fn(pretrained_model, gpu_ids, task_queue, result_queue, max_model_len, max_generation_token, temp):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    print(f"Loading model on GPUs {gpu_ids}...")
    llm = LLM(
        model=pretrained_model,
        dtype="bfloat16",
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len
    )

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        max_tokens=max_generation_token,
        stop=["</answer>", "User:", "Human:", "Assistant:", "<|im_end|>", "<|endoftext|>"]
    )

    while True:
        task = task_queue.get()
        if task == "STOP":
            print("Stopping worker...")
            break
        task_id, prompts = task
        outputs = llm.generate(prompts, sampling_params)
        result_texts = [out.outputs[0].text for out in outputs]
        result_queue.put((task_id, result_texts))

# To run the worker setup:
def start_workers(pretrained_model, gpu_configs, max_model_len, max_generation_token, temp):
    task_queues = []
    result_queues = []
    processes = []

    for i, gpu_ids in enumerate(gpu_configs):
        task_q = mp.Queue()
        result_q = mp.Queue()
        p = mp.Process(
            target=worker_fn,
            args=(pretrained_model, gpu_ids, task_q, result_q, max_model_len, max_generation_token, temp)
        )
        p.start()
        task_queues.append(task_q)
        result_queues.append(result_q)
        processes.append(p)
    
    return task_queues, result_queues, processes

# Submit tasks
def submit_prompt_set(task_queues, prompt_sets):
    for i, prompts in enumerate(prompt_sets):
        task_queues[i].put((i, prompts))

# Collect results
def collect_results(result_queues, num_sets):
    results = [None] * num_sets
    for q in result_queues:
        task_id, result = q.get()
        results[task_id] = result
    return results

# Stop workers
def stop_workers(task_queues, processes):
    for q in task_queues:
        q.put("STOP")
    for p in processes:
        p.join()

# Split prompts into N chunks
def split_prompts(prompts, n):
    k, m = divmod(len(prompts), n)
    return [prompts[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def get_token_lengths(strings, tokenizer):
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]

# vllm inference
def generate_results(all_prompts, gpu_groups,task_queues, result_queues):
    prompt_sets = split_prompts(all_prompts, len(gpu_groups))
    submit_prompt_set(task_queues, prompt_sets)
    results = collect_results(result_queues, len(prompt_sets))
    result_list = []
    for result_set in results:
        for r in result_set:
            result_list.append(r)
    return result_list


import random 
def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list







def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node 
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]




from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


    

if __name__ == "__main__":

    config = get_config()


    k_sample = config.rollout.reward_model.k_sample
    reward_chunk_length = config.rollout.reward_model.reward_chunk_length
    gpu_groups = [[0,1,2,3],[4,5,6,7]]
    max_model_len = config.rollout.reward_model.max_model_len
    max_generation_token = config.rollout.reward_model.max_generation_token
    temp = config.rollout.reward_model.temperature

    project_name = config.experiment.project
    dataset = config.dataset.train_dataset
    reward_llm = config.model.process_reward_model
    num_node = config.experiment.num_node
    node_index = config.experiment.node_index

    if config.dataset.data_type == "code":
        raise NotImplementedError("Data type 'code' is not supported for process reward yet.")
    else:
        is_code = False
        system_prompts = """<|im_start|>You are a helpful assistant. <|im_end|>\n<|im_start|>user 
For the question below, you will be given an entire solution (we have already known it is {{if_correct}}) and an excerpt (a middle part) from it. Your task is to grade the excerpt for correctness using 1 (correct) or -1 (incorrect).

Rules:
- Judge ONLY the excerpt's correctness given the question and the whole solution (which is {{if_correct}}) as context.
- The excerpt may be truncated at the beginning or end due to chunking; do NOT penalize boundary truncation.
- If the excerpt contains any error, unjustified inference, or contradiction, score -1. If it is completely correct, score 1.
- Ignore minor grammar issues that do not affect correctness.

This is the question:
{{question}}

This is the final ground truth answer:
{{gt_answer}}

This is the given solution ({{if_correct}}):
{{whole_solution}}

This is the excerpt I need you to score:
{{excerpt}}

You need to put your final score in \\boxed{}. <|im_end|>\n<|im_start|>assistant<think>"""


    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    # read dataset
    outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
    if num_node > 1:
        with open("../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json", 'r') as f:
            data = json.load(f)
    else:
        with open("../" + project_name + f"/temp_data/outputs-" + outputs_name + ".json", 'r') as f:
            data = json.load(f)

    num = len(data)

    # load model, tokenizer, build vllm engines...
    task_queues, result_queues, processes = start_workers(reward_llm, gpu_groups, max_model_len, max_generation_token, temp)

    # obtain prompt
    def get_prompt(excerpt, question, correctness, full_output, gt_answer):
        if correctness:
            correctness_string = "correct"
        else:
            correctness_string = "incorrect"
        return Template(system_prompts).render(
            excerpt = excerpt,
            question = question,
            if_correct = correctness_string,
            whole_solution = full_output,
            gt_answer = gt_answer
            )




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


    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    # initialization
    generation_prompts = []
    index_list = []


    def get_prefix_text_by_chunk_end(chunk_end, token_ids) -> str:
        if chunk_end <= 0:
            return ""
        
        return tokenizer.decode(
            token_ids[:chunk_end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
    

    for i in range(num):
        # preproces
        question_i = data[i]["question"]
        n_j = len(data[i]["full_output"])
        data[i]["process_reward_list"] = []
        data[i]["reward_prompt"] = [[] for _ in range(n_j)]
        for j in range(n_j):
            data[i]["process_reward_list"].append([])
            full_output_j = data[i]["full_output"][j]
            length_j = data[i]["response_length"][j]
            correctness_j = data[i]["correctness"][j]
            enc = tokenizer(full_output_j, add_special_tokens=False)
            token_ids = enc["input_ids"]
            data[i]["reward_prompt"][j] = []
            n_k = max(1, int((length_j - 1) / reward_chunk_length + 1))
            for k in range(n_k):
                data[i]["process_reward_list"][j].append([])
                chunk_end = min(length_j, (k + 1) * reward_chunk_length)
                chunk_text = get_prefix_text_by_chunk_end(chunk_end, token_ids)
                prompt_text = get_prompt(chunk_text, question_i, correctness_j, full_output_j, data[i]["ground_truth_answer"])
                for _ in range(k_sample):
                    index_list.append((i, j, k))
                generation_prompts = generation_prompts + [prompt_text] * k_sample
                data[i]["reward_prompt"][j].append(prompt_text)




    # sampling process

    cprint("start generation...", "green")

    # shuffle first, to achieve efficiency
    all_prompts = generation_prompts
    N = len(all_prompts)
    indices = list(range(N))
    shuffled_idx = indices[:]      
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]
    # generate
    shuffled_outputs = generate_results(shuffled_prompts, gpu_groups, task_queues, result_queues)
    restored_outputs = [None] * N
    for out, idx in zip(shuffled_outputs, shuffled_idx):
        restored_outputs[idx] = out

    cprint("generation job done!", "green")






    






    





    # process generated codes
    i = 0
    for full_output in restored_outputs:
        extracted_score = extract_final_boxed_answer(full_output)
        i1, j1, k1 = index_list[i]
        if extracted_score != "1" and extracted_score != "-1":
            extracted_score = 0.0
        extracted_score = float(extracted_score)
        data[i1]["process_reward_list"][j1][k1].append(extracted_score)
        i += 1

    def all_equal(seq):
        it = iter(seq)
        try:
            first = next(it)
        except StopIteration:
            return True  
        return all(x == first for x in it)

    def z_score_normalize(lst):
        if len(lst) == 1:
            return lst
        mean = sum(lst) / len(lst)
        std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
        if std == 0:
            return [0 for x in lst]
        return [(x - mean) / std for x in lst]
    
    for i in range(len(data)):
        max_nk = -1
        for j in range(len(data[i]["full_output"])):
            n_k = len(data[i]["process_reward_list"][j])
            if n_k > max_nk:
                max_nk = n_k
            for k in range(n_k):
                if all_equal(data[i]["process_reward_list"][j][k]):
                    data[i]["process_reward_list"][j][k] = data[i]["process_reward_list"][j][k][0]
                else:
                    data[i]["process_reward_list"][j][k] = 2.0 * int(data[i]["correctness"][j]) - 1.0
        
        for k in range(max_nk):
            target_value = []
            target_index = []
            for j in range(len(data[i]["full_output"])):
                if (k + 1) <= len(data[i]["process_reward_list"][j]):
                    target_value.append(data[i]["process_reward_list"][j][k])
                    target_index.append(j)
            target_value = z_score_normalize(target_value)
            for j in target_index:
                data[i]["process_reward_list"][j][k] = target_value[target_index.index(j)]





    # output the data
    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    




    stop_workers(task_queues, processes)










