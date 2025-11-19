import json
import math_utils
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint
from omegaconf import MISSING
from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":

    config = get_config()

    reward_chunk_length = config.rollout.reward_model.reward_chunk_length

    project_name = config.experiment.project
    
    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name
    

    if config.experiment.function == "train":
        shrink = config.training.shrink
        dataset = config.dataset.train_dataset
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
        
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset
    
    
    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, 'r') as f:
        data = json.load(f)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    for i in range(len(data)):
        n_j = len(data[i]["full_output"])
        data[i]["process_reward"] = [[] for _ in range(n_j)]
        for j in range(n_j):
            data[i]["process_reward"][j] = [0 for _ in range(len(data[i]["step_map"][j]))]
            length_j = data[i]["response_length"][j]
            full_output_j = data[i]["full_output"][j]
            enc = tokenizer(full_output_j, add_special_tokens=False)
            token_ids = enc["input_ids"]
            n_k = int((length_j - 1)/reward_chunk_length + 1)
            step_map_j = data[i]["step_map"][j]
            for k in range(n_k):
                reward_value = data[i]["process_reward_list"][j][k]
                if k != n_k - 1:
                    chunk_end = min(length_j, (k + 1) * reward_chunk_length)
                    start = max(0, chunk_end - config.rollout.block_size)
                    end = chunk_end
                    max_val = max(step_map_j[start:end])
                    for idx in range(start, end):
                        if step_map_j[idx] == max_val:
                            data[i]["process_reward"][j][idx] = reward_value
                else:
                    max_val = max(step_map_j)
                    for idx, val in enumerate(step_map_j):
                        if val == max_val:
                            data[i]["process_reward"][j][idx] = reward_value
    
    final_data = []
    for i in range(len(data)):
        for j in range(len(data[i]["full_output"])):
            final_data.append({
                "prompt": data[i]["prompt"],
                "response": data[i]["full_output"][j],
                "step_map": data[i]["step_map"][j],
                "reward": data[i]["process_reward"][j]
            })
    
    with open("../" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)


    import os
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


    

                    
                
