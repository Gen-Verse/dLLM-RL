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


    index_list = []
    extracted_output_list = []
    ground_truth_list = []
    for i in range(len(data)):
        data[i]["correctness"] = []
        index_list = index_list + [i] * len(data[i]["extracted_output"])
        extracted_output_list = extracted_output_list + data[i]["extracted_output"]
        ground_truth_list = ground_truth_list + [data[i]["ground_truth_answer"]] * len(data[i]["extracted_output"])

    nest_asyncio.apply()

    async def get_correctness():
        executor = ThreadPoolExecutor(max_workers=64)
        tasks = []
        for i in range(len(index_list)):
            tasks.append(math_utils.is_equal(extracted_output_list[i], ground_truth_list[i], executor))
        results = await asyncio.gather(*tasks)
        return results

    correctness_list = asyncio.run(get_correctness())
    for i in range(len(index_list)):
        index_i = index_list[i]
        data[index_i]["correctness"].append(correctness_list[i])



    def z_score_normalize(lst):
        mean = sum(lst) / len(lst)
        std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
        if std == 0:
            return [0 for x in lst]
        return [(x - mean) / std for x in lst]


    def to_last_step_vector_reward(step_map, scalar_reward):
        if not step_map:
            return []
        m = max(step_map)
        r = float(scalar_reward)
        return [r if s == m else 0.0 for s in step_map]



    def set_last_t(lst: list, t: int) -> None:
        new_lst = lst.copy()
        new_val = max(lst) + 1
        new_lst[-t:] = [new_val] * t
        return new_lst
    

    def get_data_chunk(data, num_nodes, node_idx):
        total = len(data)
        start = (total * node_idx) // num_nodes
        end   = (total * (node_idx + 1)) // num_nodes
        return data[start:end]
    
    final_data = []
    response_length_list = []
    for i in range(len(data)):
        correctness = data[i]["correctness"]
        lengths = data[i]["response_length"]
        response_length_list = response_length_list + data[i]["response_length"]
        for j in range(len(lengths)):
            if OmegaConf.select(config, "rollout.max_gen_length", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_gen_length - 5:
                correctness[j] = False
            if OmegaConf.select(config, "rollout.max_token", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_token - 5:
                correctness[j] = False

        proportion = sum(correctness) / len(correctness)
        if proportion > 0.8 or proportion < 0.2:
            continue
        rewards = z_score_normalize(correctness)
        data[i]["normlized_outcome"] = rewards

        final_data.append(data[i])

    import os

    num_node = config.experiment.num_node
    if num_node > 1:
        for node_index in range(num_node):
            divide_data = get_data_chunk(final_data, num_node, node_index)
            output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"

            os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
            with open(output_file_name, "w", encoding="utf-8") as f:
                json.dump(divide_data, f, indent=2, ensure_ascii=False)
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)


    outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        # Save + print
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")
        
        acc = sum(correctness_list)/len(correctness_list)
        avg_len = sum(response_length_list)/len(response_length_list)

        output_text = f"train step: {config.experiment.current_epoch}  "
        
        if config.model.model_base != "sdar" and config.model.model_base != "trado":
            output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  block_size: {config.rollout.block_size}  acc: {acc}  avg length: {avg_len}"
        else:
            output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  top_k: {config.rollout.top_k}  acc: {acc}  avg length: {avg_len}"
    
        save_and_print(output_text)

    
