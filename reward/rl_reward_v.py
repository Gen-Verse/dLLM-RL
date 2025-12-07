import json
import math_utils_v
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint
from omegaconf import MISSING
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import List, Union
import os

def resolve_image_abs_path(sample: dict, dataset_image_root: str | None) -> str | None:
    p = sample.get("image_abs_path")
    if p is None:
        p = sample.get("image") or sample.get("image_path")
        if p is None:
            return None
        if not os.path.isabs(p) and dataset_image_root:
            p = os.path.join(dataset_image_root, p)
    try:
        return os.path.abspath(p)
    except Exception:
        return p

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def remove_endoftext(obj: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Remove all occurrences of '<|endoftext|>' from a string or from each string in a list.
    Non-str elements in the list are left unchanged.
    """
    token = '<|endoftext|>'
    if isinstance(obj, str):
        return obj.replace(token, '')
    if isinstance(obj, list):
        return [s.replace(token, '') if isinstance(s, str) else s for s in obj]
    raise TypeError("Expected str or list of str")

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
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset + f"-step{config.experiment.current_epoch}"
        
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset + f"-step{config.experiment.current_epoch}"
    
    

    
    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, 'r') as f:
        data = json.load(f)


    index_list = []
    extracted_output_list = []
    response_length_list = []
    for i in range(len(data)):
        data[i]["correctness"] = []
        response_length_list += data[i]["response_length"]
        index_list += [i] * len(data[i]["extracted_output"])
        extracted_output_list += remove_endoftext(data[i]["extracted_output"])

    nest_asyncio.apply()
    async def get_correctness_all():
        executor = ThreadPoolExecutor(max_workers=64)
        tasks = [math_utils_v.is_equal(
                    extracted_output_list[p],
                    data[index_list[p]]["ground_truth_answer"],
                    executor
                ) for p in range(len(index_list))]
        return await asyncio.gather(*tasks)

    correctness_all = asyncio.run(get_correctness_all())
    for p, ok in enumerate(correctness_all):
        index_i = index_list[p]
        data[index_i]["correctness"].append(bool(ok))





    def z_score_normalize(lst):
        mean = sum(lst) / len(lst) if len(lst) > 0 else 0.0
        std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5 if len(lst) > 0 else 0.0
        if std == 0:
            return [0 for _ in lst]
        return [(x - mean) / std for x in lst]

    def set_last_t(lst: list, t: int) -> None:
        new_lst = lst.copy()
        new_val = (max(lst) + 1) if len(lst) > 0 else 1
        new_lst[-t:] = [new_val] * t
        return new_lst

    final_data = []
    for i in range(len(data)):
        lengths = data[i]["response_length"]
        metric_list = data[i]["correctness"]
        if config.reward.strict_len_check:
            for j in range(len(lengths)):
                if OmegaConf.select(config, "rollout.max_gen_length", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_gen_length - 5:
                    metric_list[j] = False
                if OmegaConf.select(config, "rollout.max_token", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_token - 5:
                    metric_list[j] = False

        rewards = z_score_normalize(metric_list)
        data[i]["rewards"] = rewards

        if config.experiment.function == "train":
            proportion = sum(data[i]["correctness"]) / len(data[i]["correctness"]) if len(data[i]["correctness"]) > 0 else 0.0
            if proportion > 0.95 or proportion < 0.05:
                continue

            for j in range(len(rewards)):
                data_i = {}
                data_i["prompt"] = data[i]["prompt"]
                data_i["reward"] = rewards[j]
                data_i["response"] = data[i]["full_output"][j]
                data_i["step_map"] = data[i]["step_map"][j]
                if "image_token_ids" in data[i]:
                    data_i["image_token_ids"] = data[i]["image_token_ids"]
                p_abs = resolve_image_abs_path(data[i], config.dataset.get("image_root", None))
                if p_abs is not None:
                    data_i["image_abs_path"] = p_abs

                final_data.append(data_i)

        if config.experiment.function == "evaluation":
            data[i]["step_map"] = []



    if config.experiment.function == "train":
        with open("../" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)


    import os
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


    outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        # Save + print
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")
        
        acc = (sum(correctness_all)/len(correctness_all)) if len(correctness_all) > 0 else float("nan")
        avg_len = sum(response_length_list)/len(response_length_list)

        output_text = f"train step: {config.experiment.current_epoch}  "
        if config.experiment.function == "train":
            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  block_size: {config.rollout.block_size}  acc: {acc}  avg length: {avg_len}"
            else:
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  top_k: {config.rollout.top_k}  acc: {acc}  avg length: {avg_len}"
        else:
            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  block_size: {config.evaluation.block_size}  acc: {acc}  avg length: {avg_len}"
            else:
                output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  top_k: {config.evaluation.top_k}  acc: {acc}  avg length: {avg_len}"
        save_and_print(output_text)
