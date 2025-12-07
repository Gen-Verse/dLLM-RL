# -*- coding: utf-8 -*-
# Path: rl_process_reward.py
import os
import json
from statistics import mean

from termcolor import cprint
from omegaconf import OmegaConf


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def z_score_normalize(lst):
    if len(lst) <= 1:
        return lst[:]
    m = sum(lst) / len(lst)
    var = sum((x - m) ** 2 for x in lst) / len(lst)
    if var <= 0:
        return [0.0 for _ in lst]
    std = var ** 0.5
    return [(x - m) / std for x in lst]


if __name__ == "__main__":
    config = get_config()

    reward_chunk_length = int(config.model.process_reward_chunk_size)
    project_name = config.experiment.project

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    if config.experiment.function == "train":
        dataset = config.dataset.train_dataset
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset + f"-step{config.experiment.current_epoch}"
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset + f"-step{config.experiment.current_epoch}"
    else:
        raise ValueError(f"Unknown function: {config.experiment.function}")

    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in range(len(data)):
        n_j = len(data[i]["full_output"])

        # outcome reward（每条 response 一个标量）
        outcome_list = data[i]["correctness"]
        if (not isinstance(outcome_list, list)) or len(outcome_list) != n_j:
            raise ValueError(f"data[{i}]['correctness'] must be a list of length {n_j}")

        raw_chunk_scores = []
        max_nk = 0

        data[i].setdefault("process_reward", [])
        if len(data[i]["process_reward"]) < n_j:
            data[i]["process_reward"].extend(
                [[] for _ in range(n_j - len(data[i]["process_reward"]))]
            )

        pr_all = data[i].get("process_reward_list", [])

        for j in range(n_j):
            step_map_j = data[i]["step_map"][j]
            L = len(step_map_j)

            data[i]["process_reward"][j] = [0.0 for _ in range(L)]

            # 当前 response 的 process_reward_list
            pr_lists = []
            if isinstance(pr_all, list) and j < len(pr_all) and isinstance(pr_all[j], list):
                pr_lists = pr_all[j]

            outcome_r = 2 * float(outcome_list[j]) - 1.0

            per_j_scores = []
            for k in range(len(pr_lists)):
                samples = pr_lists[k] if isinstance(pr_lists[k], list) else []

                if samples and len(set(samples)) == 1 and mean(samples) != 0:
                    raw_val = mean(samples)
                else:
                    raw_val = outcome_r

                per_j_scores.append(float(raw_val))

            raw_chunk_scores.append(per_j_scores)
            if len(per_j_scores) > max_nk:
                max_nk = len(per_j_scores)

        norm_chunk_scores = [scores[:] for scores in raw_chunk_scores]  # deep copy
        for k in range(max_nk):
            vals = []
            idx_map = []
            for j in range(n_j):
                if k < len(raw_chunk_scores[j]):
                    vals.append(raw_chunk_scores[j][k])
                    idx_map.append(j)
            if not vals:
                continue
            vals_norm = z_score_normalize(vals)
            for jj, v in zip(idx_map, vals_norm):
                norm_chunk_scores[jj][k] = v

        for j in range(n_j):
            step_map_j = data[i]["step_map"][j]
            L = len(step_map_j)
            if L == 0:
                continue

            per_j_scores = norm_chunk_scores[j]
            n_k = len(per_j_scores)

            for k in range(n_k):
                reward_value = per_j_scores[k]

                chunk_start = k * reward_chunk_length
                if chunk_start >= L:
                    break  

                chunk_end = min((k + 1) * reward_chunk_length, L)
                if chunk_end <= chunk_start:
                    continue

                window_vals = step_map_j[chunk_start:chunk_end]
                if not window_vals:
                    continue

                max_step = max(window_vals)

                for idx in range(chunk_start, chunk_end):
                    if step_map_j[idx] == max_step:
                        data[i]["process_reward"][j][idx] = reward_value


    final_data = []
    for i in range(len(data)):
        for j in range(len(data[i]["full_output"])):
            data_i = {
                "prompt": data[i]["prompt"],
                "response": data[i]["full_output"][j],
                "step_map": data[i]["step_map"][j],
                "reward": data[i]["process_reward"][j],
            }
            if "image_token_ids" in data[i]:
                data_i["image_token_ids"] = data[i]["image_token_ids"]
            if "image_abs_path" in data[i]:
                data_i["image_abs_path"] = data[i]["image_abs_path"]
            final_data.append(data_i)

    out_training_json = (
        "../" + project_name + "/temp_data/" + config.dataset.optimization_data + f"-step{config.experiment.current_epoch}.json"
    )
    os.makedirs(os.path.dirname(out_training_json), exist_ok=True)
    with open(out_training_json, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    cprint(f"[DONE] Wrote training data to: {out_training_json}", "green")
    cprint(f"[DONE] Updated rollout file: {file_name}", "green")
