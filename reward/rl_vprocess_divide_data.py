import json
import math_utils_v
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint
from omegaconf import MISSING
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import List, Union
from functools import partial
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.functional.multimodal import clip_score
from transformers import AutoTokenizer
import os

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
    clip_hf_tokenizer = AutoTokenizer.from_pretrained(
        config.model.get("clip_model_path", "/data_storage/shared/pretrained_models/clip-vit-large-patch14-336/")
    )
    clip_score_fn = partial(
        clip_score,
        model_name_or_path=config.model.get("clip_model_path", "/data_storage/shared/pretrained_models/clip-vit-large-patch14-336/")
    )
    def load_image_uint8(path: str, image_root: str | None, resolution: int | None) -> torch.Tensor:
        if not os.path.isabs(path) and image_root is not None:
            path = os.path.join(image_root, path)
        img = Image.open(path).convert("RGB")
        tfms = []
        if resolution is not None:
            tfms += [transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(resolution)]
        tfms += [transforms.ToTensor()]
        t = transforms.Compose(tfms)(img) * 255.0
        return t.to(torch.uint8)
    def safe_clip_score(img_tensor: torch.Tensor, text: str, clip_fn):
        enc = clip_hf_tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=77,
            return_attention_mask=False
        )
        ids = enc["input_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        text_trunc = clip_hf_tokenizer.decode(ids, skip_special_tokens=True).strip()

        try:
            return clip_fn(img_tensor, text_trunc)
        except Exception as e:
            msg = str(e)
            if "upgrade torch to at least v2.6" in msg or "require users to upgrade torch" in msg:
                fallback_repo = "laion/CLIP-ViT-L-14-336-laion2B-s32B-b82K"
                return clip_score(img_tensor, text_trunc, model_name_or_path=fallback_repo)
            raise


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
    caption_flag_list = []
    for i in range(len(data)):
        data[i]["correctness"] = []
        data[i]["clip_scores"] = []
        response_length_list += data[i]["response_length"]
        index_list += [i] * len(data[i]["extracted_output"])
        extracted_output_list += remove_endoftext(data[i]["extracted_output"])
        caption_flag_list += [bool(data[i].get("is_caption", False))] * len(data[i]["extracted_output"])

    nest_asyncio.apply()

    noncap_positions = [p for p in range(len(index_list)) if not caption_flag_list[p]]
    ground_truth_list_noncap = [data[index_list[p]]["ground_truth_answer"] for p in noncap_positions]

    async def get_correctness_noncap():
        executor = ThreadPoolExecutor(max_workers=64)
        tasks = [math_utils_v.is_equal(extracted_output_list[pos], ground_truth_list_noncap[i], executor)
                for i, pos in enumerate(noncap_positions)]
        results = await asyncio.gather(*tasks)
        return results

    correctness_noncap = asyncio.run(get_correctness_noncap())
    for k, pos in enumerate(noncap_positions):
        index_i = index_list[pos]
        data[index_i]["correctness"].append(correctness_noncap[k])

    clip_scores_all = []
    for pos in range(len(index_list)):
        if caption_flag_list[pos]:
            index_i = index_list[pos]
            img_path = data[index_i]["image"]
            img_tensor = load_image_uint8(
                img_path,
                config.dataset.get("image_root", None),
                config.model.get("image_resolution", None)
            )
            score = safe_clip_score(img_tensor, extracted_output_list[pos], clip_score_fn)
            s = float(score.item()) if hasattr(score, "item") else float(score)
            data[index_i]["clip_scores"].append(s)
            clip_scores_all.append(s)




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
    data1 = []
    for i in range(len(data)):
        is_caption = bool(data[i].get("is_caption", False))
        lengths = data[i]["response_length"]

        if is_caption:
            metric_list = data[i]["clip_scores"]
        else:
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
            if not is_caption:
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
                if "image" in data[i]:
                    data_i["image"] = data[i]["image"]
                    data_i["image_abs_path"] = os.path.join(config.dataset.image_root, data[i]["image"])
                final_data.append(data_i)
            
            data1.append(data[i])

        if config.experiment.function == "evaluation":
            data[i]["step_map"] = []



    if config.experiment.function == "train":
        with open("../" + project_name + "/temp_data/" + config.dataset.optimization_data + f"-step{config.experiment.current_epoch}.json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)



    def get_data_chunk(data, num_nodes, node_idx):
        total = len(data)
        start = (total * node_idx) // num_nodes
        end   = (total * (node_idx + 1)) // num_nodes
        return data[start:end]
    
    import os

    num_node = config.experiment.num_node
    if num_node > 1:
        for node_index in range(num_node):
            divide_data = get_data_chunk(data1, num_node, node_index)
            output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"

            os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
            with open(output_file_name, "w", encoding="utf-8") as f:
                json.dump(divide_data, f, indent=2, ensure_ascii=False)
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w", encoding="utf-8") as f:
            json.dump(data1, f, indent=2, ensure_ascii=False)


    outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        # Save + print
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")
        
        acc = (sum(correctness_noncap)/len(correctness_noncap)) if len(correctness_noncap) > 0 else float("nan")
        avg_clip = (sum(clip_scores_all)/len(clip_scores_all)) if len(clip_scores_all) > 0 else float("nan")
        avg_len = sum(response_length_list)/len(response_length_list)

        output_text = f"train step: {config.experiment.current_epoch}  "
        if config.experiment.function == "train":
            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  block_size: {config.rollout.block_size}  acc(non-math): {acc}  avg CLIP(caption): {avg_clip}  avg length: {avg_len}"
            else:
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  top_k: {config.rollout.top_k}  acc(non-math): {acc}  avg CLIP(caption): {avg_clip}  avg length: {avg_len}"
        else:
            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  block_size: {config.evaluation.block_size}  acc(non-math): {acc}  avg CLIP(caption): {avg_clip}  avg length: {avg_len}"
            else:
                output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  top_k: {config.evaluation.top_k}  acc(non-math): {acc}  avg CLIP(caption): {avg_clip}  avg length: {avg_len}"
        save_and_print(output_text)

