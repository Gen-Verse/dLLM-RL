import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoConfig

from train.utils import get_config
from models import LlavaLLaDAModelLM



def _get_hidden_size(cfg):
    for k in ["hidden_size", "d_model", "n_embd", "embedding_size"]:
        if hasattr(cfg, k):
            return int(getattr(cfg, k))
    raise ValueError("Cannot infer hidden size from config (expected one of hidden_size/d_model/n_embd/embedding_size).")


def _get_value_model(base_llm_class, value_head_prefix="value_head"):
    class ValueModel(base_llm_class):
        def __init__(self, config: AutoConfig):
            try:
                import deepspeed
                zero_off = deepspeed.zero.Init(enabled=False)
            except Exception:
                from contextlib import nullcontext
                zero_off = nullcontext()

            with zero_off:
                super().__init__(config)

            self.value_head_prefix = value_head_prefix
            hs = _get_hidden_size(config)
            vh = nn.Linear(hs, 1, bias=False)
            setattr(self, value_head_prefix, vh)

        def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
            kwargs.pop("labels", None)
            kwargs.pop("position_ids", None)
            outputs = self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            return getattr(self, self.value_head_prefix)(hidden).squeeze(-1)
    return ValueModel



def save_checkpoint(model, tokenizer, config, name, accelerator=None):
    from pathlib import Path
    import json, shutil, os, glob, importlib, inspect, time

    output_dir = Path(f"../{config.experiment.project}")
    output_dir.mkdir(parents=True, exist_ok=True)

    def _is_main():
        if accelerator is not None:
            return accelerator.is_main_process
        import torch.distributed as dist
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    is_main = _is_main()

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)
    if is_main and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]) if "-" in p.name else -1,
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    if accelerator is not None:
        model_to_save = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(model)
        save_function = accelerator.save
    else:
        model_to_save = model
        state_dict = model.state_dict()
        save_function = None

    if is_main:
        save_dir = save_base / name
        model_to_save.save_pretrained(
            save_dir,
            state_dict=state_dict,
            safe_serialization=True,
            save_function=save_function,
        )
        tokenizer.save_pretrained(str(save_dir))

        def _copy_dynamic_modules(dst_dir, model_obj, tok_obj):
            copied = 0
            modules = set()
            for obj in [model_obj, getattr(model_obj, "config", None), tok_obj]:
                if obj is None:
                    continue
                modname = getattr(obj.__class__, "__module__", None)
                if modname:
                    modules.add(modname)

            for modname in modules:
                try:
                    mod = importlib.import_module(modname)
                    src_file = inspect.getsourcefile(mod)
                    if not src_file or not os.path.exists(src_file):
                        continue
                    base_dir = os.path.dirname(src_file)
                    for pattern in ("modeling_*.py", "configuration_*.py", "tokenization_*.py", "processing_*.py"):
                        for fn in glob.glob(os.path.join(base_dir, pattern)):
                            dst = os.path.join(dst_dir, os.path.basename(fn))
                            if not os.path.exists(dst):
                                shutil.copy2(fn, dst)
                                copied += 1
                except Exception:
                    pass

        _copy_dynamic_modules(str(save_dir), model_to_save, tokenizer)

        metadata = {"save_time": time.strftime("%Y-%m-%d %H:%M:%S")}
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

    if accelerator is None:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def main():
    config = get_config()
    pretrained_model = config.model.value_base_model

    value_model_class = _get_value_model(LlavaLLaDAModelLM, "value_head")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    value_model = value_model_class.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

    with torch.no_grad():
        getattr(value_model, "value_head").weight.zero_()

    save_checkpoint(value_model, tokenizer, config, config.model.optimized_value_name)


if __name__ == "__main__":
    main()
