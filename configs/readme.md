# Instructions

We introduce the recommended configs for different tasks and explain how to modify your own configs. Each config file governs one project/job. Make sure you fill in all the required parameters in the config file before using it; at the very least, this includes the dataset name, the model name, and the num of gpus you have. The details on how to modify the given example configs, along with definitions of their elements, are included within them.



## Which config to use?


### Evaluation:

For TraDo instruction models, use `trado_eval.yaml` or `trado_multinode_eval.yaml`. 

For SDAR models, use `sdar_eval.yaml` or `sdar_multinode_eval.yaml`. 

For long-CoT model, TraDo-8B-Thinking, use `trado_longcot_eval.yaml` or `trado_longcot_multinode_eval.yaml`. 

For Dream series and diffu-coder, use `dream_eval.yaml` or `dream_multinode_eval.yaml`. 

For LLaDA series and MMaDA, use `llada_eval.yaml` or `llada_multinode_eval.yaml`.

Then use `eval.py` or `multinode_eval.py` to start your evaluation!

### SFT:

For TraDo models, use `sft_trado.yaml`. 

For SDAR models, use `sft_sdar.yaml`. 

For dream and diffu-coder, use `sft_dream.yaml`. 

For LLaDA and MMaDA, use `sft_llada.yaml`.

### RL:

For TraDo models, use `rl_trado.yaml` or `multinode_rl_trado.yaml`. 
If use value model, use `rl_trado_with_value.yaml` or `multinode_rl_trado_with_value.yaml`. 

For SDAR models, use `rl_sdar.yaml` or `multinode_rl_sdar.yaml`. 
If use value model, use `rl_sdar_with_value.yaml` or `multinode_rl_sdar_with_value.yaml`. 

For dream and diffu-coder, use `sft_dream.yaml` or `multinode_rl_dream.yaml`. 

For LLaDA and MMaDA, use `sft_llada.yaml` or `multinode_rl_llada.yaml`. 

We also support coding rl, see an example script `rl_sdar_code.yaml`.

Then use `rl.py` or `multinode_rl.py` to start your RL!


## Main required fields:

The model name, dataset to eval on or train on (and the data type, math or code),  the number of nodes you have (corresponding deepspeed config).


## which python script (and command) to use?

### Evaluation:

Use `eval.py` or `multinode_eval.py` (if uou have multi-nodes).

Then simply:
```
python eval.py config=configs/CONFIG
# sample: CONFIG = sdar_eval.yaml
```
or (for multi-nodes)
```
if [[ ${MLP_ROLE_INDEX:-0} -eq 0 ]]; then   
    python multinode_eval.py config=configs/CONFIG
else
    exec tail -f /dev/null
fi
# sample: CONFIG = dream_multinode_eval.yaml
```

### SFT:

For Trado: `sft_trado.py`

For SDAR: `sft_sdar.py`

For Dream and Diffu-Coder: `sft_dream.py`

For LLaDA: `sft_llada.py`

For MMaDA: `sft_mmada.py`

Then simply:
```
accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 8888 \
  --config_file accelerate_configs/YOUR_DEEPSPEED_CONFIG \
  train/PYTHON_SCRIPT \
  config=configs/CONFIG
# example: YOUR_DEEPSPEED_CONFIG = 1_node_8_gpus_deepspeed_zero3.yaml, PYTHON_SCRIPT = sft_sdar.py, CONFIG = sft_sdar.yaml
```


### RL:

Use `rl.py` or `multinode_rl.py` (if uou have multi-nodes).

Then simply:
```
python rl.py config=configs/CONFIG
# sample: CONFIG = rl_sdar.yaml
```
or (for multi-nodes)
```
if [[ ${MLP_ROLE_INDEX:-0} -eq 0 ]]; then   
    python multinode_rl.py config=configs/CONFIG
else
    exec tail -f /dev/null
fi
# sample: CONFIG = multinode_rl_dream.yaml
```


## Some tips:

Not all configs have been tested, if you find a problem, feel free to raise a issue in this repo.

How to resume your training? Simply set `experiment.start_from_scratch = False` and `experiment.current_epoch` as your last RL step.

For model with large block size, use `max_active` smaller, like 16 for block size of 64.

If you have stagnation during inference, decrease `max_active`.

If you find RL training not stable, a common issue for RL, try increase `gradient_accumulation_steps`, `num_task_per_step`, and `num_response_per_task`. Note that in Open-Reaonser-Zero's experiments for llm, even using `gradient_update_step=1` (number of training/update step per RL step), `num_task_per_step=128` and `num_response_per_task=64` with GRPO, the training can be potentially instable (they use value model to stablize).


## Create your own configs:

Keep `experiment.project` same as the corresponding file name. You can first try some related example configs to get familiar with.

