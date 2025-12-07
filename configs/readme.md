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
If use value model, use `rl_trado_with_value.yaml` or `multinode_rl_trado_with_value.yaml`. For process reward and RLHF, use `rl_trado_process_reward_rlhf_with_value.yaml` or `multinode_rl_trado_process_reward_rlhf_with_value.yaml`.

For SDAR models, use `rl_sdar.yaml` or `multinode_rl_sdar.yaml`. 
If use value model, use `rl_sdar_with_value.yaml` or `multinode_rl_sdar_with_value.yaml`. For process reward and RLHF, use `rl_sdar_process_reward_rlhf_with_value.yaml` or `multinode_rl_sdar_process_reward_rlhf_with_value.yaml`.

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








## 🌋 Multimodal Training with dLLM-RL: MMaDA & LLaDA-V

This document serves as a supplementary guide to the dLLM-RL framework, focusing specifically on the training and evaluation of multimodal models. We introduce support for **MMaDA** and **LLaDA-V**, enabling advanced multimodal understanding capabilities within the diffusion language model paradigm. By extending our reinforcement learning and supervised fine-tuning pipelines to the multimodal domain, we aim to unlock new potentials in how diffusion models process and reason with visual information alongside text.

### 🛠️ Environment Preparation

To get started, you will need to set up two separate Conda environments: a **main environment** for general training/inference and a **vLLM environment** specifically for accelerated generation and evaluation.

#### 1. Main Environment
Install the core dependencies using the provided requirements file:
```bash
pip install -r requirements_v.txt
```

#### 2. vLLM Environment
This environment is crucial for running evaluations and efficient inference. You need a version of `vllm` that supports at least **Qwen3-VL-MoE**, as our experiments are conducted on **Qwen3-VL-8B-Thinking**.

*   **Installation Guide**: Please refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/) for the best installation instructions tailored to your CUDA version. `vllm` has strict compatibility requirements for PyTorch and CUDA.
*   **Example (CUDA 12.6)**:
    ```bash
    pip install https://github.com/vllm-project/vllm/releases/download/v0.12.0/vllm-0.12.0-cp38-abi3-manylinux_2_31_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu126
    ```
*   **Additional Dependencies**: After installing `vllm`, install these required packages in the same environment:
    ```bash
    pip install termcolor qwen_vl_utils omegaconf
    ```

---

### 📦 Model Preparation

#### MMaDA Specifics
For **MMaDA**, the **VQ-VAE** component is required for visual tokenization.
1.  Download **showlab/magvitv2**.
2.  Update your YAML configuration file:
    *   Locate the `vq_model_path` field.
    *   Fill in the absolute path to your downloaded `magvitv2` model.

---

### 📂 Data Preparation

We categorize data into two types: **SFT Data** (Supervised Fine-Tuning) and **RL/Eval Data** (Reinforcement Learning & Evaluation).

#### Datasets
*   **SFT**: `data/sft_test_v.json` (provided as an example).
*   **RL & Eval**: `data/SEEDBench_IMG_32.json` (provided as an example).

#### Image Directory
All images used in these datasets should be stored in a unified directory, e.g., `data/images/`.
*   **Configuration**: In your YAML config files, set `image_root` to this directory path.
*   **Usage**: The system will automatically join `image_root` with the image paths defined in the dataset JSON files.

---

### 📊 Evaluation

We support both single-node and multi-node evaluation.

### Single-Node Evaluation
To evaluate LLaDA-V:
```bash
python eval_v.py config=configs/lladav_eval.yaml
```
To evaluate MMaDA:
```bash
python eval_v.py config=configs/mmada_v_eval.yaml
```

#### Multi-Node Evaluation
We also support multi-node evaluation for large-scale testing.

For LLaDA-V:
```bash
if [[ ${MLP_ROLE_INDEX:-0} -eq 0 ]]; then   
    python multinode_eval_v.py config=configs/lladav_eval.yaml
else
    exec tail -f /dev/null
fi
```

For MMaDA:
```bash
if [[ ${MLP_ROLE_INDEX:-0} -eq 0 ]]; then   
    python multinode_eval_v.py config=configs/mmada_v_eval.yaml
else
    exec tail -f /dev/null
fi
```

---

### 🚀 Reinforcement Learning (RL)

#### Single-Node RL
To train LLaDA-V with RL:
```bash
python rl_v.py config=configs/rl_lladav.yaml
```
To train MMaDA with RL:
```bash
python rl_v.py config=configs/rl_mmada_v.yaml
```

#### Multi-Node RL
For distributed RL training, use the following command structure (example for LLaDA-V):

```bash
if [[ ${MLP_ROLE_INDEX:-0} -eq 0 ]]; then   
    python multinode_rl_v.py config=configs/multinode_rl_lladav.yaml
else
    exec tail -f /dev/null
fi
```
*For MMaDA, use `configs/multinode_rl_mmada_v.yaml`.*

> **Note on RL Methods & Configurations:**
> *   **Random / Coupled-GRPO / TraceRL (No Value Model)**: Use the standard YAML files mentioned above (e.g., `rl_lladav.yaml` or `multinode_rl_lladav.yaml`). Modify the `training.method` field in the config to select your desired method (e.g., `random`, `coupled`, `tracerl`).
> *   **Multimodal Process Reward / Fine-Grained Alignment**: For methods involving process rewards as described in the paper, please use the configuration files ending in `process_reward_rlhf_with_value`.
> *   **Value Model Only**: If you wish to use methods that only utilize a value model, please use the configuration files ending in `with_value`.

---

### 🔧 Supervised Fine-Tuning (SFT)

#### Single-Node SFT
Use `accelerate` to launch the SFT training.

**LLaDA-V:**
```bash
accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 8888 \
  --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml \
  train/sft_lladav.py \
  config=configs/sft_lladav.yaml
```

**MMaDA:**
*   Script: `train/sft_mmada_v.py`
*   Config: `configs/sft_mmada_v.yaml`

#### Multi-Node SFT
For training on multiple nodes (e.g., 4 nodes), use the environment variables for worker configuration.

**LLaDA-V:**
```bash
accelerate launch \
  --num_machines $MLP_WORKER_NUM \
  --machine_rank $MLP_ROLE_INDEX \
  --main_process_ip $MLP_WORKER_0_HOST \
  --main_process_port $MLP_WORKER_0_PORT \
  --config_file accelerate_configs/4_node_8_gpus_deepspeed_zero3.yaml \
  train/sft_lladav.py \
  config=configs/sft_lladav.yaml
```

**MMaDA:**
*   Script: `train/sft_mmada_v.py`
*   Config: `configs/sft_mmada_v.yaml`









## Some tips:

Not all configs have been tested, if you find a problem, feel free to raise a issue in this repo.

How to resume your training? Simply set `experiment.start_from_scratch = False` and `experiment.current_epoch` as your last RL step.

For model with large block size, use `max_active` smaller, like 16 for block size of 64.

If you have stagnation during inference, decrease `max_active`.

When you are using process reward/rlhf for trado or sdar models, make sure your environment support vllm and jetengine at the same time. One example usage: vllm==0.8.5.post1, torch==2.6.0, python3.10, triton==3.2.0, flash-attn==2.7.4.post1, flashinfer-python==0.3.1

If you find RL training not stable, a common issue for RL, try increase `gradient_accumulation_steps`, `num_task_per_step`, and `num_response_per_task`. Note that in Open-Reaonser-Zero's experiments for llm, even using `gradient_update_step=1` (number of training/update step per RL step), `num_task_per_step=128` and `num_response_per_task=64` with GRPO, the training can be potentially instable (they use value model to stablize).


## Create your own configs:

Keep `experiment.project` same as the corresponding file name. You can first try some related example configs to get familiar with.

