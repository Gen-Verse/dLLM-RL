<div align="center">
  <br>
  <img src="assets/logo.png" width="200">
  <h3>Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models</h3>
  <h4>Most comprehensive framework for dLLM's and multimodal dLLM's post-training</h4>
</div>


<p align="center">
  <a href="https://arxiv.org/abs/2509.06949">
    <img
      src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&logoColor=red"
      alt="CURE Paper on arXiv"
    />
  <a href="https://huggingface.co/collections/Gen-Verse/trado-series-68beb6cd6a26c27cde9fe3af">
    <img 
        src="https://img.shields.io/badge/Datasets-Hugging%20Face%20Data-orange?logo=huggingface&logoColor=yellow" 
        alt="Coding Datasets on Hugging Face"
    />
  </a>
  <a href="https://huggingface.co/collections/Gen-Verse/trado-series-68beb6cd6a26c27cde9fe3af">
    <img 
        src="https://img.shields.io/badge/TraDo%204B/8B-Hugging%20Face%20Model-FFCC00?logo=huggingface&logoColor=yellow" 
        alt="ReasonFlux Coders on Hugging Face"
    />
  </a>
    <a href="https://yinjjiew.github.io/projects/dllmrl/">
    <img
      src="https://img.shields.io/badge/Blog-TraceRL-blue?logo=rss&logoColor=white"
      alt="Blog"
    />
  </a>
</p>




<p align="center">
  <img src="assets/figure1.png"  alt="Overview"  width="750">
</p>



## 🌱 Features 

- **Model Support**: [TraDo](https://arxiv.org/abs/2509.06949), [SDAR](https://github.com/JetAstra/SDAR), [Dream](https://github.com/DreamLM/Dream), [LLaDA](https://github.com/ML-GSAI/LLaDA), [MMaDA](https://github.com/Gen-Verse/MMaDA), [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V), and [Diffu-Coder](https://github.com/apple/ml-diffucoder) Almost all open-sourced discrete diffusion language models are supported here.
- **Diverse Settings**: We support deployment, **SFT**, **RL** (with **optional value model** for variance reduction and **process reward model** for fine-grained supervision), and **RLHF** across diverse settings (**math, coding, multimodal**) and different architectures (**both full/block attention dLLMs**).
- **Inference Acceleration**: improved [KV-cache](https://github.com/NVlabs/Fast-dLLM/tree/main), [jetengine](https://github.com/Labman42/JetEngine/tree/0ddc55ad3fb712b6374515b78d656f420e1a7243) (based on nano-vllm), different sampling strategies, support multi-nodes, easy to build your own accelerated inference methods.
- **RL Training**: [TraceRL (support diffusion value model)](https://arxiv.org/abs/2509.06949), [coupled RL](https://github.com/apple/ml-diffucoder), [random masking RL](https://github.com/Gen-Verse/MMaDA), accelerated sampling, including Math, coding, and general RL tasks, support multi-nodes, easy to build your reinforcement learning methods across diverse settings
- **SFT**: [Block SFT](https://github.com/kuleshov-group/bd3lms), semi-AR SFT, random masking SFT, support multi-nodes and long-CoT finetune.



## 🧠 RL Methods (TraceRL) & Models (TraDo)

We propose **TraceRL**, a trajectory-aware reinforcement learning method for diffusion language models, which demonstrates the best performance among RL approaches for DLMs. We also introduce a diffusion-based value model that reduces variance and improves stability during optimization.


<p align="center">
  <img src="assets/sft.png" width="48%"/>
  <img src="assets/rl.png" width="48%"/>
</p>

Based on TraceRL, we derive a series of diffusion language models, **TraDo**, which achieve state-of-the-art performance on math and coding reasoning tasks. TraDo-4B-Instruct and TraDo-8B-Instruct are trained solely with TraceRL, while the first long-CoT diffusion language model, TraDo-8B-Thinking, is obtained through a combination of TraceRL and long-CoT data SFT. TraDo models challenge AR models with strong empirical results, as shown in the following table.

<p align="center">
  <img src="assets/maintable.png"  alt="Main Table"  width="750">
</p>

We can download and try our model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import block_diffusion_generate

model_name = "Gen-Verse/TraDo-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype="float16", device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "What's the solution of x^2 - 2x + 1 = 0\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

tokens = tokenizer.batch_encode_plus([text], return_tensors='pt', padding=True, truncation=True, max_length=200)
tokens = {k: v.to(model.device) for k, v in tokens.items()}

output_ids = block_diffusion_generate(
    model,
    prompt=tokens,
    mask_id=151669,
    gen_length=200,
    block_length=4, denoising_steps=4,
    temperature=1.0, top_k=0, top_p=1.0,
    remasking_strategy="low_confidence_dynamic",
    confidence_threshold=0.9
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
cleaned_text = output_text.replace('<|MASK|>', '').replace('<|endoftext|>', '')
print(cleaned_text)


```

## 📰 Latest Updates
* **[2026-01-26]** Our [TraceRL](https://openreview.net/forum?id=KNAyc9DMe3) paper has been accepted by ICLR 2026! 
* **[2025-12-07]** 🔥 We support RLHF, fine-grained process reward, and multimodal RL/SFT now!
* **[2025-09-08]** We release our models, [TraDo-4B-Instruct](https://huggingface.co/Gen-Verse/TraDo-4B-Instruct) and [TraDo-8B-Instruct](https://huggingface.co/Gen-Verse/TraDo-8B-Instruct), and the long-CoT diffusion language model [TraDo-8B-Thinking](https://huggingface.co/Gen-Verse/TraDo-8B-Thinking).
* **[2025-09-08]** We release inference and training (SFT and RL) code compatible with a wide range of diffusion language models, including [TraDo](https://arxiv.org/abs/2509.06949), [SDAR](https://github.com/JetAstra/SDAR), [Dream](https://github.com/DreamLM/Dream), [LLaDA](https://github.com/ML-GSAI/LLaDA), [MMaDA](https://github.com/Gen-Verse/MMaDA), and [Diffu-Coder](https://github.com/apple/ml-diffucoder).


## 🚀 Quick Start


```bash
conda create --name dllm-rl python=3.10
source activate dllm-rl
pip install torch==2.6.0
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
# or requirements_v.txt for multimodal settings, see more details in the multimodal section in ./configs
```


## ⚙️ Data

You can navigate to `./data` to download datasets for evaluation and training, for example as follows. In that directory, you will also find detailed instructions on how to modify your own dataset.
```bash
cd data
python download_data.py --dataset MATH500
python download_data.py --dataset MATH_train
cd ..
```

After downloading the data, you are almost ready to evaluate or train diffusion language models. The only remaining step is to select (or create) a config file in `./configs` that corresponds to your project, and then use the following commands. Details on how to select and modify (or create) a config file are provided in `./configs`.


## 📊 Inference & Evaluations

After downloading the data, take TraDo models as an example. You can set the configurations in `configs/trado_eval.yaml` (see instructions and details in `./configs`) and run the following commands to perform inference with different sampling strategies.
```bash
python eval.py config=configs/trado_eval.yaml
# python eval.py config=configs/trado_longcot_eval.yaml
# python eval.py config=configs/sdar_eval.yaml
# python eval.py config=configs/dream_eval.yaml
# python eval.py config=configs/llada_eval.yaml
# python eval_v.py config=configs/lladav_eval.yaml
# python eval_v.py config=configs/mmada_v_eval.yaml
# see details in ./configs
```
Use `trado_eval.yaml` for TraDo models' inference, `sdar_eval.yaml` for SDAR, `dream_eval.yaml` for Dream and Diffu-Coder, and `llada_eval.yaml` for LLaDA and MMaDA. Instructions on how to set the configurations are provided in the corresponding configuration files.  
We support both general tasks and coding tasks (including automated execution of code) in evaluation.  

There are two main sampling methods you can choose:

**Static Sampling:** unmask fixed number of tokens each time

**Dynamic Sampling:** unmask tokens based on a chosen threshold, faster than static

To have a look how diffusion language models sample, open `./sample/trace.viewer.html` in your browser, or generate trajectory by your self with `./sample/get_trace_viewer.py`.


You can also perform inference across multiple nodes using `multinode_eval.py` with the same configuration files, with only minor modifications as instructed in the configuration files.
In multi-node setup, the first node controls the others. You can run  
`python multinode_eval.py config=configs/dream_multinode_eval.yaml` on the first node to eval, or submit the following as the entry command for a job:

```bash
if [[ ${MLP_ROLE_INDEX:-0} -eq 0 ]]; then   
    python multinode_eval.py config=configs/dream_multinode_eval.yaml
else
    exec tail -f /dev/null
fi
# python multinode_eval.py config=configs/trado_longcot_multinode_eval.yaml
# python multinode_eval.py config=configs/llada_multinode_eval.yaml
# python multinode_eval_v.py config=configs/lladav_eval.yaml
# python multinode_eval_v.py config=configs/mmada_v_eval.yaml
# ...
```


## 🔧 Reinforcement Learning

After downloading the data and model and setting the configuration, you can start reinforcement learning simply with:
```bash
python rl.py config=configs/rl_trado.yaml
# python rl.py config=configs/rl_sdar.yaml
# python rl.py config=configs/rl_dream.yaml
# python rl.py config=configs/rl_llada.yaml
# python rl.py config=configs/rl_mmada.yaml
# python rl_v.py config=configs/rl_lladav.yaml
# python rl_v.py config=configs/rl_mmada_v.yaml
# see details in ./configs
```

We support TraceRL (optionally with a diffusion-based value model), Coupled RL, and random masking RL across different diffusion language models. The sampling process has been accelerated in all cases by KV-cache.

**TraceRL**: We optimize the policy based on how it generates sequences. For block-attention models, training can be performed efficiently thanks to block attention. For full-attention models, we introduce a shrinkage parameter, s, that aggregates every s neighboring steps to accelerate training. We also provide a choice of value models for TraceRL, which we find can reduce variance and improve training stability, enabling the use of larger learning rates or fewer gradient accumulation steps more reliably than without using value model.


**Random Masking RL**: The sampled data are randomly masked and used as training data in RL with a PPO-like objective.


**Coupled RL**: For each sampled random masking setting, Coupled RL additionally introduces its complement, serving as an extra data sample for training.


We also support a multi-node RL framework; you can submit the following as the entry command:
```bash
if [[ ${MLP_ROLE_INDEX:-0} -eq 0 ]]; then   
    python multinode_rl.py config=configs/multinode_rl_trado.yaml
else
    exec tail -f /dev/null
fi
# python multinode_rl.py config=configs/multinode_rl_sdar.yaml
# python multinode_rl.py config=configs/multinode_rl_dream.yaml
# python multinode_rl.py config=configs/multinode_rl_llada.yaml
# python multinode_rl.py config=configs/multinode_rl_mmada.yaml
# python multinode_rl_v.py config=configs/multinode_rl_lladav.yaml
# python multinode_rl_v.py config=configs/multinode_rl_mmada_v.yaml
```

## 🔧 Supervised Finetuning

After downloading the data and setting the configurations, you can start supervised fine-tuning with:
```bash
accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 8888 \
  --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml \
  train/sft_trado.py \
  config=configs/sft_trado.yaml
# sft_sdar.py, sft_sdar.yaml
# sft_dream.py, sft_dream.yaml
# sft_llada.py, sft_llada.yaml
# sft_mmada.py, sft_mmada.yaml
# sft_mmada_v.py, sft_mmada_v.yaml
# sft_lladav.py, sft_lladav.yaml
# see details in ./configs
```

We support different SFT strategies for different models.

**Block diffusion models** (e.g., TraDo and SDAR): support semi-autoregressive fine-tuning or trace fine-tuning (requires setting a specific trace first).

**Adapted full-attention models** (e.g., Dream and DiffuCoder): support the semi-autoregressive method (using sliced data), random-masking SFT, and AR training (i.e., standard SFT for LLMs).

**Pretrained full-attention models** (e.g., LLaDA and MMaDA): support semi-autoregressive and random-masking SFT.



To use multi-node, simply run:
```bash
accelerate launch \
  --num_machines $MLP_WORKER_NUM \
  --machine_rank $MLP_ROLE_INDEX \
  --main_process_ip $MLP_WORKER_0_HOST \
  --main_process_port $MLP_WORKER_0_PORT \
  --config_file accelerate_configs/4_node_8_gpus_deepspeed_zero3.yaml \
  train/sft_dream.py \
  config=configs/sft_dream.yaml
# sft_trado.py, sft_trado.yaml
# ...
```


## 🤝 Acknowledgement

This work is heavily built on the following open-source models:

[SDAR](https://github.com/JetAstra/SDAR), [Dream](https://github.com/DreamLM/Dream), [LLaDA](https://github.com/ML-GSAI/LLaDA), [MMaDA](https://github.com/Gen-Verse/MMaDA/tree/main), [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V), and [Diffu-coder](https://github.com/apple/ml-diffuCoder).

these acceleration methods (engines):

[Fast-dllm](https://github.com/NVlabs/Fast-dLLM/tree/main), [jetengine](https://github.com/Labman42/JetEngine/tree/0ddc55ad3fb712b6374515b78d656f420e1a7243),

and theoretical foundations:

[MDLM](https://arxiv.org/pdf/2406.07524), [DiffuLLaMA](https://arxiv.org/abs/2410.17891), [Block Diffusion](https://arxiv.org/abs/2503.09573).


## 📖 Citation

```
@article{wang2025revolutionizing,
  title={Revolutionizing reinforcement learning framework for diffusion large language models},
  author={Wang, Yinjie and Yang, Ling and Li, Bowen and Tian, Ye and Shen, Ke and Wang, Mengdi},
  journal={arXiv preprint arXiv:2509.06949},
  year={2025}
}
```





