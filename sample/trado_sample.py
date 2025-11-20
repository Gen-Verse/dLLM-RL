import os as _os
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")



# Consolidate all caches into the local high-speed disk (NVMe or /dev/shm)
# Local high-speed cache (NVMe or /dev/shm)
_cache_root = "/dev/shm/torch_cache"       # or "/local_nvme/torch_cache"
_os.makedirs(_cache_root, exist_ok=True)
_os.environ["TORCH_EXTENSIONS_DIR"] = _os.path.join(_cache_root, "torch_extensions")
_os.environ["TRITON_CACHE_DIR"]      = _os.path.join(_cache_root, "triton")
_os.environ["XDG_CACHE_HOME"]        = _cache_root
_os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")


_os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
_os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
_os.environ.pop("NCCL_BLOCKING_WAIT", None)
_os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

import os
import re
import json
from termcolor import cprint
import random
import torch.multiprocessing as mp
from jinja2 import Template

from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf



# obtain prompt
def get_prompt(data_i):
    return Template(system_prompts, keep_trailing_newline=True).render(problem = data_i["question"])

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




def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output


'''
def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node 
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]
'''
def get_data_chunk(data, num_nodes, node_idx):
    total = len(data)
    start = (total * node_idx) // num_nodes
    end   = (total * (node_idx + 1)) // num_nodes
    return data[start:end]


import socket

def _patch_safe_destroy():
    import torch.distributed as dist
    _real_destroy = dist.destroy_process_group
    def _safe_destroy(group=None):
        try:
            if not dist.is_available():
                return
            try:
                if not dist.is_initialized():
                    return
            except Exception:
                return
            _real_destroy(group)
        except AssertionError:
            pass
    dist.destroy_process_group = _safe_destroy



def _llm_worker_run(args):
    (model_path, tp, block_size, sampling_kwargs, vis_ids,
     prompts_slice, indices_slice, enforce_eager, max_active, store_port) = args

    import os
    # 1) Setup environment (critical for correct worker behavior)
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.pop("NCCL_BLOCKING_WAIT", None)
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, vis_ids))
    #os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_ext_worker_{store_port}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(store_port)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 1.1) Create a per-worker `sitecustomize.py` and inject it into PYTHONPATH 
    # (this must be done before importing torch/jetengine)
    patch_dir = f"/tmp/je_site_{store_port}"
    os.makedirs(patch_dir, exist_ok=True)
    patch_file = os.path.join(patch_dir, "sitecustomize.py")
    # Important: the content must start at column 0 (no indentation)!
    with open(patch_file, "w") as _f:
        _f.write(
            "import os\n"
            "import torch.distributed as dist\n"
            "_real = dist.init_process_group\n"
            "def _wrapped(backend, init_method=None, *args, **kwargs):\n"
            "    port = os.environ.get('JE_TCP_PORT')\n"
            "    if port and isinstance(init_method, str) and init_method.startswith('tcp://localhost:2333'):\n"
            "        init_method = f'tcp://127.0.0.1:{port}'\n"
            "    return _real(backend, init_method, *args, **kwargs)\n"
            "dist.init_process_group = _wrapped\n"
        )
    os.environ["PYTHONPATH"] = patch_dir + (":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")
    os.environ["JE_TCP_PORT"] = str(store_port)

    # 2) Import torch and patch the current worker process
    import torch
    import torch.distributed as dist
    _patch_dist_port(store_port)   # Patch port binding for this process
    _patch_safe_destroy()          # Avoid AssertionError in destroy_process_group
    torch.cuda.set_device(0)       # From this worker’s perspective, cuda:0 is the first visible device

    # For debugging: print the worker’s CUDA_VISIBLE_DEVICES and assigned port
    print(f"[worker pid={os.getpid()}] CVD={os.environ['CUDA_VISIBLE_DEVICES']}, port={store_port}, prompts={len(prompts_slice)}", flush=True)

    # 3) Import jetengine and create the engine 
    # (child processes inherit the sitecustomize patch)
    from jetengine_ext.llm import LLM
    from jetengine_ext.sampling_params import SamplingParams

    llm = None
    triples = []
    try:
        llm = LLM(
            model_path,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tp,
            mask_token_id=151669,
            block_length=block_size
        )
        sp = SamplingParams(**sampling_kwargs)

        # Keep max_active sane for each worker’s slice to avoid rare internal exits
        local_max_active = min(max_active, max(1, len(prompts_slice)))
        outs = llm.generate_streaming(prompts_slice, sp, max_active=local_max_active)

        # Collect results incrementally so we can return partials on any exit
        for j, o in enumerate(outs):
            triples.append((
                indices_slice[j],
                o["text"],
                o.get("first_unmask_times", None)
            ))
    except BaseException as e:
        # Swallow SystemExit/KeyboardInterrupt/etc. so we can return partials
        print(f"[worker pid={os.getpid()}] Caught {type(e).__name__}: {e}. Returning partial results ({len(triples)})", flush=True)
    finally:
        try:
            if llm is not None and hasattr(llm, "shutdown"):
                llm.shutdown()
        except Exception:
            pass

    return triples



def _llm_worker_entry(args, out_q):
    import traceback, os
    try:
        res = _llm_worker_run(args)
        # Even if partial, report as 'ok' so parent can use what we have
        out_q.put(("ok", res))
    except BaseException:
        tb = traceback.format_exc()
        # Fall back to 'err' path if even the call above exploded
        try:
            out_q.put(("err", {
                "pid": os.getpid(),
                "port": args[-1],
                "traceback": tb,
            }))
        except Exception:
            pass


def _find_free_port():
    s = socket.socket(); s.bind(('', 0))
    p = s.getsockname()[1]; s.close()
    return p

def _patch_dist_port(port: int):
    import torch.distributed as _dist
    _real_init = _dist.init_process_group

    def _wrapped(backend, init_method=None, *args, **kwargs):
        # jetengine internally hardcodes "tcp://localhost:2333" — replace the port here
        if isinstance(init_method, str) and init_method.startswith("tcp://localhost:2333"):
            init_method = f"tcp://127.0.0.1:{port}"
        return _real_init(backend, init_method, *args, **kwargs)

    _dist.init_process_group = _wrapped



if __name__ == "__main__":

    config = get_config()

    

    tp = int(get_config().rollout.tensor_parallel_size)  # Or check after loading config

    if tp == 1:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        # These two are NCCL’s own variables, keep using the NCCL_ prefix
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
    else:
        # For multi-GPU communication, do not disable P2P/IB;
        # also clean up related variables (both old and new names)
        for k in [
            "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
            "TORCH_NCCL_BLOCKING_WAIT", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "NCCL_BLOCKING_WAIT", "NCCL_ASYNC_ERROR_HANDLING",
        ]:
            os.environ.pop(k, None)



    from transformers import AutoTokenizer

    # --- graceful shutdown & unique port ---
    import os, sys, atexit, signal, torch.distributed as dist

    # 2) Automatically set compile architecture according to the local GPU
    # (do NOT hardcode 8.0)
    def _set_arch():
        try:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:
            pass
    _set_arch()

    

    # 1) Use a new port at each startup to avoid conflicts with 2333
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_find_free_port())
    # (If JetEngine hardcodes tcp://localhost:2333 instead of using env://,
    # see the “special case” section at the end)

    # 2) Intercept Ctrl-C/TERM to destroy distributed groups & engine gracefully
    _llm = None
    _child_ps = []    # If you create your own mp.Process/Pool, append objects here

    def _cleanup():
        # 2.1) Shutdown JetEngine engine (if API available)
        global _llm
        try:
            if _llm is not None and hasattr(_llm, "shutdown"):
                _llm.shutdown()
        except Exception:
            pass
        # 2.3) Kill/join child processes
        for p in _child_ps:
            try:
                if hasattr(p, "terminate"): p.terminate()
            except Exception:
                pass
        for p in _child_ps:
            try:
                if hasattr(p, "join"): p.join(timeout=2)
            except Exception:
                pass

    atexit.register(_cleanup)
    def _sig_handler(sig, frame):
        _cleanup()
        # 130: standard exit code for SIGINT; 143: for SIGTERM
        sys.exit(130 if sig == signal.SIGINT else 143)

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)








    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    
    k_sample = config.rollout.num_response_per_task
    # non-cot prompt
    #system_prompts = '''<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''
    # cot prompt
    system_prompts = '''<|im_start|>user\n{{problem}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'''
    if config.rollout.start_with_think:
        system_prompts = '''<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant<think>\n'''
    
    project_name = config.experiment.project

    code_eval = False

    dataset = config.dataset.eval_dataset
    pretrained_model = config.model
    if config.dataset.data_type == "code":
        code_eval = True
        system_prompts_function = '''<|im_start|>user\n{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>\n<|im_start|>assistant\n'''
        system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant\n'''
        if config.rollout.start_with_think:
            system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant<think>\n'''
    elif config.dataset.data_type == "option":
        system_prompts = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}. <|im_end|>\n<|im_start|>assistant\n'''
        if config.rollout.start_with_think:
            system_prompts = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}. <|im_end|>\n<|im_start|>assistant<think>\n'''
    
    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    with open("../data/" + dataset + ".json", 'r') as f:
        data = json.load(f)
    #data = [data[i] for i in range(100, 300)]
    #data = data[400:]

    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    if num_node > 1:
        #random.shuffle(data)
        data = get_data_chunk(data, num_node, node_index)
    
    num = len(data)


    model_path = os.path.expanduser(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Initialize the LLM

    block_size = config.rollout.block_size
    


    

    

    



    # initialization
    generation_prompts = []
    prefix_list = []
    index_list = []
    for i in range(num):
        # preprocess
        if code_eval:
            if data[i]["test_method"] == "stdio":
                system_prompts = system_prompts_stdio
                prefix_list = prefix_list + [None] * k_sample
            else:
                system_prompts = system_prompts_function + data[i]["prefix"]
                prefix_list = prefix_list + [data[i]["prefix"]] * k_sample
        generation_prompts = generation_prompts + [get_prompt(data[i])] * k_sample
        
        index_list = index_list + [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = get_prompt(data[i])
    




    # --------------------------- 1. shuffle --------------------------
    cprint("start generation...", "green")

    all_prompts = generation_prompts
    N = len(all_prompts)

    shuffled_idx     = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]


    import torch, math
    print(f"[preflight] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[preflight] parent sees torch.cuda.device_count()={torch.cuda.device_count()}")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        visible_gpus = [x.strip() for x in cvd.split(",") if x.strip() != ""]
        device_ids = [int(x) for x in visible_gpus]         
    else:
        device_ids = list(range(torch.cuda.device_count()))
    
    gpu_num = len(device_ids)     
    tp = int(config.rollout.tensor_parallel_size)
    assert gpu_num >= tp, f"Visible GPUs ({gpu_num}) < tensor_parallel_size ({tp})."
    assert gpu_num >= 1, "No GPU visible"
    if tp > 1:
        ngroups = 1
    else:
        ngroups = max(1, gpu_num // max(1, tp))
    
    groups = [ device_ids[i*tp : (i+1)*tp] for i in range(ngroups) ]



    def to_single_token_stop_ids(tokenizer, stop_token_list):
        if not stop_token_list:
            return []
        ids, seen = [], set()
        for s in stop_token_list:
            if isinstance(s, int):
                tid = [s]
            elif isinstance(s, str):
                tid = tokenizer.encode(s, add_special_tokens=False)
            elif isinstance(s, (list, tuple)) and all(isinstance(x, int) for x in s):
                tid = list(s)
            else:
                continue  
            if len(tid) == 1:
                t = tid[0]
                if t not in seen:
                    seen.add(t)
                    ids.append(t)
        return ids
    
    from omegaconf import MISSING
    if OmegaConf.select(config, "rollout.stop_token_list", default=MISSING) is not MISSING:
        stop_token_id_list = to_single_token_stop_ids(tokenizer, config.rollout.stop_token_list)
    else:
        stop_token_id_list = []

    sampling_kwargs = dict(
        temperature          = config.rollout.temperature,
        topk                 = config.rollout.top_k,
        topp                 = config.rollout.top_p,
        max_tokens           = config.rollout.max_token,
        remasking_strategy   = config.rollout.remasking_strategy,
        block_length         = block_size,
        denoising_steps      = config.rollout.denoising_steps_per_block,
        dynamic_threshold    = config.rollout.dynamic_threshold,
        stop_words           = stop_token_id_list
    )
    max_active_local = config.rollout.max_active

    def _chunk_by_groups(lst, ng):
        L = len(lst)
        if ng <= 1: return [lst]
        chunk_size = math.ceil(L / ng)
        return [ lst[i*chunk_size : min((i+1)*chunk_size, L)] for i in range(ng) ]

    prompt_chunks = _chunk_by_groups(shuffled_prompts, ngroups)
    index_chunks  = _chunk_by_groups(shuffled_idx,     ngroups)

    for a, b in zip(prompt_chunks, index_chunks):
        assert len(a) == len(b)

    seq_pairs = []

    if ngroups == 1:
        from jetengine_ext.llm import LLM
        from jetengine_ext.sampling_params import SamplingParams

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, groups[0]))
        import torch
        torch.cuda.set_device(0)

        if config.rollout.tensor_parallel_size > 1:
            enforce_eager = False
        else:
            enforce_eager = True
        llm = LLM(
            model_path,
            enforce_eager=enforce_eager,
            tensor_parallel_size=config.rollout.tensor_parallel_size,
            mask_token_id=151669,   # Optional: only needed for masked/diffusion models
            block_length=block_size
        )
        _llm = llm

        # Set sampling/generation parameters
        sampling_params = SamplingParams(
            temperature=config.rollout.temperature,
            topk=config.rollout.top_k,
            topp=config.rollout.top_p,
            max_tokens=config.rollout.max_token,
            remasking_strategy=config.rollout.remasking_strategy,
            block_length=block_size,
            denoising_steps=config.rollout.denoising_steps_per_block,
            dynamic_threshold=config.rollout.dynamic_threshold,
            stop_words           = stop_token_id_list
        )
        try:
            outputs = llm.generate_streaming(prompt_chunks[0], sampling_params, max_active=config.rollout.max_active)
            for j, o in enumerate(outputs):
                seq_pairs.append( (
                    index_chunks[0][j],
                    o["text"],
                    o.get("first_unmask_times", None)
                ) )
        finally:
            _cleanup()
    else:
        import time
        ctx = mp.get_context("spawn")
        enforce_eager_local = False if tp > 1 else True

        base_port = 29000
        store_ports = [base_port + g for g in range(ngroups)]

        out_q = ctx.Queue()
        procs = []
        for g in range(ngroups):
            if len(prompt_chunks[g]) == 0:
                continue
            args = (
                model_path, tp, block_size, sampling_kwargs, groups[g],
                prompt_chunks[g], index_chunks[g],
                enforce_eager_local, max_active_local, store_ports[g],
            )
            p = ctx.Process(target=_llm_worker_entry, args=(args, out_q), daemon=False)
            p.start()
            #time.sleep(2)
            procs.append(p)
            _child_ps.append(p)   

        import queue, time

        results_needed = len(procs)
        results_got = 0

        while results_got < results_needed:
            try:
                kind, payload = out_q.get(timeout=3600 * 24) 
            except queue.Empty:
                dead = [p for p in procs if not p.is_alive()]
                if dead:
                    for p in dead:
                        print(f"[parent] worker pid={p.pid} exitcode={p.exitcode} (no result)", flush=True)
                    for p in procs:
                        if p.is_alive():
                            p.terminate()
                    for p in procs:
                        p.join(timeout=5)
                    raise RuntimeError("Some workers died without returning results. See logs above.")
                continue

            if kind == "ok":
                seq_pairs.extend(payload)
                results_got += 1
            else:  # "err"
                print(f"[parent] worker error on port {payload['port']} pid {payload['pid']}:\n{payload['traceback']}", flush=True)
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                for p in procs:
                    p.join(timeout=5)
                raise RuntimeError("Worker failed. See traceback above.")

        for p in procs:
            p.join()


    # ------------------- 3. restore original order -------------------


    restored_outputs = [None] * N
    restored_steps   = [None] * N

    for item in seq_pairs:
        if len(item) == 2:
            gi, text = item
            steps = None
        else:
            gi, text, steps = item
        restored_outputs[gi] = text
        restored_steps[gi]   = steps


    for i in range(N):
        if restored_outputs[i] is None:
            restored_outputs[i] = ""
        if restored_steps[i] is None:
            restored_steps[i] = ""

    cprint("generation job done!", "green")






    def get_token_lengths(strings, tokenizer):
        pad_token = tokenizer.pad_token

        escaped = re.escape(pad_token)
        pattern = rf"(?:{escaped})+"
        remove_pattern = escaped

        collapse_re = re.compile(pattern)

        lengths = []
        for s in strings:
            s_clean = collapse_re.sub(lambda _: pad_token if isinstance(pad_token, str) else '', s)
            s_clean = re.sub(remove_pattern, '', s_clean)
            lengths.append(len(tokenizer.encode(s_clean, add_special_tokens=False)))
        return lengths

    response_length = get_token_lengths(restored_outputs, tokenizer)
    mean_response_length = sum(response_length) / len(response_length)




    # process generated codes
    i = 0
    for full_output in restored_outputs:
        if code_eval:
            if data[int(i/k_sample)]["test_method"] == "function":
                extracted_output = extract_code(prefix_list[i] + full_output)
            else:
                extracted_output = extract_code(full_output)
        else:
            extracted_output = extract_final_boxed_answer(full_output)
        index_i = index_list[i]
        data[index_i]["full_output"].append(full_output)
        step_map_i = restored_steps[i] if restored_steps[i] is not None else []
        #print(step_map_i)
        data[index_i]["step_map"].append(step_map_i)
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])
        i += 1

    # output the data
    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



