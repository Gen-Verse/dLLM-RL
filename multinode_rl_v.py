import os
import subprocess
import shlex
from omegaconf import DictConfig, ListConfig, OmegaConf


def run_local(cmd: str, check: bool = True) -> None:
    subprocess.run(f"bash -lc {shlex.quote(cmd)}", shell=True, check=check)

def run_local_async(cmd: str) -> subprocess.Popen:
    return subprocess.Popen(f"bash -lc {shlex.quote(cmd)}", shell=True)

def run_remote(host: str, cmd: str, check: bool = True) -> None:
    ssh_cmd = f'ssh root@{host} "bash -lc {shlex.quote(cmd)}"'
    subprocess.run(ssh_cmd, shell=True, check=check)

def run_remote_async(host: str, cmd: str) -> subprocess.Popen:
    ssh_cmd = f'ssh root@{host} "bash -lc {shlex.quote(cmd)}"'
    return subprocess.Popen(ssh_cmd, shell=True)


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)

def begin_with(file_name: str):
    with open(file_name, "w"):
        pass



def make_init_bash(cfg) -> str:
    sc = cfg.system 
    http_proxy  = sc.HTTP_PROXY
    https_proxy = sc.HTTP_PROXY
    hf_home     = sc.HF_HOME
    envs_dir    = sc.envs_dir

    lines = []
    lines.append("set -e")
    if http_proxy is not None:
        lines.append(f"echo 'export HTTP_PROXY={http_proxy}' >> ~/.bashrc")
    if https_proxy is not None:
        lines.append(f"echo 'export HTTPS_PROXY={https_proxy}' >> ~/.bashrc")
    if hf_home is not None:
        lines.append(f"echo 'export HF_HOME={hf_home}' >> ~/.bashrc")
    lines.append("") 

    if envs_dir is not None:
        lines.append(f"conda config --append envs_dirs {envs_dir} || true")
        lines.append("") 

    lines.append("echo 'source ~/.bashrc' >> ~/.bash_profile")
    lines.append("")

    return "\n".join(lines)








if __name__ == "__main__":


    def init_node(host: str):
        run_remote(host, INIT_BASH, check=False)

    def init_hosts(worker_hosts):
        for h in worker_hosts:
            init_node(h)


    def env_prefix(env) -> str:
        return (
            "source ~/.bashrc && "
            f"source activate {env} && "
        )


    def sample(worker_hosts, epoch, cfg, type, model_base, block_size = None, top_k = None, remasking_strategy = None):
        project = cfg.experiment.project
        procs = []
        if model_base == "dream":
            script_name = "dream_rl_rollout.py"
        elif model_base == "llada" or model_base == "mmada":
            script_name = "mmada_v_rl_rollout.py"
        elif model_base == "lladav":
            script_name = "lladav_rl_rollout.py"
        elif model_base == "sdar":
            script_name = "sdar_rl_rollout.py"
        elif model_base == "trado":
            script_name = "trado_rl_rollout.py"
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR}/sample && "
                f"python {script_name} "
                f"config=../configs/{project}.yaml "
                f"experiment.current_epoch={epoch} "
                f"experiment.function={type} "
                f"evaluation.top_k={top_k} "
                f"evaluation.remasking_strategy={remasking_strategy} "
                f"evaluation.block_size={block_size} "
                f"experiment.node_index={idx}"
            )
            full_cmd = env_prefix(env_name) + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()


    def execute(worker_hosts, epoch, cfg, type):
        project = cfg.experiment.project
        procs = []
        for idx, host in enumerate(worker_hosts):
            full_cmd = env_prefix(env_name) + (
                f"cd {BASE_DIR}/reward && "
                f"python rl_execute.py "
                f"config=../configs/{project}.yaml "
                f"experiment.function={type} "
                f"experiment.current_epoch={epoch} "
                f"experiment.node_index={idx}"
            )
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()


    def aggregate(epoch, cfg, type):
        project = cfg.experiment.project
        full_cmd = env_prefix(env_name) + (
            f"cd {BASE_DIR}/reward && "
            f"python rl_aggregate_data_v.py "
            f"config=../configs/{project}.yaml "
            f"experiment.function={type} "
            f"experiment.current_epoch={epoch}"
        )
        run_local(full_cmd)


    def reward(epoch, cfg, type, is_code_task, block_size = None, top_k = None, remasking_strategy = None):
        project = cfg.experiment.project
        if is_code_task:
            script_name = "rl_code_reward.py"
        else:
            script_name = "rl_reward_v.py"
        full_cmd = env_prefix(env_name) + (
            f"cd {BASE_DIR}/reward && "
            f"python {script_name} "
            f"config=../configs/{project}.yaml "
            f"experiment.function={type} "
            f"evaluation.block_size={block_size} "
            f"evaluation.top_k={top_k} "
            f"evaluation.remasking_strategy={remasking_strategy} "
            f"experiment.current_epoch={epoch}"
        )
        run_local(full_cmd)


    def train(worker_hosts, epoch, cfg, model_base, target = None):
        project = cfg.experiment.project
        ds_file = cfg.experiment.deepspeed_file
        num_nodes = len(worker_hosts)
        master_ip = os.environ["MLP_WORKER_0_HOST"]
        master_port = os.environ["MLP_WORKER_0_PORT"]
        if target is None:
            if model_base == "dream":
                script_name = "rl_dream.py"
            elif model_base == "llada":
                script_name = "rl_llada.py"
            elif model_base == "mmada":
                script_name = "rl_mmada_v.py"
            elif model_base == "lladav":
                script_name = "rl_lladav.py"
            elif model_base == "sdar":
                script_name = "rl_sdar.py"
            elif model_base == "trado":
                script_name = "rl_trado.py"
        elif target == "policy":
            if model_base == "sdar":
                script_name = "train_sdar_policy.py"
            elif model_base == "trado":
                script_name = "train_trado_policy.py"
            elif model_base == "mmada":
                script_name = "train_mmada_v_policy.py"
            elif model_base == "lladav":
                script_name = "train_lladav_policy.py"
        elif target == "value":
            if model_base == "sdar":
                script_name = "train_sdar_value.py"
            elif model_base == "trado":
                script_name = "train_trado_value.py"
            elif model_base == "mmada":
                script_name = "train_mmada_v_value.py"
            elif model_base == "lladav":
                script_name = "train_lladav_value.py"
        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR} && "
                "export DS_SKIP_CUDA_CHECK=1 && "
                "accelerate launch "
                f"--num_machines {num_nodes} "
                f"--machine_rank {idx} "
                f"--main_process_ip {master_ip} "
                f"--main_process_port {master_port} "
                f"--config_file accelerate_configs/{ds_file}.yaml "
                f"train/{script_name} "
                f"config=configs/{project}.yaml "
                f"experiment.current_epoch={epoch}"
            )
            full_cmd = env_prefix(env_name) + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
            print(f"[DISPATCH] train node {idx} → {host}")
        for p in procs:
            p.wait()
        print("All train nodes finished.")



    def init_value_model(epoch, model_base, cfg):
        project = cfg.experiment.project
        if model_base == "sdar":
            script_name = "init_sdar_value_model.py"
        elif model_base == "trado":
            script_name = "init_trado_value_model.py"
        elif model_base == "mmada":
            script_name = "init_mmada_v_value_model.py"
        elif model_base == "lladav":
            script_name = "init_lladav_value_model.py"
        full_cmd = env_prefix(env_name) + (
            f"cd {BASE_DIR}/train && "
            f"python {script_name} "
            f"config=../configs/{project}.yaml "
            f"experiment.current_epoch={epoch}"
        )
        run_local(full_cmd)

    def process_reward(worker_hosts, epoch, cfg):
        project = cfg.experiment.project

        # --- Step A: node0 preprocess ---
        cmd_div = (
            env_prefix(env_name) +
            f"cd {BASE_DIR}/reward && "
            f"python rl_vprocess_divide_data.py "
            f"config=../configs/{project}.yaml "
            f"experiment.current_epoch={epoch}"
        )
        run_local(cmd_div)
        print("[OK] rl_process_divide_data.py (node0) done.")

        # --- Step B: multinode LLM rewarding ---
        procs = []
        for idx, host in enumerate(worker_hosts):
            cmd_llm = (
                env_prefix(reward_env_name) +
                f"cd {BASE_DIR}/sample && "
                f"python vlm_process_reward.py "
                f"config=../configs/{project}.yaml "
                f"experiment.current_epoch={epoch} "
                f"experiment.node_index={idx}"
            )
            if idx == 0:
                procs.append(run_local_async(cmd_llm))
            else:
                procs.append(run_remote_async(host, cmd_llm))
            print(f"[DISPATCH] llm_process_reward node {idx} → {host}")
        for p in procs:
            p.wait()
        print("[OK] llm_process_reward.py (all nodes) done.")

        # --- Step C: node0 aggregate ---
        aggregate(epoch, cfg, "train")
        print("[OK] aggregation done.")

        # --- Step D: node0 final process ---
        cmd_fin = (
            env_prefix(reward_env_name) +
            f"cd {BASE_DIR}/reward && "
            f"python rl_vprocess_reward.py "
            f"config=../configs/{project}.yaml "
            f"experiment.current_epoch={epoch}"
        )
        run_local(cmd_fin)
        print("[OK] rl_process_reward.py (node0) done.")

    from omegaconf import MISSING

    cfg = get_config()
    INIT_BASH = make_init_bash(cfg)
    BASE_DIR = cfg.system.base_dir
    env_name = cfg.system.env_name
    if OmegaConf.select(cfg, "system.reward_env_name", default=MISSING) is not MISSING:
        reward_env_name = cfg.system.reward_env_name
    total_step = cfg.experiment.total_step
    project = cfg.experiment.project
    num_node = cfg.experiment.num_node
    worker_hosts = [os.environ[f"MLP_WORKER_{i}_HOST"] for i in range(num_node)]

    
    if OmegaConf.select(cfg, "model.value_base_model", default=MISSING) is not MISSING:
        have_value_model = True
    else:
        have_value_model = False

    import time
    time.sleep(30)

    init_hosts(worker_hosts)

    import time
    time.sleep(10)

    model_base = cfg.model.model_base

    if cfg.experiment.start_from_scratch:
        os.makedirs(f"{project}/results", exist_ok=True)
        optimized = f"../{project}/ckpt/{cfg.model.optimized_name}"
        path = (
            f"{project}/results/results-rl-"
            f"{optimized.replace('/', '.')}-"
            f"{cfg.dataset.train_dataset}.txt"
        )
        begin_with(path)
        path = (
            f"{project}/results/results-eval-"
            f"{optimized.replace('/', '.')}-"
            f"{cfg.evaluation.eval_dataset}.txt"
        )
        begin_with(path)
        if have_value_model:
            init_value_model(1, model_base, cfg)
            optimized_value = f"../{project}/ckpt/{cfg.model.optimized_value_name}"
            path = (
                f"{project}/results/results-rl-"
                f"{optimized_value.replace('/', '.')}-"
                f"{cfg.dataset.train_dataset}.txt"
            )
            begin_with(path)


    epoch = cfg.experiment.current_epoch

    if OmegaConf.select(cfg, "model.process_reward_model", default=MISSING) is not MISSING and cfg.model.process_reward_model is not None:
        is_process_reward = True
    else:
        is_process_reward = False
    
    if cfg.dataset.data_type == "code":
        is_code_task = True
    else:
        is_code_task = False
    
    while epoch <= total_step:
        print(f"\n========== epoch {epoch} ==========")
        
        
        sample(worker_hosts, epoch, cfg, "train", model_base)
        if is_code_task:
            execute(worker_hosts, epoch, cfg, "train")
        aggregate(epoch, cfg, "train")

        if is_process_reward:
            process_reward(worker_hosts, epoch, cfg)
        else:
            reward(epoch, cfg, "train", is_code_task)

        if have_value_model:
            train(worker_hosts, epoch, cfg, model_base, target = "value")
            train(worker_hosts, epoch, cfg, model_base, target = "policy")
        else:
            train(worker_hosts, epoch, cfg, model_base, target = None)

        if cfg.evaluation.if_eval:
            if epoch % cfg.experiment.eval_every == 0:
                if model_base in ["sdar", "trado"]:
                    remasking_strategy_list = cfg.evaluation.remasking_strategy
                    top_k_list = cfg.evaluation.top_k
                    block_size = cfg.evaluation.block_size
                    for j in range(len(remasking_strategy_list)):
                        remasking_strategy = remasking_strategy_list[j]
                        top_k = top_k_list[j]
                        sample(worker_hosts, epoch, cfg, "evaluation", model_base, block_size = block_size, top_k = top_k, remasking_strategy = remasking_strategy)
                        if is_code_task:
                            execute(worker_hosts, epoch, cfg, "evaluation")
                        aggregate(epoch, cfg, "evaluation")
                        reward(epoch, cfg, "evaluation", is_code_task, block_size = block_size, top_k = top_k, remasking_strategy = remasking_strategy)
                else:
                    block_size = cfg.evaluation.block_size
                    remasking_strategy_list = cfg.evaluation.remasking_strategy
                    if OmegaConf.select(cfg, "evaluation.top_k", default=MISSING) is not MISSING:
                        top_k = cfg.evaluation.top_k
                    else:
                        top_k = None
                    for j in range(len(remasking_strategy_list)):
                        remasking_strategy = remasking_strategy_list[j]
                        sample(worker_hosts, epoch, cfg, "evaluation", model_base, block_size = block_size, top_k = top_k, remasking_strategy = remasking_strategy)
                        if is_code_task:
                            execute(worker_hosts, epoch, cfg, "evaluation")
                        aggregate(epoch, cfg, "evaluation")
                        reward(epoch, cfg, "evaluation", is_code_task, block_size = block_size, top_k = top_k, remasking_strategy = remasking_strategy)

        epoch += 1
