import os
import sys
import subprocess
import shlex
from termcolor import cprint

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


    def env_prefix() -> str:
        return (
            "source ~/.bashrc && "
            f"source activate {env_name} && "
        )

    def sample(worker_hosts, cfg, strategy):
        project = cfg.experiment.project
        model_base = cfg.model_base
        procs = []
        if model_base == "dream":
            python_name = "dream_sample"
        elif model_base == "llada" or model_base == "mmada":
            python_name = "mmada_v_sample"
        elif model_base == "sdar":
            python_name = "sdar_sample"
        elif model_base == "trado":
            python_name = "trado_sample"
        elif model_base == "lladav":
            python_name = "lladav_sample"
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR}/sample && "
                f"python {python_name}.py "
                f"config=../configs/{project}.yaml "
                f"experiment.node_index={idx} "
                f"model={cfg.model} "
                f"rollout.remasking_strategy={strategy}"
            )
            full_cmd = env_prefix() + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()


    def execute(worker_hosts, cfg, strategy):
        project = cfg.experiment.project
        procs = []
        for idx, host in enumerate(worker_hosts):
            full_cmd = env_prefix() + (
                f"cd {BASE_DIR}/reward && "
                f"python execute.py "
                f"config=../configs/{project}.yaml "
                f"experiment.node_index={idx} "
                f"model={cfg.model} "
                f"rollout.remasking_strategy={strategy}"
            )
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()

    def aggregate(cfg, strategy):
        project = cfg.experiment.project
        full_cmd = env_prefix() + (
            f"cd {BASE_DIR}/reward && "
            f"python aggregate_data_v.py "
            f"config=../configs/{project}.yaml "
            f"model={cfg.model} "
            f"rollout.remasking_strategy={strategy}"
        )
        run_local(full_cmd)


    def reward(cfg, strategy):
        project = cfg.experiment.project
        full_cmd = env_prefix() + (
            f"cd {BASE_DIR}/reward && "
            f"python reward_v.py "
            f"config=../configs/{project}.yaml "
            f"model={cfg.model} "
            f"rollout.remasking_strategy={strategy}"
        )
        run_local(full_cmd)


    # use first node to control others

    cfg = get_config()
    INIT_BASH = make_init_bash(cfg)
    BASE_DIR = cfg.system.base_dir
    env_name = cfg.system.env_name
    project = cfg.experiment.project
    num_node = cfg.experiment.num_node
    worker_hosts = [os.environ[f"MLP_WORKER_{i}_HOST"] for i in range(num_node)]
    eval_type = cfg.dataset.data_type

    import time
    time.sleep(30)

    init_hosts(worker_hosts)

    import time
    time.sleep(10)
    
    os.makedirs(f"{project}/results", exist_ok=True)
    
    remasking_strategies = cfg.rollout.remasking_strategy
    if not isinstance(remasking_strategies, (list, ListConfig)):
        remasking_strategies = [remasking_strategies]

    for strategy in remasking_strategies:
        cprint(f"--- Running evaluation for strategy: {strategy} ---", "green")
        sample(worker_hosts, cfg, strategy)
        
        if eval_type == "code":
            execute(worker_hosts, cfg, strategy)
        
        aggregate(cfg, strategy)
        
        reward(cfg, strategy)









