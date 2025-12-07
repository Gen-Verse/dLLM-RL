import os
import sys
import subprocess
from termcolor import cprint

from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":
    config = get_config()

    project_name = config.experiment.project
    eval_type = config.dataset.data_type

    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")
        
    def sample(model_base, strategy):
        cprint(f"This is sampling.", color = "green")
        if model_base == "dream":
            subprocess.run(
                f'python dream_sample.py '
                f'config=../configs/{project_name}.yaml '
                f'rollout.remasking_strategy={strategy}',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "llada" or model_base == "mmada":
            subprocess.run(
                f'python mmada_v_sample.py '
                f'config=../configs/{project_name}.yaml '
                f'rollout.remasking_strategy={strategy}',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "sdar":
            subprocess.run(
                f'python sdar_sample.py '
                f'config=../configs/{project_name}.yaml '
                f'rollout.remasking_strategy={strategy}',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "trado":
            subprocess.run(
                f'python trado_sample.py '
                f'config=../configs/{project_name}.yaml '
                f'rollout.remasking_strategy={strategy}',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "lladav":
            subprocess.run(
                f'python lladav_sample.py '
                f'config=../configs/{project_name}.yaml '
                f'rollout.remasking_strategy={strategy}',
                shell=True,
                cwd='sample',
                check=True,
            )
    
    def reward(strategy):
        cprint(f"This is the rewarding.", color = "green")
        subprocess.run(
            f'python reward_v.py '
            f'config=../configs/{project_name}.yaml '
            f'rollout.remasking_strategy={strategy}',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    def execute(strategy):
        cprint(f"This is the execution.", color = "green")
        subprocess.run(
            f'python execute.py '
            f'config=../configs/{project_name}.yaml '
            f'rollout.remasking_strategy={strategy}',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    
    
    os.makedirs(f"{project_name}/results", exist_ok=True)
    
    remasking_strategies = config.rollout.remasking_strategy
    if not isinstance(remasking_strategies, (list, ListConfig)):
        remasking_strategies = [remasking_strategies]
    
    for strategy in remasking_strategies:
        cprint(f"--- Running evaluation for strategy: {strategy} ---", "green")
        sample(config.model_base, strategy)
        if eval_type == "code":
            execute(strategy)
        reward(strategy)



