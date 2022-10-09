#!/usr/bin/env python3

"""Train a policy with PPO."""

import hydra
import omegaconf
import os

from mvp.utils.hydra_utils import omegaconf_to_dict, print_dict, dump_cfg
from mvp.utils.hydra_utils import set_np_formatting, set_seed
from mvp.utils.hydra_utils import parse_sim_params, parse_task
from mvp.utils.hydra_utils import process_ppo


@hydra.main(config_name="config", config_path="../configs/ppo")
def train_ppo(cfg: omegaconf.DictConfig):

    # Assume no multi-gpu training
    assert cfg.num_gpus == 1

    # Parse the config
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # Create logdir and dump cfg
    if not cfg.test:
        os.makedirs(cfg.logdir, exist_ok=True)
        dump_cfg(cfg, cfg.logdir)

    # Set up python env
    set_np_formatting()
    set_seed(cfg.train.seed, cfg.train.torch_deterministic)

    # Construct task
    sim_params = parse_sim_params(cfg, cfg_dict)
    env = parse_task(cfg, cfg_dict, sim_params)

    # Perform training
    ppo = process_ppo(env, cfg, cfg_dict, cfg.logdir, cfg.cptdir)
    ppo.run(num_learning_iterations=cfg.train.learn.max_iterations, log_interval=cfg.train.learn.save_interval)


if __name__ == '__main__':
    train_ppo()
