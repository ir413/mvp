#!/usr/bin/env python3

"""Train a policy with BC."""

import hydra
import omegaconf
import os

from mvp.utils.hydra_utils import omegaconf_to_dict, print_dict, dump_cfg
from mvp.utils.hydra_utils import set_np_formatting, set_seed
from mvp.utils.hydra_utils import parse_sim_params, parse_task

import mvp.bc.bc as bc


@hydra.main(config_name="real", config_path="../configs/bc")
def train(cfg: omegaconf.DictConfig):

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
    set_seed(cfg.seed, cfg.torch_deterministic)

    # Construct env for sim
    if cfg.data.type == "sim":
        sim_params = parse_sim_params(cfg, cfg_dict)
        vec_env = parse_task(cfg, cfg_dict, sim_params)
    else:
        vec_env = None

    # Perform training
    bc.train(cfg, vec_env)


if __name__ == '__main__':
    train()
