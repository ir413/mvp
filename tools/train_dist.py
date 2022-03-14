#!/usr/bin/env python3

"""Train a policy with distributed PPO."""

import hydra
import omegaconf
import os
import random

from mvp.utils.hydra_utils import omegaconf_to_dict, print_dict, dump_cfg
from mvp.utils.hydra_utils import set_np_formatting, set_seed
from mvp.utils.hydra_utils import parse_sim_params, parse_task
from mvp.utils.hydra_utils import process_ppo

import torch


def single_proc_train(local_rank, port, world_size, cfg):

    # Init process group
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:{}".format(port),
        world_size=world_size,
        rank=local_rank
    )
    torch.cuda.set_device(local_rank)

    # Parse the config
    cfg_dict = omegaconf_to_dict(cfg)

    # Create logdir and dump cfg
    if local_rank == 0:
        print_dict(cfg_dict)
        os.makedirs(cfg.logdir, exist_ok=True)
        dump_cfg(cfg, cfg.logdir)

    # Set a different seed in each proc
    seed = cfg.train.seed * world_size + local_rank
    set_np_formatting()
    set_seed(seed, cfg.train.torch_deterministic)

    # Construct task
    sim_params = parse_sim_params(cfg, cfg_dict)
    env = parse_task(cfg, cfg_dict, sim_params)

    # Perform training
    ppo = process_ppo(env, cfg, cfg_dict, cfg.logdir, cfg.cptdir)
    ppo.run(num_learning_iterations=cfg.train.learn.max_iterations, log_interval=cfg.train.learn.save_interval)

    # Clean up
    torch.distributed.destroy_process_group()


@hydra.main(config_name="config", config_path="../configs")
def train(cfg: omegaconf.DictConfig):

    # Assume multi-gpu training
    assert cfg.num_gpus > 1

    # Select a port for proc group init randomly
    port_range = [10000, 65000]
    port = random.randint(port_range[0], port_range[1])

    # Start a process per GPU
    torch.multiprocessing.start_processes(
        single_proc_train,
        args=(port, cfg.num_gpus, cfg),
        nprocs=cfg.num_gpus,
        start_method="spawn"
    )


if __name__ == '__main__':
    train()
