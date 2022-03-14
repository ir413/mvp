#!/usr/bin/env python3

"""Utils."""

import hydra
import numpy as np
import os
import random

from omegaconf import DictConfig, OmegaConf
from typing import Dict

from isaacgym import gymapi
from isaacgym import gymutil

import torch

from pixmc.tasks.base.vec_task import VecTaskPython
from pixmc.tasks.franka_cabinet import FrankaCabinet
from pixmc.tasks.franka_move import FrankaMove
from pixmc.tasks.franka_pick import FrankaPick
from pixmc.tasks.franka_pick_object import FrankaPickObject
from pixmc.tasks.franka_reach import FrankaReach
from pixmc.tasks.kuka_cabinet import KukaCabinet
from pixmc.tasks.kuka_move import KukaMove
from pixmc.tasks.kuka_pick import KukaPick
from pixmc.tasks.kuka_pick_object import KukaPickObject
from pixmc.tasks.kuka_reach import KukaReach

from mvp.ppo import PPO
from mvp.ppo import ActorCritic
from mvp.ppo import PixelActorCritic


# Available tasks
_TASK_MAP = {
    "FrankaCabinet": FrankaCabinet,
    "FrankaMove": FrankaMove,
    "FrankaPick": FrankaPick,
    "FrankaPickObject": FrankaPickObject,
    "FrankaReach": FrankaReach,
    "KukaCabinet": KukaCabinet,
    "KukaMove": KukaMove,
    "KukaPick": KukaPick,
    "KukaPickObject": KukaPickObject,
    "KukaReach": KukaReach,
}


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionory."""
    if type(val) == dict:
        if not start:
            print('')
        nesting += 4
        for k in val:
            print(nesting * ' ', end='')
            print(k, end=': ')
            print_dict(val[k], nesting, start=False)
    else:
        print(val)


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def parse_sim_params(cfg, cfg_dict):

    # previously args defaults
    args_use_gpu_pipeline = (cfg.pipeline in ["gpu", "cuda"])
    args_use_gpu = ("cuda" in cfg.sim_device)
    args_subscenes = 0
    args_slices = args_subscenes
    args_num_threads = 0

    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1 / 60.
    sim_params.num_client_threads = args_slices

    assert cfg.physics_engine == "physx"
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = args_use_gpu
    sim_params.physx.num_subscenes = args_subscenes
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args_use_gpu_pipeline
    sim_params.physx.use_gpu = args_use_gpu

    # if sim options are provided in cfg parse them and update/override above:
    if "sim" in cfg_dict["task"]:
        print("Setting sim options")
        gymutil.parse_sim_config(cfg_dict["task"]["sim"], sim_params)

    # Override num_threads if specified
    if cfg.physics_engine == "physx" and args_num_threads > 0:
        sim_params.physx.num_threads = args_num_threads

    return sim_params


def parse_task(cfg, cfg_dict, sim_params):

    physics_engine = gymapi.SIM_PHYSX if cfg.physics_engine == "physx" else gymapi.SIM_GLEX
    device_type = "cuda" if "cuda" in cfg.sim_device else "cpu"
    device_id = int(cfg.sim_device.split(":")[1]) if "cuda" in cfg.sim_device else 0
    headless = cfg.headless
    rl_device = cfg.rl_device

    if cfg.num_gpus > 1:
        curr_device = torch.cuda.current_device()
        device_id = curr_device
        rl_device = curr_device

    task = _TASK_MAP[cfg.task.name](
        cfg=cfg_dict["task"],
        sim_params=sim_params,
        physics_engine=physics_engine,
        device_type=device_type,
        device_id=device_id,
        headless=headless
    )
    env = VecTaskPython(task, rl_device)

    return env


def process_ppo(env, cfg, cfg_dict, logdir, cptdir):

    learn_cfg = cfg_dict["train"]["learn"]
    is_testing = learn_cfg["test"]
    chkpt = learn_cfg["resume"]

    # Override resume and testing flags if they are passed as parameters.
    if not is_testing:
        is_testing = cfg.test
    if cfg.resume > 0:
        chkpt = cfg.resume

    is_pixel = (cfg.task.env.obs_type == "pixels")
    ac_cls = PixelActorCritic if is_pixel else ActorCritic

    ppo = PPO(
        vec_env=env,
        actor_critic_class=ac_cls,
        num_transitions_per_env=learn_cfg["nsteps"],
        num_learning_epochs=learn_cfg["noptepochs"],
        num_mini_batches=learn_cfg["nminibatches"],
        clip_param=learn_cfg["cliprange"],
        gamma=learn_cfg["gamma"],
        lam=learn_cfg["lam"],
        init_noise_std=learn_cfg.get("init_noise_std", 0.3),
        value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
        entropy_coef=learn_cfg["ent_coef"],
        learning_rate=learn_cfg["optim_stepsize"],
        max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
        use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
        schedule=learn_cfg.get("schedule", "fixed"),
        encoder_cfg=cfg_dict["train"].get("encoder", None),
        policy_cfg=cfg_dict["train"]["policy"],
        device=env.rl_device,
        sampler=learn_cfg.get("sampler", 'sequential'),
        log_dir=logdir,
        is_testing=is_testing,
        print_log=learn_cfg["print_log"],
        apply_reset=False,
        num_gpus=cfg.num_gpus
    )

    # TODO: improve checkpointing and avoid overwriting logs
    if is_testing:
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        ppo.test("{}/model_{}.pt".format(logdir, chkpt))
    elif chkpt > 0:
        print("Loading model from {}/model_{}.pt".format(cptdir, chkpt))
        ppo.load("{}/model_{}.pt".format(cptdir, chkpt))

    return ppo


def dump_cfg(cfg, logdir):
    out_f = os.path.join(logdir, "config.yaml")
    with open(out_f, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print("Wrote config to: {}".format(out_f))
