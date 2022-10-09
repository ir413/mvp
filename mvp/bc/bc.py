#!/usr/bin/env python3

"""Behavior cloning (BC)."""

import copy
import json
import math
import os
import random
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from mvp.bc.actor import PixelActor
from mvp.bc.dataset import RealDemoDataset
from mvp.bc.dataset import SimDemoDataset
from mvp.bc.utils import AverageMeter
from mvp.bc.utils import adjust_lr
from mvp.bc.utils import save_checkpoint, save_best_checkpoint


def get_optimizer_groups(model, default_wd):
    param_group_names, param_group_vars = dict(), dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # ks = [k for (k, x) in enumerate(["bn", "ln", "norm", "bias", ""]) if x in n]
        if "token" in n:
            name_apx = "t"
            wd_val = 0.0
        elif 'pos_embed' in n:
            name_apx = "p"
            wd_val = 0.0
        elif "bn" in n or "ln" in n or "norm" in n:
            name_apx = "n"
            wd_val = 0.0
        elif "bias" in n:
            name_apx = "b"
            wd_val = 0.0
        else:
            name_apx = 'w'
            wd_val = default_wd

        param_group = f"wd:{name_apx}"
        if param_group not in param_group_names:
            item = {"params": [], "weight_decay": wd_val}
            param_group_names[param_group] = copy.deepcopy(item)
            param_group_vars[param_group] = copy.deepcopy(item)
        param_group_names[param_group]["params"].append(n)
        param_group_vars[param_group]["params"].append(p)

    param_list = list(param_group_vars.values())

    param_group_str = colored(
        json.dumps(param_group_names, sort_keys=True, indent=2), "blue"
    )
    print("Parameter groups:\n" + param_group_str)

    return param_list


@torch.no_grad()
def test_env(cfg, vec_env, model, writer, cur_epoch):
    model.eval()
    vec_env.task.reset(torch.arange(vec_env.num_envs, device=vec_env.rl_device))
    current_obs = vec_env.reset()
    current_states = vec_env.get_state()

    maxlen = 200
    cur_reward_sum = torch.zeros(vec_env.num_envs, dtype=torch.float, device=vec_env.rl_device)
    cur_episode_length = torch.zeros(vec_env.num_envs, dtype=torch.float, device=vec_env.rl_device)

    reward_sum = []
    episode_length = []
    successes = []

    while len(reward_sum) <= maxlen:
        images = current_obs
        states = current_states

        # Scale the current_states
        if hasattr(model, "dof_lower_limits"):
            scaled_states = 2.0 * (states - model.dof_lower_limits) / (model.dof_upper_limits - model.dof_lower_limits) - 1.0
        else:
            scaled_states = states

        # Compute actions
        actions = model(images, scaled_states)

        # Unscale the action and convert to absolute pose
        if hasattr(model, "dof_lower_limits"):
            actions = actions * (model.dof_upper_limits - model.dof_lower_limits) / 2.0
            actions = actions + states

        # Step the vec_environment
        next_obs, rews, dones, infos = vec_env.step(actions)
        next_states = vec_env.get_state()
        current_obs.copy_(next_obs)
        current_states.copy_(next_states)

        cur_reward_sum[:] += rews
        cur_episode_length[:] += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)
        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        successes.extend(infos["successes"][new_ids][:, 0].cpu().numpy().tolist())
        cur_reward_sum[new_ids] = 0
        cur_episode_length[new_ids] = 0

    mean_success = statistics.mean(successes) * 100
    print("[env] epoch={}, success={:.3f}".format(cur_epoch, mean_success))
    writer.add_scalar("env/success", mean_success, cur_epoch)


@torch.no_grad()
def test_epoch(cfg, test_loader, model, writer, cur_epoch):
    model.eval()
    losses = AverageMeter("test_loss")
    for cur_iter, (obs, states, actions) in enumerate(test_loader):
        obs, states = obs.cuda(), states.cuda()
        actions = actions.cuda(non_blocking=True)
        preds = model(obs, states)
        loss = F.mse_loss(preds, actions)
        losses.update(loss.item(), obs.shape[0])
    print("[test] epoch={}, loss={:.6f}".format(cur_epoch, losses.avg))
    writer.add_scalar("test/ep_loss", losses.avg, cur_epoch)
    return losses.avg


def train_epoch(cfg, train_loader, model, optimizer, writer, cur_epoch):
    model.train()
    epoch_iters = len(train_loader)
    losses = AverageMeter("train_loss")
    for cur_iter, (obs, states, actions) in enumerate(train_loader):
        lr = adjust_lr(
            optimizer, cfg.train.lr, cur_epoch + float(cur_iter) / len(train_loader), cfg.train.warmup_ep, cfg.train.num_ep
        )
        obs, states = obs.cuda(), states.cuda()
        actions = actions.cuda(non_blocking=True)
        preds = model(obs, states)
        loss = F.mse_loss(preds, actions)
        optimizer.zero_grad()
        loss.backward()
        if cfg.train.clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_norm)
        optimizer.step()
        losses.update(loss.item(), obs.shape[0])
        if cur_iter % 10 == 0:
            cur_iter_abs = cur_epoch * epoch_iters + cur_iter
            print("[train] epoch={}, iter={}, loss={:.6f}".format(cur_epoch, cur_iter, losses.val))
            writer.add_scalar("train/iter_loss", losses.val, cur_iter_abs)
            writer.add_scalar("train/lr", lr, cur_iter_abs)
    print("[train] epoch={}, loss={:.6f}".format(cur_epoch, losses.avg))
    writer.add_scalar("train/ep_loss", losses.avg, cur_epoch)


def train(cfg, vec_env):
    """Trains a model with BC."""

    # Log stats to tensorboard
    writer = SummaryWriter(log_dir=cfg.logdir, flush_secs=10)

    # Retrieve real/sim dataset class
    dataset_cls = RealDemoDataset if cfg.data.type == "real" else SimDemoDataset

    kargs = {}
    if cfg.data.type == "real":
        kargs["valid_state_dim"] = cfg.data.valid_state_dim
        kargs["joint_scaling"] = cfg.actor.joint_scaling
        kargs["dof_lower_limits"] = cfg.actor.dof_lower_limits
        kargs["dof_upper_limits"] = cfg.actor.dof_upper_limits
        kargs["im_size"] = cfg.data.im_size
        kargs["lower_crop"] = cfg.data.lower_crop
        kargs["cam"] = cfg.data.cam
        kargs["frame_skip"] = cfg.data.frame_skip
        kargs["gripper_scaling"] = cfg.data.gripper_scaling
        # augmentations
        kargs["color_jitter_prob"] = cfg.data.color_jitter_prob
        kargs["color_jitter_param"] = tuple(cfg.data.color_jitter_param)
        kargs["gray_scale_prob"] = cfg.data.gray_scale_prob
        kargs["gaussian_blur_prob"] = cfg.data.gaussian_blur_prob
        kargs["gaussian_blur_param"] = tuple(cfg.data.gaussian_blur_param)
        kargs["joint_noise_std"] = cfg.data.joint_noise_std

    # Construct train dataset/loader
    random.seed(cfg.seed)
    train_dataset = dataset_cls(
        cfg.data.demo_path, end_ind=cfg.data.num_train,
        max_shift=cfg.data.max_shift, **kargs
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train.mb_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True
    )

    # Construct test dataset/loader
    if cfg.data.num_test > 0:
        if cfg.data.type == "real":
            kargs["color_jitter_prob"] = 0.0
            kargs["gray_scale_prob"] = 0.0
            kargs["gaussian_blur_prob"] = 0.0
            kargs["joint_noise_std"] = 0.0
        random.seed(cfg.seed)
        test_dataset = dataset_cls(
            cfg.data.demo_path,
            start_ind=cfg.data.num_train, end_ind=cfg.data.num_train+cfg.data.num_test,
            max_shift=0, **kargs
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg.train.mb_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False
        )

    # Construct the model
    model = PixelActor(
        state_dim=cfg.actor.state_dim,
        action_dim=cfg.actor.action_dim,
        encoder_cfg=cfg.encoder,
        policy_cfg=cfg.policy
    )
    if cfg.actor.joint_scaling:
        dof_lower_limits = torch.Tensor(cfg.actor.dof_lower_limits).float()
        dof_upper_limits = torch.Tensor(cfg.actor.dof_upper_limits).float()
        model.register_buffer("dof_lower_limits", dof_lower_limits)
        model.register_buffer("dof_upper_limits", dof_upper_limits)
    model = model.cuda()

    # Construct the optimizer
    optimizer = torch.optim.AdamW(
        get_optimizer_groups(model, default_wd=cfg.train.wd),
        lr=cfg.train.lr,
        weight_decay=cfg.train.wd
    )

    # Perform training
    best_loss, loss = None, None
    for cur_epoch in range(cfg.train.num_ep):
        train_epoch(cfg, train_loader, model, optimizer, writer, cur_epoch)
        if (cur_epoch + 1) % cfg.train.test_freq == 0 or (cur_epoch + 1) == cfg.train.num_ep:
            if cfg.data.num_test > 0:
                loss = test_epoch(cfg, test_loader, model, writer, cur_epoch)
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    save_best_checkpoint(cfg.logdir, model, cur_epoch + 1, best_loss)
            if vec_env:
                test_env(cfg, vec_env, model, writer, cur_epoch)
        if (cur_epoch  + 1) % 100 == 0:
            save_checkpoint(cfg.logdir, model, cur_epoch + 1)

    writer.close()
    print("Wrote results to: {}".format(cfg.logdir))
