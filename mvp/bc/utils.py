#!/usr/bin/env python3

"""Utils."""

import math
import numpy as np
import os
import torch


class AverageMeter(object):

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def adjust_lr(optimizer, base_lr, cur_epoch, warmup_epoch, num_epoch):
    if cur_epoch < warmup_epoch:
        lr = base_lr * cur_epoch / warmup_epoch
    else:
        lr = base_lr * 0.5 * (1. + math.cos(math.pi * (cur_epoch - warmup_epoch) / (num_epoch - warmup_epoch)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(out_dir, model, cur_epoch):
    fname = "model_ep{:04d}.pt".format(cur_epoch)
    out_f = os.path.join(out_dir, fname)
    torch.save({
        "epoch": cur_epoch,
        "model_state": model.state_dict()
    }, out_f)
    print("Saved checkpoint to: {}".format(out_f))


def save_best_checkpoint(out_dir, model, cur_epoch, best_loss):
    fname = "model_best.pt"
    out_f = os.path.join(out_dir, fname)
    torch.save({
        "epoch": cur_epoch,
        "model_state": model.state_dict(),
        "loss": best_loss
    }, out_f)
    print("Saved best checkpoint to: {}".format(out_f))


def load_checkpoint(file_path, model):
    assert os.path.exists(file_path)
    checkpoint = torch.load(file_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    print("Loaded checkpoint from: {}".format(file_path))
