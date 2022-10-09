#!/usr/bin/env python3

"""Demo dataset."""

import cv2
import math
import numpy as np
import os
import pickle
import random

from tqdm import tqdm
from PIL import Image, ImageFilter

import torch
import torchvision.transforms as transforms


# Per-channel mean and standard deviation (in RGB order)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def lower_center_crop(im, crop_size):
    """Performs lower-center cropping."""
    w = im.shape[1]
    x = math.ceil((w - crop_size) / 2)
    return im[-crop_size:, x:(x + crop_size), :]


def lower_random_shift_crop(im, crop_size, max_shift):
    w = im.shape[1]
    x = (w - crop_size) // 2
    assert x + max_shift + crop_size <= w
    assert x - max_shift >= 0
    shift = np.random.randint(-max_shift, max_shift + 1)
    x = x + shift
    return im[-crop_size:, x:(x + crop_size), :]


def resize_crop(im, size, max_shift=0, lower_crop=True):
    """Performs image resize and crop."""
    if max_shift > 0:
        # (480, 640, 3) -> (448, 448, 3) or (480, 480, 3)
        crop_size = 448 if lower_crop else 480
        im = lower_random_shift_crop(im, crop_size, max_shift)
        # (448, 448, 3) or (480, 480, 3) -> (size, size, 3)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        # (480, 640, 3) -> (448, 448, 3) or (480, 480, 3)
        crop_size = 448 if lower_crop else 480
        im = lower_center_crop(im, crop_size)
        # (448, 448, 3) or (480, 480, 3) -> (size, size, 3)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    return im


def color_norm(im, mean, std):
    """Performs per-channel normalization."""
    for i in range(3):
        im[:, :, i] = (im[:, :, i] - mean[i]) / std[i]
    return im


def color_unnorm(im, mean, std):
    """Performs per-channel unnormalization."""
    for i in range(3):
        im[:, :, i] = im[:, :, i] * std[i] + mean[i]
    return im


def normalize(im):
    """Performs image normalization."""
    # [0, 255] -> [0, 1]
    im = im.astype(np.float32) / 255.0
    # Color norm
    im = color_norm(im, _MEAN, _STD)
    # HWC -> CHW
    im = im.transpose([2, 0, 1])
    return im


def unnormalize(im):
    """Performs image unnormalization."""
    # CHW -> HWC
    im = im.transpose([1, 2, 0])
    # Color unnorm
    im = color_unnorm(im, _MEAN, _STD)
    # [0, 1] -> [0, 255]
    im = (im.astype(np.float32) * 255.0).astype(np.uint8)
    return im


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(.1, 2.)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RealDemoDataset(torch.utils.data.Dataset):
    """Real demo dataset."""

    def __init__(
            self, demo_root, start_ind=0, end_ind=1000000,
            valid_state_dim=13, joint_scaling=False, gripper_scaling=0.2,
            dof_lower_limits=None, dof_upper_limits=None,
            im_size=224, max_shift=0, lower_crop=True, cam="ego", frame_skip=0,
            color_jitter_prob=0.0, color_jitter_param=(0.2, 0.2, 0.2, 0.1),
            gray_scale_prob=0.0, gaussian_blur_prob=0.0, gaussian_blur_param=(.1, 2.),
            joint_noise_std=0.0,
        ):
        assert cam in ["ego", "exo"]
        self._cam = cam
        self._demo_root = demo_root
        self._l_ind = start_ind
        self._r_ind = end_ind
        self._valid_state_dim = valid_state_dim
        self._joint_scaling = joint_scaling
        self._gripper_scaling = gripper_scaling
        self._dof_lower_limits = np.array(dof_lower_limits, dtype=np.float32)
        self._dof_upper_limits = np.array(dof_upper_limits, dtype=np.float32)
        self._im_size = im_size
        self._max_shift = max_shift
        self._lower_crop = lower_crop
        self._frame_skip = frame_skip
        _augmentation = [
            transforms.RandomApply([
                transforms.ColorJitter(*color_jitter_param)  # not strengthened
            ], p=color_jitter_prob),
            transforms.RandomGrayscale(p=gray_scale_prob),
            transforms.RandomApply([GaussianBlur(gaussian_blur_param)], p=gaussian_blur_prob),
        ]
        self._augmentation = transforms.Compose(_augmentation)
        self._joint_noise_std = joint_noise_std
        self._dataset = self._construct()
        print("Demo range: [{}, {})".format(start_ind, end_ind))
        print("Num tuples: {:,}".format(len(self._dataset)))

    def _scale_joint(self, joints):
        return 2.0 * (joints - self._dof_lower_limits) / (self._dof_upper_limits - self._dof_lower_limits) - 1.0

    def _compute_joint_state(self, obs_t):
        _jpos, gripper_open = obs_t["joint_pos"], np.array([float(obs_t["gripper_open"])])
        jpos = np.concatenate([_jpos, gripper_open])
        return self._scale_joint(jpos) if self._joint_scaling else jpos

    def _compute_joint_action(self, obs_t, obs_t1):
        _jpos, gripper_open = obs_t["joint_pos"], np.array([float(obs_t["gripper_open"])])
        gripper_open[0] *= self._gripper_scaling
        jpos_t = np.concatenate([_jpos, gripper_open])
        _jpos, gripper_open = obs_t1["joint_pos"], np.array([float(obs_t1["gripper_open"])])
        gripper_open[0] *= self._gripper_scaling
        jpos_t1 = np.concatenate([_jpos, gripper_open])
        if self._joint_scaling:
            return self._scale_joint(jpos_t1) - self._scale_joint(jpos_t)
        else:
            return jpos_t1 - jpos_t

    def _filter_demo(self, demo, pos_filter_thres=0.004):
        valid_steps = [demo[0]]
        for cur_step in demo[1:]:
            prev_step = valid_steps[-1]
            if (
                np.max(np.abs(cur_step["joint_pos"] - prev_step["joint_pos"])) > pos_filter_thres or
                cur_step["gripper_open"] != prev_step["gripper_open"]
            ):
                valid_steps.append(cur_step)
        return valid_steps

    def _construct(self):
        print("Loading demos from: {}".format(self._demo_root))
        dataset = []
        demo_lens, unfiltered_demo_lens = [], []
        demo_dirs = sorted(os.listdir(self._demo_root))
        random.shuffle(demo_dirs)
        demo_dirs = demo_dirs[self._l_ind:self._r_ind]
        for i, demo_dir in enumerate(tqdm(demo_dirs)):
            # Extract demo obs
            demo_obs = []
            demo_path = os.path.join(self._demo_root, demo_dir)
            for j, obs_file in enumerate(sorted(os.listdir(demo_path))):
                obs_path = os.path.join(demo_path, obs_file)
                with open(obs_path, "rb") as f:
                    obs = pickle.load(f)
                element = {
                    "joint_pos": obs["joint_positions"][:self._valid_state_dim],
                    "joint_vel": obs["joint_velocites"][:self._valid_state_dim],
                    "gripper_open": obs["gripper_open"] if "gripper_open" in obs else 0,
                }
                if "rgb" in obs:  # old demo format, only ego
                    element["rgb_im"] = obs["rgb"]
                elif self._cam == "ego":
                    element["rgb_im"] = obs["rgb_ego"]
                else:
                    element["rgb_im"] = obs["rgb_exo"]
                demo_obs.append(element)
            unfiltered_demo_lens.append(len(demo_obs))
            # Filter demo
            demo_obs = self._filter_demo(demo_obs)
            demo_lens.append(len(demo_obs))
            # Compute actions
            for k in range(0, len(demo_obs) - self._frame_skip - 1):
                t1 = k + self._frame_skip + 1
                # Special case for gripper status change in skip frame
                for j in range(k + 1, t1):
                    if demo_obs[j]["gripper_open"] != demo_obs[k]["gripper_open"]:
                        t1 = j
                        break
                obs_t, obs_t1 = demo_obs[k], demo_obs[t1]
                # for obs_t, obs_t1 in zip(demo_obs[:-1], demo_obs[1:]):
                state_t = self._compute_joint_state(obs_t)
                act_t = self._compute_joint_action(obs_t, obs_t1)
                element = {
                    "demo_ind": i,
                    "demo_dir": demo_dir,
                    "step_ind": k,
                    "state": state_t,
                    "action": act_t
                }
                element["rgb_im"] = obs_t["rgb_im"]
                dataset.append(element)
        print("Num demos: {:,}".format(len(demo_lens)))
        print("Mean original demo len: {:.3f}".format(np.mean(unfiltered_demo_lens)))
        print("Mean filtered demo len: {:.3f}".format(np.mean(demo_lens)))
        return dataset

    def process_image(self, im):
        im_pil = Image.fromarray(im).convert("RGB")
        im_pil = self._augmentation(im_pil)
        im = np.array(im_pil).astype(np.float32)
        im = resize_crop(im, self._im_size, max_shift=self._max_shift, lower_crop=self._lower_crop)
        im = normalize(im)
        return im

    def __getitem__(self, ind):
        entry = self._dataset[ind]
        im = entry["rgb_im"].copy()
        im = self.process_image(im)
        state = entry["state"].astype(np.float32)
        if self._joint_noise_std > 0.0:
            state = state + np.random.normal(0.0, self._joint_noise_std, size=len(state)).astype(np.float32)
        action = entry["action"].astype(np.float32)
        return im, state, action

    def __len__(self):
        return len(self._dataset)


class SimDemoDataset(torch.utils.data.Dataset):
    """Sim dataset."""

    def __init__(
        self, demo_root, start_ind=0, end_ind=1000000,
        im_size=224, max_shift=0
    ):
        self._demo_root = demo_root
        self._l_ind = start_ind
        self._r_ind = end_ind
        self._dataset = self._construct()
        self._im_size = im_size
        self._max_shift = max_shift
        print("Demo range: [{}, {})".format(start_ind, end_ind))
        print("Num tuples: {:,}".format(len(self._dataset)))

    def _construct(self):
        print("Loading demos from: {}".format(self._demo_root))
        dataset = []
        demo_lens = []
        demo_dirs = [d for d in os.listdir(self._demo_root) if "demo" in d]
        demo_dirs = [d for d in demo_dirs if os.path.exists(os.path.join(self._demo_root, d, "SUCCESS.txt"))]
        demo_dirs = sorted(demo_dirs)[self._l_ind:self._r_ind]
        for i, demo_dir in enumerate(tqdm(demo_dirs)):
            # Extract demo obs
            demo_len = 0
            demo_path = os.path.join(self._demo_root, demo_dir)
            for j, obs_file in enumerate(sorted(os.listdir(demo_path))):
                if ".txt" in obs_file:
                    continue
                obs_path = os.path.join(demo_path, obs_file)
                with open(obs_path, "rb") as f:
                    obs = pickle.load(f)
                dataset.append({
                    "demo_ind": i,
                    "demo_dir": demo_dir,
                    "image": obs["obs"],
                    "state": obs["state"],
                    "action": obs["action"]
                })
                demo_len += 1
            demo_lens.append(demo_len)
        print("Num demos: {:,}".format(len(demo_lens)))
        print("Mean demo len: {:.3f}".format(np.mean(demo_lens)))
        return dataset

    def __getitem__(self, ind):
        entry = self._dataset[ind]
        im = entry["image"].copy()
        im = np.array(im).astype(np.float32)
        state = entry["state"].astype(np.float32)
        action = entry["action"].astype(np.float32)
        return im, state, action

    def __len__(self):
        return len(self._dataset)
