#!/usr/bin/env python3

"""
Vision Transformer (ViT) implementation.
"""

import os
import timm.models.vision_transformer

from functools import partial
from iopath.common.file_io import PathManagerFactory

import torch
import torch.nn as nn

pathmgr = PathManagerFactory.get()


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer
        referene:
            - MAE:  https://github.com/facebookresearch/mae/blob/main/models_vit.py
            - timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        # remove the classifier
        del self.pre_logits, self.head

    def extract_feat(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = x[:, 0].detach().float()
        return x

    def forward_norm(self, x):
        return self.norm(x)

    def forward(self, x):
        return self.forward_norm(self.extract_feat(x))

    def freeze(self):
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

        def _freeze_module(m):
            for p in m.parameters():
                p.requires_grad = False

        _freeze_module(self.patch_embed)
        _freeze_module(self.blocks)

        trainable_params = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_params.append(name)

        #print("Trainable parameters in the encoder:")
        #print(trainable_params)


def vit_s16(pretrained, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    assert os.path.exists(pretrained) or pretrained in ["none"]
    # load from checkpoint
    if pretrained != "none":
        load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    hidden_dim = 384
    return model, hidden_dim


def unwrap_model(model):
    """Remove the DistributedDataParallel wrapper if present."""
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


def load_checkpoint(checkpoint_file, model):
    """Loads a checkpoint selectively based on the input options."""
    assert pathmgr.exists(checkpoint_file), "Checkpoint '{}' not found".format(
        checkpoint_file
    )
    with pathmgr.open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    state_dict = checkpoint["model"]

    r = unwrap_model(model).load_state_dict(state_dict, strict=False)
    if r.unexpected_keys or r.missing_keys:
        print(f"Loading weights, unexpected keys: {r.unexpected_keys}")
        print(f"Loading weights, missing keys: {r.missing_keys}")
