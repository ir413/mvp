#!/usr/bin/env python3

"""Actor."""

import numpy as np
import os
import torch
import torch.nn as nn

from mvp.backbones import vit


###############################################################################
# Pixel encoder
###############################################################################

_MODELS = {
    "vits-mae-hoi": "mae_pretrain_hoi_vit_small.pth",
    "vits-mae-in": "mae_pretrain_imagenet_vit_small.pth",
    "vits-sup-in": "sup_pretrain_imagenet_vit_small.pth",
    "vitb-mae-egosoup": "mae_pretrain_egosoup_vit_base.pth",
    "vitl-256-mae-egosoup": "mae_pretrain_egosoup_vit_large_256.pth",
}
_MODEL_FUNCS = {
    "vits": vit.vit_s16,
    "vitb": vit.vit_b16,
    "vitl": vit.vit_l16,
}


class Encoder(nn.Module):

    def __init__(self, model_name, pretrain_dir, freeze, emb_dim):
        super(Encoder, self).__init__()
        assert model_name in _MODELS, f"Unknown model name {model_name}"
        model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        img_size = 256 if "-256-" in model_name else 224
        pretrain_path = os.path.join(pretrain_dir, _MODELS[model_name])
        self.backbone, gap_dim = model_func(pretrain_path, img_size=img_size)
        if freeze:
            self.backbone.freeze()
        self.freeze = freeze
        self.projector = nn.Linear(gap_dim, emb_dim)

    def forward_feat_norm(self, x):
        feat = self.backbone.extract_feat(x)
        return self.backbone.forward_norm(feat)

    def forward_projector(self, x):
        return self.projector(x)

    def forward(self, x):
        feat = self.backbone.extract_feat(x)
        return self.projector(self.backbone.forward_norm(feat))


###############################################################################
# Pixel actor
###############################################################################

class PixelActor(nn.Module):

    def __init__(self, state_dim, action_dim, encoder_cfg, policy_cfg):
        super(PixelActor, self).__init__()

        # Encoder
        emb_dim = encoder_cfg.emb_dim
        self.image_enc = Encoder(
            model_name=encoder_cfg.name,
            pretrain_dir=encoder_cfg.pretrain_dir,
            freeze=encoder_cfg.freeze,
            emb_dim=emb_dim
        )
        self.state_enc = nn.Linear(state_dim, emb_dim)
        self.image_dropout = nn.Dropout(encoder_cfg.dropout)
        self.state_dropout = nn.Dropout(policy_cfg.dropout)

        # Policy
        actor_hidden_dim = policy_cfg.ws
        activation = nn.SELU()
        actor_layers = []
        actor_layers.append(nn.Linear(emb_dim * 2, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for li in range(len(actor_hidden_dim)):
            if li == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[li], action_dim))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dim[li], actor_hidden_dim[li + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        print(self.image_enc)
        print(self.state_enc)
        print(self.actor)

        # Initialize the actor weights
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        assert policy_cfg.init in ["orthogonal", "xavier_uniform"]
        self.init_weights(self.actor, actor_weights, policy_cfg.init)

    @staticmethod
    def init_weights(sequential, scales, init_method):
        if init_method == "orthogonal":
            [
                torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
                for idx, module in enumerate(
                    mod for mod in sequential if isinstance(mod, nn.Linear)
                )
            ]
        elif init_method == "xavier_uniform":
            for module in sequential:
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
        else:
            raise NotImplementedError

    def forward(self, images, states):
        image_emb = self.image_enc(images)
        image_emb = self.image_dropout(image_emb)
        state_emb = self.state_enc(states)
        state_emb = self.state_dropout(state_emb)
        joint_emb = torch.cat([image_emb, state_emb], dim=1)
        actions = self.actor(joint_emb)
        return actions
