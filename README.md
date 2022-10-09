## Masked Visual Pre-training for Robotics

<div align="center">
  <image src="assets/figs/teaser_real.png" width="720px" />
  <p></p>
</div>

### Overview

This repository contains the PyTorch implementation of the following two papers:

- [Masked Visual Pre-training for Motor Control](https://arxiv.org/abs/2203.06173)
- [Real-World Robot Learning with Masked Visual Pre-training](https://arxiv.org/abs/2210.03109)

It includes the pre-trained vision models and PPO/BC training code used in the papers.

### Pre-trained vision enocoders

We provide our pre-trained vision encoders. The models are in the same format as [mae](https://github.com/facebookresearch/mae) and [timm](https://github.com/rwightman/pytorch-image-models):

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">backbone</th>
<th valign="bottom">params</th>
<th valign="bottom">images</th>
<th valign="bottom">objective</th>
<th valign="bottom">md5</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW vits-mae-hoi -->
<tr>
<td align="center">ViT-S</td>
<td align="center">22M</td>
<td align="center">700K</td>
<td align="center">MAE</td>
<td align="center"><tt>fe6e30</tt></td>
<td align="center"><a href="https://berkeley.box.com/shared/static/m93ynem558jo8vltlads5rcmnahgsyzr.pth">model</a></td>
</tr>
<!-- vitb-mae-egosoup -->
<tr>
<td align="center">ViT-B</td>
<td align="center">86M</td>
<td align="center">4.5M</td>
<td align="center">MAE</td>
<td align="center"><tt>526093</tt></td>
<td align="center"><a href="https://berkeley.box.com/shared/static/0ckepd2ja3pi570z89ogd899cn387yut.pth">model</a></td>
</tr>
<!-- vitl-mae-egosoup -->
<tr>
<td align="center">ViT-L</td>
<td align="center">307M</td>
<td align="center">4.5M</td>
<td align="center">MAE</td>
<td align="center"><tt>5352b0</tt></td>
<td align="center"><a href="https://berkeley.box.com/shared/static/6p0pc47mlpp4hhwlin2hf035lxlgddxr.pth">model</a></td>
</tr>
<!-- END TABLE -->
</tbody></table>

You can use our pre-trained models directly in your code (e.g., to extract image features) or use them with our training code. We provde instructions for both use-cases next.

### Using pre-trained models in your code

Install [PyTorch](https://pytorch.org/get-started/locally/) and mvp package:

```
pip install git+https://github.com/ir413/mvp
```

Import pre-trained models:

```python
import mvp

model = mvp.load("vitb-mae-egosoup")
model.freeze()
```

### Benchmark suite and training code

Please see [`TASKS.md`](TASKS.md) for task descriptions and [`GETTING_STARTED.md`](GETTING_STARTED.md) for installation and training instructions.

### Citation

If you find the code or pre-trained models useful in your research, please consider citing an appropriate subset of the following papers:

```
@article{Xiao2022
  title = {Masked Visual Pre-training for Motor Control},
  author = {Tete Xiao and Ilija Radosavovic and Trevor Darrell and Jitendra Malik},
  journal = {arXiv:2203.06173},
  year = {2022}
}

@article{Radosavovic2022,
  title = {Real-World Robot Learning with Masked Visual Pre-training},
  author = {Ilija Radosavovic and Tete Xiao and Stephen James and Pieter Abbeel and Jitendra Malik and Trevor Darrell},
  year = {2022},
  journal = {CoRL}
}
```

### Acknowledgments

We thank NVIDIA IsaacGym and PhysX teams for making the simulator and preview code examples available.
