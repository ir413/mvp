## Masked Visual Pre-training for Motor Control

<div align="center">
  <image src="assets/figs/teaser.png" width="360px" />
  <p></p>
</div>
  
This is a PyTorch implementation of the paper [Masked Visual Pre-training for Motor Control](https://arxiv.org/abs/2203.06173). It contains the benchmark suite, pre-trained models, and the training code to reproduce the results from the paper.

### Pre-trained visual enocoders

We provide pre-trained visual encoders used in the paper. The models are in the same format as [mae](https://github.com/facebookresearch/mae) and [timm](https://github.com/rwightman/pytorch-image-models):

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">backbone</th>
<th valign="bottom">objective</th>
<th valign="bottom">data</th>
<th valign="bottom">md5</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW MAE-HOI -->
<tr>
<td align="center">ViT-S</td>
<td align="left">MAE</td>
<td align="left">in-the-wild</td>
<td align="center"><tt>fe6e30</tt></td>
<td align="center"><a href="https://www.dropbox.com/s/51fasmar8hjfpeh/mae_pretrain_hoi_vit_small.pth?dl=1">model</a></td>
</tr>
<!-- ROW MAE-IN -->
<tr>
<td align="center">ViT-S</td>
<td align="left">MAE</td>
<td align="left">ImageNet</td>
<td align="center"><tt>29a004</tt></td>
<td align="center"><a href="https://www.dropbox.com/s/3whtrak5wsfzoaw/mae_pretrain_imagenet_vit_small.pth?dl=1">model</a></td>
</tr>
<!-- ROW Supervised-IN -->
<tr>
<td align="center">ViT-S</td>
<td align="left">Supervised</td>
<td align="left">ImageNet</td>
<td align="center"><tt>f8f23b</tt></td>
<td align="center"><a href="https://www.dropbox.com/s/dw3uf5aff6yzmx3/sup_pretrain_imagenet_vit_small.pth?dl=1">model</a></td>
</tr>
<!-- END TABLE -->
</tbody></table>

You can use our pre-trained encoders directly in your code (e.g., to extract image features) or use them with our benchmark suite and RL training code. We provde instructions for both use-cases next.

### Using pre-trained encoders in your code

Install the python package:

```
pip install git+https://github.com/ir413/mvp
```

Import pre-trained encoders:

```python
import mvp

model = mvp.load("vits-mae-hoi")
model.freeze()
```

### Benchmark suite and RL training code

Please see [`INSTALL.md`](INSTALL.md) for installation instructions.

### Example training commands

Train `FrankaPick` from states:

```
python tools/train.py task=FrankaPick
```

Train `FrankaPick` from pixels:

```
python tools/train.py task=FrankaPickPixels
```

Train on 8 GPUs:

```
python tools/train_dist.py num_gpus=8
```

Test a policy after N iterations:

```
python tools/train.py test=True headless=False logdir=/path/to/job resume=N
```

### Citation

If you find the code or pre-trained models useful in your research, please use the following BibTeX entry:

```
@article{Xiao2022
  title = {Masked Visual Pre-training for Motor Control},
  author = {Tete Xiao and Ilija Radosavovic and Trevor Darrell and Jitendra Malik},
  journal = {arXiv:2203.06173},
  year = {2022}
}
```

### Acknowledgments

We thank NVIDIA IsaacGym and PhysX teams for making the simulator and preview code examples available.
