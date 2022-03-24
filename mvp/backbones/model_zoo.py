#!/usr/bin/env python3

"""Model zoo (based on Detectron, pycls, and CLIP)."""

import os
import re
import sys

from urllib import request as urlrequest

from mvp.backbones.vit import vit_s16


# Model download cache directory
_DOWNLOAD_CACHE = "/tmp/mvp-download-cache"

# Pretrained models
_MODELS = {
    "vits-mae-hoi": "https://www.dropbox.com/s/51fasmar8hjfpeh/mae_pretrain_hoi_vit_small.pth",
    "vits-mae-in": "https://www.dropbox.com/s/3whtrak5wsfzoaw/mae_pretrain_imagenet_vit_small.pth",
    "vits-sup-in": "https://www.dropbox.com/s/dw3uf5aff6yzmx3/sup_pretrain_imagenet_vit_small.pth",
}


def _progress_bar(count, total):
    """Report download progress. Credit:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)
    sys.stdout.write(
        "  [{}] {}% of {:.1f}MB file  \r".format(bar, percents, total / 1024 / 1024)
    )
    sys.stdout.flush()
    if count >= total:
        sys.stdout.write("\n")


def download_url(url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar):
    """Download url and write it to dst_file_path. Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    url = url + "?dl=1" if "dropbox" in url else url
    req = urlrequest.Request(url)
    response = urlrequest.urlopen(req)
    total_size = response.info().get("Content-Length").strip()
    total_size = int(total_size)
    bytes_so_far = 0
    with open(dst_file_path, "wb") as f:
        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break
            if progress_hook:
                progress_hook(bytes_so_far, total_size)
            f.write(chunk)
    return bytes_so_far


def cache_url(url_or_file, cache_dir=_DOWNLOAD_CACHE, download=True):
    """Download the file specified by the URL to the cache_dir and return the path to
    the cached file. If the argument is not a URL, simply return it as is.
    """
    is_url = re.match(r"^(?:http)s?://", url_or_file, re.IGNORECASE) is not None
    if not is_url:
        return url_or_file
    url = url_or_file
    fname = os.path.basename(url)
    cache_file_path = os.path.join(cache_dir, fname)
    if os.path.exists(cache_file_path):
        return cache_file_path
    cache_file_dir = os.path.dirname(cache_file_path)
    if not os.path.exists(cache_file_dir):
        os.makedirs(cache_file_dir)
    if download:
        print("Downloading remote file {} to {}".format(url, cache_file_path))
        download_url(url, cache_file_path)
    return cache_file_path


def available_models():
    """Retrieves the names of available models."""
    return list(_MODELS.keys())


def load(name):
    """Loads a pre-trained model."""
    assert name in _MODELS.keys(), \
        "Model {} not available".format(name)
    pretrained = cache_url(_MODELS[name])
    model, _ = vit_s16(pretrained)
    return model
