#!/usr/bin/env python3

"""Setup MVP."""

from setuptools import find_packages, setup


setup(
    name="mvp",
    version="0.1.0",
    description="Masked Visual Pre-training for Motor Control",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "iopath",
        "timm",
    ],
)
