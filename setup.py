# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="convnext-perceptual-loss",
    version="0.1.0",
    author="Yipeng Sun",
    author_email="yipeng.sun@fau.de",
    description="A perceptual loss using ConvNext models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sypsyp97/convNext_perceptual_loss",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
    ],
)
