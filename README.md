# CoCoNet-Experiments

This repository contains integration experiments for CoCoNet[https://github.com/parasailteam/coconet].
* `pytorch` directory contains modified PyTorch to be used by BERT and Megatron-LM.
* `nccl` contains NCCL modified with scattered tensors implementation for Data Parallel Training and overlapping implementation for Model and Pipeline Parallel Inference.
* `BERT` contains NVIDIA BERT integrated with CoCoNet's scattered tensors implementation of Adam and LAMB.
* `Megatron-LM` contains Megatron-LM integrated with CoCoNet's optimized Model Parallel Inference.

# Installation

Follow all the instructions in https://github.com/parasailteam/coconet to setup and install all the pre-requisites.
Clone all submodules by

```
git submodule sync
git submodule update --init --recursive
```

## Build PyTorch
We will now build modified PyTorch. However, make sure to remove existing PyTorch installations from the conda environment.

```
conda remove torch
```

Install pre-requisites for building PyTorch

```
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
conda install -c pytorch magma-cuda90
```

Build PyTorch by:

```
cd pytorch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

# Experiments
