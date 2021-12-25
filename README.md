# CoCoNet Experiments

This repository contains integration experiments for CoCoNet[https://github.com/parasailteam/coconet].
* `pytorch` directory contains modified PyTorch to be used by BERT and Megatron-LM.
* `pytorch/third_party/nccl/nccl` contains NCCL modified with scattered tensors implementation for Data Parallel Training and overlapping implementation for Model and Pipeline Parallel Inference.
* `BERT` contains NVIDIA BERT integrated with CoCoNet's scattered tensors implementation of Adam and LAMB.
* `model-parallel-Megatron-LM` contains Megatron-LM integrated with CoCoNet's optimized Model Parallel Inference.

# Installation

Follow all the instructions in https://github.com/parasailteam/coconet to setup and install all the pre-requisites.
Clone all submodules by

```
git submodule sync
git submodule update --init --recursive
```
## Install cuDNN

Download cudnn 8.2.1 for CUDA 11.3 from (https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-linux-x64-v8.2.1.32.tgz). 
Install cudnn by:
```
tar -xvf cudnn-11.3-linux-x64-v8.2.1.32.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
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

Check the torch installation by executing `import torch` in python in a directory other than pytorch directory

## Install Apex

NVIDIA Apex (https://github.com/NVIDIA/apex) is one of the baselines. After building Pytorch, we need to reinstall Apex.
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Install BERT and Megatron-LM Dependencies

```
conda install h5py boto3
```

To install DLLogger, run (separate commands):

```
pip install nvidia-pyindex
pip install nvidia-dllogger
```
# Data Parallel Training Experiment

Download the dataset from here[] and extract the directory to `coconet-experiments/BERT/training-data`. 
Execute `data_parallel_training_results.py` to replicate Table 1 results

```
cd coconet-experiments/BERT
python data_parallel_training_results.py coconet-experiments/BERT/training-data
```

