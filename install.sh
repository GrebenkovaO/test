#!/bin/bash
git clone git@github.com:CompVis/EDGS.git --recursive 
cd EDGS
git submodule update --init --recursive 

conda create -y -n edgs python=3.9 pip
conda activate edgs

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes
conda install nvidia/label/cuda-12.1.0::cuda-toolkit --yes

# Optionally set path to CUDA
#export CUDA_HOME=/usr/local/cuda-12.1
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#export PATH=$CUDA_HOME/bin:$PATH

pip install -e submodules/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e submodules/gaussian-splatting/submodules/simple-knn

# For VGGT
pip install Pillow huggingface_hub einops safetensors sympy==1.13.1


pip install wandb hydra-core tqdm torchmetrics lpips matplotlib rich plyfile imageio imageio-ffmpeg
conda install numpy=1.26.4 -y -c conda-forge --override-channels

pip install -e submodules/RoMa
conda install anaconda::jupyter --yes


