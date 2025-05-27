# Inserting faults to ResNet-Spiking

This repository expands the code from [SEW-ResNet](https://github.com/fangwei123456/Spike-Element-Wise-ResNet). The idea is to insert faults to pre-trained weights with approximate adders, and then evaluate the performance of defective spiking models.

To run the code without any issues, we installed the same conda environment as SEW-ResNet.

## Environment Setup

```bash
conda env create --name <your-env-name> python=3.9
conda activate <your-env-name>
conda install gcc_linux-64
conda install gxx_linux-64
conda install wheel==0.36.2
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.20.2 tensorboard==2.5.0 six==1.16.0 scipy==1.7.0 setuptools==57.0.0 tqdm==4.61.1 matplotlib==3.4.2 ninja==1.10.2 protobuf==3.8.0 pillow==8.2.0 cython binary_fractions
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<the-path-to-conda>/miniconda3/lib
export PATH=$PATH:<the-path-to-conda>/miniconda3/bin
```
### SpikingJelly Setup
The origin codes uses a specific SpikingJelly. To maximize reproducibility, the user can download the latest SpikingJelly and rollback to the version that we used to train:

```bash
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
git reset --hard 2958519df84ad77c316c6e6fbfac96fb2e5f59a3
python setup.py install
```

Here is the commit information:

```bash
commit 2958519df84ad77c316c6e6fbfac96fb2e5f59a3
Author: fangwei123456 <fangwei123456@pku.edu.cn>
Date:   Wed May 12 18:05:33 2021 +0800
```

Note that there is a bug in this version of SpikingJelly:

Bug: MultiStepParametricLIFNode

https://github.com/fangwei123456/spikingjelly/blob/master/bugs.md

## Spike-Element-Wise-ResNet Setup

This repository contains the codes for the paper [Deep Residual Learning in Spiking Neural Networks](https://arxiv.org/abs/2102.04159). An identical seed is used during training, and it can ensure that the user can get almost the same accuracy when using our codes to train. 

Some of the trained models at last epoch or max test acc1 for **ImageNet** and **DVS Gesture** are available at: https://figshare.com/articles/software/Spike-Element-Wise-ResNet/14752998. The model with max test acc1 on **CIFAR10-DVS** is also available at this url, which was asked by a researcher. Some other models of **CIFAR10-DVS** are missed.

```bash
cd 
git clone https://github.com/klab-aizu/ResNet_Spiking.git
cd ResNet_Spiking
```
# Running Examples

### Train on ImageNet

```bash
cd imagenet
```

Train the Spiking ResNet-18 with zero-init:

```bash
python -m torch.distributed.launch --use_env train.py --cos_lr_T 320 --model spiking_resnet18 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320 --data-path /home/data/ImageNet --device cuda:0 --zero_init_residual
```

Train the SEW ResNet-18:

```bash
python -m torch.distributed.launch --use_env train.py --cos_lr_T 320 --model sew_resnet18 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --connect_f ADD --T 4 --lr 0.1 --epoch 320 --data-path /home/data/ImageNet --device cuda:0
```

Train the SEW ResNet-18 with 8 GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --cos_lr_T 320 --model sew_resnet18 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --connect_f ADD --T 4 --lr 0.1 --epoch 320 --data-path /home/data/ImageNet
```

### Test-only with Checkpoints (pre-trained weights)

```bash
python -m torch.distributed.launch --use_env train.py --cos_lr_T 320 --model spiking_resnet18 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320 --data-path /home/data/ImageNet --device cuda:0 --zero_init_residual --resume /home/data/checkpoint-imagenet/sew18_checkpoint_319.pth --test-only
```
### Test-only with Faults

Turn on the flag **is-fault**.

```bash
python -m torch.distributed.launch --use_env train.py --cos_lr_T 320 --model spiking_resnet18 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320 --data-path /home/data/ImageNet --device cuda:0 --zero_init_residual --resume /home/data/checkpoint-imagenet/sew18_checkpoint_319.pth --test-only --is-fault True --ftype flip-bit --frate 0.01
```

The code supports 4 type of faults (--ftype): **flip-bit**, **stuck-at-zero**, **stuck-at-one**, **power-gate**.

Change the fault rate with **frate**.

### Test-only with Approximate Adders.

TBD.

