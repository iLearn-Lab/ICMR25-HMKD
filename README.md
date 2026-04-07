# HMKD: Heterogeneous Model Knowledge Distillation via Dual Alignment for Semantic Segmentation

> **Official PyTorch implementation of the ICMR 2025 paper "Heterogeneous Model Knowledge Distillation via Dual Alignment for Semantic Segmentation".**

## Authors

**Mingzhu Xu**<sup>1</sup>, **Jing Wang**<sup>1</sup>, **Mingcai Wang**<sup>1</sup>, **Yiping Li**<sup>1</sup>, **Yupeng Hu**<sup>1</sup>, **Xuemeng Song**<sup>2</sup>\*, **Weili Guan**<sup>3</sup>

<sup>1</sup> `Shandong University`  
<sup>2</sup> `City University of Hong Kong`  
<sup>3</sup> `Harbin Institute of Technology (Shen Zhen)`  
\* Corresponding author

## Links

- **Paper**: [`ICMR 2025`](#)
- **Code Repository**: [`GitHub`](https://github.com/iLearn-Lab/HMKD-ICMR)

---

## Table of Contents

- [Introduction](#introduction)
- [Highlights](#highlights)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Checkpoints / Models](#checkpoints--models)
- [Dataset / Benchmark](#dataset--benchmark)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Introduction

This project is the official implementation of the paper **"Heterogeneous Model Knowledge Distillation via Dual Alignment for Semantic Segmentation"**.

HMKD proposes a heterogeneous model knowledge distillation framework for semantic segmentation, which effectively addresses the challenge of knowledge transfer caused by structural differences between teacher and student networks through a dual alignment mechanism:
- **Core Idea**：Utilize dual alignment mechanisms to perform knowledge alignment simultaneously in both feature space and logit space.
- **Distillation Scenario**：Supports knowledge transfer from large heterogeneous models (e.g., SegFormer) to lightweight models (e.g., DeepLabV3-ResNet18).
- **This Repository Provides**：Complete distillation training code, pretrained weight interfaces for multiple backbone networks, and evaluation pipelines on mainstream segmentation datasets.

### Example Description

We present **HMKD**, a framework for **Semantic Segmentation via Knowledge Distillation**.  
Our method addresses **architectural heterogeneity** between teacher and student models by introducing **dual alignment mechanisms**.  
This repository provides the official implementation, distilled checkpoints, and evaluation scripts.

---

## Highlights

- Supports **heterogeneous model distillation** (e.g., Transformer to CNN).
- Proposes a **Dual Alignment** mechanism that significantly improv
- Achieves strong performance on standard datasets such as **Cityscapes** and **CamVid**.

---

## Project Structure

```text
.
├── configs/               # Experiment configuration files
├── data/                  # Dataset paths and preprocessing scripts
├── models/                # Implementations of teacher and student networks
├── train_NEW_AEU_kd.py    # Core distillation training script
├── README.md
└── requirements.txt       # Environment dependencies
```

---

## Installation

### 1. System Requirements
- Ubuntu 20.04.4 LTS
- Python 3.8.10 (Recommended: [Anaconda](https://www.anaconda.com/))
- CUDA 11.3 / PyTorch 1.11.0 / NCCL 2.10.3

### 2. Clone the repository
```bash
git clone [https://github.com/iLearn-Lab/HMKD-ICMR.git](https://github.com/iLearn-Lab/HMKD-ICMR.git)
cd HMKD-ICMR
```

### 3. Install Python packages
```bash
pip install timm==0.3.2 mmcv-full==1.2.7 opencv-python==4.5.1.48
```

---

## Checkpoints / Models

### 1. Initialization Weights (for Training)
Please download the following pretrained weights according to your experimental needs:
- **DeepLabV3 - ResNet-18**: [resnet18.pth](https://download.pytorch.org/models/resnet18-5c106cde.pth)
- **DeepLabV3 - ResNet-101**: [resnet101.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)
- **Segformer - mit-b0**: [segformerb0.pth](https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing)
- **Segformer - mit-b4**: [segformerb4.pth](https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing)

### 2. Trained HMKD Weights (for Testing)
- **Download**: [Baidu Drive](https://pan.baidu.com/s/1xw_6ts5VNV73vXeOLAokwQ?pwd=jvx8) (Password: `jvx8`)

---

## Dataset / Benchmark

| Dataset | Train Size | Val Size | Test Size | Classes |
| :--- | :---: | :---: | :---: | :---: |
| Cityscapes | 2975 | 500 | 1525 | 19 |
| CamVid | 367 | 101 | 233 | 11 |

lease generate the corresponding dataset path list files (`.txt`) in the code.

---

## Usage

### Training
After downloading the pretrained weights and datasets, launch the distillation task using distributed training:

```bash
# training mode
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 train_NEW_AEU_kd.py > train_distill.log 2>&1 &

# debugging mode
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_NEW_AEU_kd.py
```

### Testing
Download the distilled weights, modify the `path` variable in the code, and run:
```bash
python evaluate.py --model_id HMKD --dataset cityscapes
```

---

## Citation

If you use this code or method in your research, please cite our paper:

```bibtex
@ARTICLE{HMKD,
  author={Xu, Mingzhu and Wang, Jing and Wang, Mingcai and Li, Yiping and Hu, Yupeng and Song, Xuemeng and Guan, Weili},
  journal={ICMR}, 
  title={Heterogeneous Model Knowledge Distillation via Dual Alignment for Semantic Segmentation}, 
  year={2025}
}
```

---

## License

This project is released under the Apache License 2.0.
