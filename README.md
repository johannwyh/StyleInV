# StyleInV: A Temporal Style Modulated Inversion Network for Unconditional Video Generation

This repository contains the official PyTorch implementation of the following paper:

> **StyleInV: A Temporal Style Modulated Inversion Network for Unconditional Video Generation**<br>
> [Yuhan Wang](https://johann.wang/), [Liming Jiang](https://liming-jiang.com/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
> In *ICCV 2023*.

From [MMLab@NTU](https://www.mmlab-ntu.com/) affiliated with S-Lab, Nanyang Technological University

[[Paper](https://johann.wang/uploads/iccv2023_styleinv.pdf)] |
[[Project Page](https://www.mmlab-ntu.com/project/styleinv/index.html)] |
[[Video](https://www.youtube.com/watch?v=R_v_L-32_Vo)]

<img src="./assets/main.gif" alt="Main experiment result 256x256" width="75%" height="auto"/>

**From left to right**: DeeperForensics, FaceForensics, SkyTimelapse, TaiChi

<img src="./assets/init.gif" alt="Initial-frame conditioned and style transfer 256x256" width="75%" height="auto"/>

**From left to right**: In-the-wild image, pSp inversion, raw animation, style transfer
## Updates
- [09/11/2023] Source code is available. Tutorial on the environment, usage, and data/model preparation is on the way.
- [08/2023] Accepted by ICCV 2023. The code is coming soon!

## Training Pipeline

### 1. StyleGAN2 pretraining

```bash
train_stylegan2.py
```

### 2. pSp pretraining for StyleInV initialization

```bash
train_psp.py
```

### 3. StyleInV training

```bash
train_styleinv.py
```

### 4. Finetuning-based style transfer

```bash
train_stylegan2.py
```

### Inference

### 1. Generate a video dataset

```bash
generate_styleinv_video.py
```

### 2. Compute the quantitative metrics

```bash
scripts/calc_metrics_video.py
```

### 3. Animation and style transfer

```bash
generate_animation.py
```

## Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@InProceedings{wang2023styleinv,
    title = {{StyleInV}: A Temporal Style Modulated Inversion Network for Unconditional Video Generation},
    author = {Wang, Yuhan and Jiang, Liming and Loy, Chen Change},
    booktitle = {ICCV},
    year = {2023}
}   
```

## Acknowledgement

This codebase is maintained by [Yuhan Wang](https://johann.wang/).

This repo is built on top of following works:
* [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)
* [StyleGAN-V](https://github.com/universome/stylegan-v)
* [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)