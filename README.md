# Image Classification Experiments with ResNet and ViT

## Overview

This repository contains a series of image classification experiments comparing the performance and training dynamics of ResNet and Vision Transformer (ViT) architectures. The experiments are conducted on two distinct datasets:

1.  **Cats vs. Dogs:** A binary classification problem using a custom dataset.
2.  **CIFAR-10:** A multi-class classification problem (10 classes) using the standard CIFAR-10 dataset.

The scripts are configured for fine-tuning pretrained models and include utilities for training, inference, and dataset analysis.

## Directory Structure

```
.
├── cifar-10-batches-py # <--- added to gitignore,but not used by any script
├── IE4483Dataset/      # <--- added to gitignore, so you prolly wont see it
│   └── datasets/       # <--- Parent directory for the Cats vs. Dogs dataset
│       ├── train/
│       │   ├── cat/
│       │   └── dog/
│       └── val/
│           ├── cat/
│           └── dog/
├── shell_scripts/    # <--- PBS script for NSCC
│   ├── ResNetTrain.sh
│   ├── ViTTrain.sh
│   └── ...
├── training_logs/      # Contains output logs from training runs
├── analyze_dataset_split.py
├── trainResNet.py
├── trainViT.py
├── trainViT_cifar10.py
├── requirements.txt
├── README.md           # This file
└── ...
```

## Setup

### 1. Environment
It is recommended to use a Conda environment with Python 3.10.
```bash
conda create -n vision-exp python=3.10
conda activate vision-exp
```

### 2. Install PyTorch
Install PyTorch with GPU support. Please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the command corresponding to your specific CUDA version. An example command is:
```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies
Install the remaining packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Datasets
- **Cats vs. Dogs:** Place your training and validation images in the `IE4483Dataset/datasets/` directory, following the structure shown above.
- **CIFAR-10:** Place the `cifar-10-batches-py` directory in the project root, as shown in the directory structure. The scripts are configured to use this local copy.

### 5. Hardware
A CUDA-enabled GPU is highly recommended for running the training scripts, as model fine-tuning is computationally intensive. The experiments were successfully run on HPC GPUs.

## How to Run

### 1. Cats vs. Dogs Classification
These scripts use a feature-extraction approach, where the pretrained backbone is frozen and only the final classifier layer is trained.

- **Train the ResNet Model:**
  ```bash
  python trainResNet.py
  ```
- **Train the ViT Model:**
  ```bash
  python trainViT.py
  ```

### 2. CIFAR-10 Classification
This script fine-tunes the entire ViT model on the CIFAR-10 dataset.

- **Train the ViT Model on CIFAR-10:**
  ```bash
  python trainViT_cifar10.py
  ```

### 3. Dataset Analysis
This script calculates and reports the class-wise distribution of the training and validation splits created for the CIFAR-10 experiment.

- **Analyze the CIFAR-10 Split:**
  ```bash
  python analyze_dataset_split.py
  ```

### Using Shell Scripts (for HPC)
The `shell_scripts/` directory contains sample scripts for submitting training jobs to an HPC cluster using a scheduler like SLURM. You may need to adapt them to your specific environment.

## Results
- The models trained on the **Cats vs. Dogs** dataset achieve high validation accuracy, demonstrating the effectiveness of the feature extraction fine-tuning method.
- The ViT model fine-tuned on **CIFAR-10** is capable of achieving approximately **98% test accuracy**, showcasing the power of transfer learning even when adapting to a low-resolution dataset.
- Training logs are saved to the `training_logs/` directory.