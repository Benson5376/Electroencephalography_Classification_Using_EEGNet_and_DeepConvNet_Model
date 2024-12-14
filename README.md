# Electroencephalography_Classification_Using_EEGNet_and_DeepConvNet_Model
This project implements EEG classification models, specifically EEGNet and DeepConvNet, using the BCI Competition III dataset. It explores the impact of different activation functions (ReLU, Leaky ReLU, and ELU) on model performance. The goal is to achieve high accuracy in classifying motor imagery EEG signals.
Based on the provided files, here’s a draft for the README in Markdown format for your project.

---

# EEG Classification with EEGNet and DeepConvNet

## Table of Contents
- [Introduction](#introduction)
- [Lab Objective](#lab-objective)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
  - [EEGNet](#eegnet)
  - [DeepConvNet](#deepconvnet)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Results](#results)
- [Configuration](#configuration)
- [Contributors](#contributors)
- [License](#license)

---

## Introduction

This project implements EEG classification models, specifically **EEGNet** and **DeepConvNet**, using the BCI Competition III dataset. It explores the impact of different activation functions (ReLU, Leaky ReLU, and ELU) on model performance. The goal is to achieve high accuracy in classifying motor imagery EEG signals.

---

## Lab Objective

1. Implement **EEGNet** and **DeepConvNet** for EEG classification.
2. Compare the performance of the models using **ReLU**, **Leaky ReLU**, and **ELU** activation functions.
3. Visualize accuracy trends during the training and testing phases.
4. Optimize and analyze model performance on the BCI Competition III dataset.

---

## Dataset

- Dataset: BCI Competition III - IIIb Cued Motor Imagery.
- Classes: Left Hand, Right Hand.
- Channels: 2 bipolar EEG channels.
- Preprocessed files:
  - `S4b_train.npz`, `X11b_train.npz`
  - `S4b_test.npz`, `X11b_test.npz`

Preprocessing details are handled by `dataloader.py`.

---

## Model Architectures

### EEGNet
- **First Convolution Layer:** 
  - 1x51 filters with padding and BatchNorm.
- **Depthwise Convolution:** 
  - 2x1 filters, grouped, with BatchNorm, Leaky ReLU, AvgPool, and Dropout.
- **Separable Convolution:** 
  - 1x15 filters with padding, BatchNorm, Leaky ReLU, AvgPool, and Dropout.
- **Final Classification Layer:** Fully connected layer with 2 output neurons.

### DeepConvNet
- **Layer Sequence:**
  1. Convolution: 1x5 filter with 25 channels.
  2. Max Pooling and Dropout: Applied after each convolution block.
  3. Successive depth increase: Channels increase from 25 → 50 → 100 → 200.
  4. Final Classification Layer: Fully connected layer with 2 output neurons.

Both models are defined in `models.py`.

---

## Installation

### Prerequisites
- Python 3.x
- PyTorch
- NumPy
- Matplotlib

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/eeg-classification.git
   cd eeg-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place preprocessed data files in the project directory.

---

## Usage

1. Prepare the data:
   ```bash
   python dataloader.py
   ```
2. Train models:
   ```bash
   python train.py --model eegnet --activation relu
   python train.py --model deepconvnet --activation leakyrelu
   ```
3. Visualize results:
   ```bash
   python plot_results.py
   ```

---

## Features
- Multiple model architectures (EEGNet, DeepConvNet).
- Customizable activation functions (ReLU, Leaky ReLU, ELU).
- Data preprocessing and handling of missing values.
- Visualization of accuracy trends.

---

## Results

### Comparison of Accuracy
- **EEGNet**:
  - Highest accuracy with ELU: **XX.XX%**.
- **DeepConvNet**:
  - Highest accuracy with Leaky ReLU: **XX.XX%**.

### Training and Testing Accuracy Trends
Plots generated using `matplotlib` showcase the training/testing accuracy across epochs.

---

## Configuration

### Hyperparameters
- **Batch Size:** 64
- **Learning Rate:** 0.01
- **Epochs:** 300
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss

These parameters can be adjusted in `config.json`.

---

## Contributors
- **Name:** Hua-En (Benson) Lee
- **Email:** enen1015@gmail.com

---

## License

This project is licensed under the [MIT License](LICENSE).

---
