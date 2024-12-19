# MNIST CNN Model

A Convolutional Neural Network (CNN) implementation for MNIST digit classification using PyTorch.

## Project Structure 
- `model_13.py` - Contains the optimized CNN architecture
- `train.py` - Training script
- `requirements.txt` - Project dependencies
- `evaluate_model.py` - Script to evaluate saved models

## Model Architecture (Model_13)
The model uses an efficient architecture with:
- Input Block: Expansion layers (1→10→16 channels)
- Feature Extraction: Channel reduction (16→8) followed by focused extraction (8→16)
- Batch Normalization after each convolution
- Strategic dropout (5% early, 2.5% later layers)
- Global Average Pooling for final output
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

## Best Model Performance
Model checkpoint: mnist_model_20241220_040419.pth
- Epoch: 10
- Training Accuracy: 98.34%
- Test Accuracy: 99.19%

## Setup and Installation

1. Clone the repository

The model will:
- Download MNIST dataset automatically
- Train for 15 epochs
- Save the best model in the `models/` directory
- Generate training logs
