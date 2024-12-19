# MNIST CNN Model

A Convolutional Neural Network (CNN) implementation for MNIST digit classification using PyTorch.

## Project Structure 
- `model.py` - Contains the CNN architecture
- `train.py` - Training script
- `requirements.txt` - Project dependencies
- `evaluate_model.py` - Script to evaluate saved models

## Model Architecture
The model consists of 7 convolutional layers with max pooling and ReLU activation functions:
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

## Best Model Performance
Model checkpoint: mnist_model_20241220_025658.pth
- Epoch: 10
- Training Accuracy: 99.54%
- Test Accuracy: 99.36%

## Setup and Installation

1. Clone the repository

The model will:
- Download MNIST dataset automatically
- Train for 15 epochs
- Save the best model in the `models/` directory
- Generate training logs
