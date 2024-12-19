# MNIST CNN Model

A Convolutional Neural Network (CNN) implementation for MNIST digit classification using PyTorch.

## Project Structure 
- `model.py` - Contains the CNN architecture
- `train.py` - Training script
- `requirements.txt` - Project dependencies

## Model Architecture
The model consists of 7 convolutional layers with max pooling and ReLU activation functions:
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

## Setup and Installation

1. Clone the repository

The model will:
- Download MNIST dataset automatically
- Train for 15 epochs
- Save the best model in the `models/` directory
- Generate training logs

## Model Performance
- Training accuracy: ~99%
- Test accuracy: ~99%