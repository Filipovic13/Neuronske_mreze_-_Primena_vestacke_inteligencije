# Deep Learning Model Comparison for Image Classification

This project compares three different Convolutional Neural Network (CNN) architectures—**ConvNet**, **DeepConvNet**, and **WideConvNet**—for image classification tasks. The models were trained on a dataset, evaluated based on training and test accuracy, and their performance was plotted over multiple epochs.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Model Architectures](#model-architectures)
  - [ConvNet](#convnet)
  - [DeepConvNet](#deepconvnet)
  - [WideConvNet](#wideconvnet)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [Conclusion](#conclusion)
- [Saving and Loading Models](#saving-and-loading-models)
- [License](#license)

## Introduction

This project explores three different CNN architectures for classifying images. The models are trained using **PyTorch** framework. Their performance is evaluated based on accuracy during training and on an unseen test set.

## Installation

To run this project, you need to install the following dependencies:

- **PyTorch**
- **Torchvision**
- **Matplotlib**

Ensure you have **PyTorch** installed with GPU support if you plan to use a CUDA-compatible device.

## Model Architectures

### ConvNet

A basic CNN architecture with the following layers:

- Convolutional layers with increasing output channels (3 → 64 → 128 → 256).
- MaxPooling layers to reduce spatial dimensions.
- Fully connected layers to classify the features learned by the network.

### DeepConvNet

A deeper CNN architecture with more convolutional layers, designed to capture more complex patterns:

- Convolutional layers with increasing output channels (3 → 32 → 64 → 128 → 256).
- MaxPooling layers to reduce spatial dimensions.
- Fully connected layers to classify the features learned by the network.

### WideConvNet

A CNN with wider layers but fewer layers compared to DeepConvNet:

- Convolutional layers with increasing output channels (3 → 64 → 128 → 256).
- MaxPooling layers to reduce spatial dimensions.
- Fully connected layers to classify the features learned by the network.

## Training

The models are trained using **CrossEntropyLoss** and the **Adam optimizer**. The training process includes:

- Forward pass
- Backward pass with gradient updates
- Accuracy calculation at each epoch

Loss and accuracy are tracked for each model and saved to files.

## Testing

After training, the models are evaluated on a test set to assess their performance on unseen data. The test accuracy is computed for each model.

## Results

The models showed high performance during training and testing:

- **WideConvNet** achieved a test accuracy of **99.67%**.
- **ConvNet (using Adam optimizer)** achieved a test accuracy of **99.23%**.
- **DeepConvNet** achieved a test accuracy of **98.02%**.

### Training Accuracy and Loss:

- **WideConvNet** performed the best during training, reaching **99.84%** accuracy.
- **ConvNet** followed with **99.25%** accuracy.
- **DeepConvNet** achieved **98.25%** accuracy, slightly lower than the other two.

### Plots:

Training accuracy and loss curves for each model are plotted for visual comparison.

## Saving and Loading Models

Models, along with their training losses and accuracies, are saved for later use. The following functions are provided:

- **Saving models**: Saves the entire model and its associated loss and accuracy data.
- **Loading models**: Loads a saved model and its associated loss and accuracy data for further analysis or re-training.

## Conclusion

- **WideConvNet** outperformed the other two models both in terms of training and test accuracy, making it the most effective architecture for this task.
- **ConvNet** was a solid performer, achieving competitive results.
- **DeepConvNet** performed well but had slightly lower accuracy, possibly due to the deeper architecture requiring more data or training time.
