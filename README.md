# PRODIGY_ML_05
# **Food101 Calorie Estimation with Custom CNN**
This project aims to classify food images and estimate their calorie content using a custom Convolutional Neural Network (CNN) built with TensorFlow. The dataset used is a modified version of the Food101 dataset, focusing on only three labels for simplicity and faster training.

# **Project Overview**
The main goal is to build a CNN model capable of identifying food items from images and predicting their corresponding calorie content. The project involves data preprocessing, model building, training, validation, and evaluation.

## **Features**
Utilizes a subset of the Food101 dataset, containing images of three food categories.

Implemented early stopping to prevent overfitting.

Saved the best-performing model based on validation accuracy during training.

## **Dataset**
The original Food101 dataset contains 101 categories, but this project uses only three categories for simplicity:

Label 1 is apple_pie.

Label 2 is baby_back_ribs.

Label 3 is baklava.

Each category contains a balanced number of images, preprocessed for training.

## **Model Architecture**
The CNN model includes:

**Base Model:** ResNet50 with imagenet weights, excluding the top classification layer.

**Global Average Pooling Layer:** To reduce the dimensions of the feature maps.

**Dense Layer:** A fully connected layer with 128 units and ReLU activation.

**Dropout Layer:** To prevent overfitting, with a dropout rate of 0.4.

**Output Layer:** A dense layer with softmax activation, where the number of units equals the number of classes (3), using L2 regularization to minimize overfitting.

## **Training Process**
The model was trained for a maximum of 30 epochs, with early stopping implemented to halt training when no further improvements were observed in validation accuracy. The best model weights were saved based on the highest validation accuracy achieved.

## **Results**
Final Training Accuracy: Achieved high training accuracy.
Final Validation Accuracy: Reached up to 89.5% before early stopping.
