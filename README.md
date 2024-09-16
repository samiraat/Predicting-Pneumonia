# Pneumonia Detection using Deep Learning

This repository contains a deep learning project focused on detecting pneumonia from chest X-ray images. The project utilizes convolutional neural networks (CNNs) for image classification, aiming to differentiate between normal and pneumonia-infected lungs.

## Project Overview

Pneumonia is a serious lung infection that requires timely diagnosis and treatment. Automated detection using deep learning techniques can assist in improving diagnostic accuracy and speed. This project implements a deep learning pipeline to classify chest X-rays into two categories: Normal and Pneumonia.

## Repository Structure

- `Pneumonia_Detection.ipynb`: The Jupyter Notebook containing the full pipeline, from data loading and preprocessing to model training and evaluation.
- `data/`: Directory expected to contain the training, validation, and test images organized in subfolders (e.g., `train/PNEUMONIA/`, `train/NORMAL/`, etc.).
- `models/`: Directory to save trained models and weights (not included by default).

## Key Features

- **Data Preprocessing**: Images are resized to 224x224 pixels and normalized. Data augmentation techniques are applied to improve model generalization.
- **Model Architecture**: A CNN model built using Keras, leveraging transfer learning with the VGG16 architecture.
- **Evaluation Metrics**: The model's performance is evaluated using accuracy, precision, recall, F1 score, and ROC-AUC.

## Dependencies

- TensorFlow
- Keras
- OpenCV
- Scikit-learn
- Matplotlib
- Seaborn
