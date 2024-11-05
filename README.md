# Image Morphing Detection using Error Level Analysis (ELA)

This project focuses on detecting morphing or manipulation in images using Error Level Analysis (ELA) and a VGG16-based deep learning model. By highlighting discrepancies caused by multiple compressions or pixel alterations, this project provides a reliable solution for identifying tampered images. This technology can be applied in areas such as digital forensics, journalism, and cybersecurity.

# Table of Contents
Overview
Features
Technologies
Dataset
Installation
Usage
Preprocessing
Training
Evaluation
Results
Future Improvements
License

# Overview
This project aims to detect image manipulations by applying Error Level Analysis to highlight differences between authentic and altered areas. ELA is an effective forensic tool for detecting compression discrepancies that signify morphing. A pre-trained VGG16 model is used to classify images as authentic or tampered based on ELA-transformed images.

# Features
Error Level Analysis (ELA): Highlights image compression differences to reveal possible tampering.
Deep Learning Model (VGG16): Leverages VGG16 for reliable classification of tampered vs. authentic images.
User-Friendly Interface: Provides a simple CLI for running predictions on new images.
High Accuracy: Achieved an accuracy rate of around 92% on test data, with robust performance across various tampering types.

# Technologies
Python 3.7
TensorFlow & Keras: For model training and evaluation.
OpenCV: For image preprocessing and ELA transformation.
NumPy & Pandas: For data handling and manipulation.

# Dataset
The dataset consists of:
Authentic Images: Original images without modifications.
Tampered Images: Images are altered with common morphing techniques such as splicing, retouching, and blending.
Each image undergoes ELA transformation to generate an error map that highlights areas with different compression levels, which is useful for detecting tampering.

Note: Due to dataset constraints, this project includes data augmentation techniques to balance the class distribution.

# Installation
Clone the repository:
bash
git clone https://github.com/your-username/image-morphing-detection.git
cd image-morphing-detection
Install dependencies:
bash
pip install -r requirements.txt
# Usage
Running Image Morphing Detection
Apply ELA Transformation on an image:
python ela_transform.py --input_path <path_to_image> --output_path <path_to_output>
Make Predictions on a new image:
python predict.py --image_path <path_to_image>

# CLI Arguments
--input_path: Path to the input image for ELA transformation.
--output_path: Path to save the ELA-transformed image.
--image_path: Path to the input image for making predictions.
# Preprocessing
Each image undergoes preprocessing as follows:

ELA Transformation: Each input image is compressed and compared with the original to generate an error map.
Resizing and Normalization: Images are resized to 256x256 pixels and normalized to match the input format for VGG16.
Augmentation: Random augmentations like flipping and rotation are applied to balance the dataset.
Training
The VGG16 model is pre-trained on ImageNet and fine-tuned for binary classification.
Data Split: Data is split into training, validation, and test sets.
Fine-Tuning: Additional dense layers are added with dropout regularization to prevent overfitting.

# Training: 
The model is trained with binary cross-entropy loss and accuracy as the evaluation metric.
Training Command
python train.py --epochs 30 --batch_size 32 --learning_rate 0.0001
# Hyperparameters
Epochs: 30
Batch Size: 32
Learning Rate: 0.0001
Dropout: 0.5
# Evaluation
The model’s performance is evaluated on the test set using accuracy, precision, recall, and F1-score.

python evaluate.py --test_data_path <path_to_test_data>

# Results

The model achieved the following metrics:
Accuracy: 92%
Precision: 0.91
Recall: 0.90
F1-score: 0.90
The model effectively distinguishes tampered images, with minor challenges in highly compressed authentic images that sometimes mimic tampering.

# Future Improvements
Enhance Model Architecture: Experiment with ResNet or EfficientNet to improve accuracy.
Additional Forensic Techniques: Combine ELA with noise and shadow analysis for better tamper detection.
Real-Time Application: Optimize for lower latency in real-time applications by leveraging edge AI capabilities.
# License
This project is licensed under the MIT License. See the LICENSE file for more details.

