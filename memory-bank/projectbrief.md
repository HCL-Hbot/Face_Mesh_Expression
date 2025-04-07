# Project Brief

## Core Objective
Develop a facial expression recognition system using MediaPipe FaceMesh for landmark detection and TensorFlow for model training.

## Key Features
- MediaPipe FaceMesh integration for 468 facial landmarks
- Data preprocessing pipeline
- TensorFlow model training and evaluation
- Hyperparameter tuning capabilities
- Comprehensive logging and model versioning

## Technical Specifications
- Input: 240x320 images
- Output: 5-class classification
- Model Input Shape: 216 features
- Batch Size: 8
- Epochs: 50
- Validation Split: 20%
- Face Mesh Detection Confidence: 0.5
- Face Mesh Tracking Confidence: 0.5
- Max Hyperparameter Tuning Trials: 1000

## Directory Structure
- Output Directory
  - Model files
  - Training history logs
  - TensorBoard logs
  - Facial landmark data

## Data Processing
- Handles JPG, PNG, JPEG formats
- Outlier detection using Isolation Forest
- Removal of zero-value landmarks
- Filtering for specific facial features (eyes, mouth)
- Conversion to TensorFlow Dataset format
- One-hot encoding of labels
