# FaceMesh Pipeline

This README describes the FaceMesh pipeline, which is used for facial expression recognition.

## Overview

The pipeline consists of the following steps:

1.  Data capture
2.  Configuration
3.  Data loading and preprocessing
4.  File handling
5.  Model graph visualization
6.  Image preparation
7.  Model training
8.  Pipeline execution
9.  TensorBoard integration
10. Training callbacks

## Scripts

*   `capture_facemesh_data.py`: Captures facial landmark data using MediaPipe's FaceMesh. This script captures facial landmark data from a webcam, classifies facial expressions (neutral, frowning, smile, smile_with_teeth), and saves the landmark coordinates with classifications to a CSV file for machine learning training.
*   `config.py`: Provides a centralized configuration system for the FaceMesh application. It manages all configurable parameters, including model architecture, training parameters, data processing settings, file paths, and hyperparameter tuning ranges.
*   `data_load.py`: Loads and preprocesses the data. It handles loading facial landmark data from CSV files or images, applying data augmentations, filtering outliers, and converting the data into TensorFlow datasets for model training.
*   `fileHandler.py`: Handles file operations. It provides utility functions for loading image files, creating timestamped output directories, and managing CSV files for facemesh landmarks.
*   `graph_model.py`: Visualizes the model architecture. It loads a trained Keras model and generates a visual diagram of the network structure using GraphViz.
*   `imagePrep.py`: Prepares the images. It provides functions for image preprocessing, including grayscale conversion, face detection, cropping, normalization, and landmark visualization.
*   `model_train_hyperband_XLA_acceration_caching.py`: Trains the model using Hyperband optimization and XLA acceleration. This script implements a 1D CNN model training pipeline with Hyperband optimization for efficient hyperparameter tuning and XLA acceleration for improved CPU performance.
*   `model_train_hyperparameters.py`: Trains the model using pre-defined hyperparameters. This script implements the training phase using pre-tuned hyperparameters from a previous Hyperband tuning process.
*   `run_pipeline.py`: Orchestrates the pipeline. It manages the execution of the ML pipeline components, including parallel process isolation, real-time output streaming, and robust error handling.
*   `tensorboard_utils.py`: Provides utilities for TensorBoard integration. It provides functionality to set up and manage TensorBoard for visualizing training metrics and model graphs.
*   `training_callbacks.py`: Configures the training callbacks. It provides a centralized configuration for Keras training callbacks, implementing best practices for model training monitoring and optimization, such as model checkpointing, early stopping, and learning rate adaptation.

## Data Flow

\`\`\`mermaid
graph LR
    A[capture_facemesh_data.py: Capture Landmarks] --> B(data_load.py: Load & Preprocess Data)
    B --> C(imagePrep.py: Prepare Images)
    C --> D(model_train_hyperband_XLA_acceration_caching.py: Hyperband Training)
    D --> E(model_train_hyperparameters.py: Train with Hyperparameters)
    E --> F(graph_model.py: Visualize Model)
    F --> G(tensorboard_utils.py: TensorBoard Integration)
    G --> H(training_callbacks.py: Configure Callbacks)
    H --> I(run_pipeline.py: Orchestrate Pipeline)
\`\`\`

## Model Architecture

\`\`\`mermaid
graph LR
    A[Input: (216,)] --> B(Conv1D: filters, kernel_size, activation)
    B --> C(MaxPooling1D: pool_size=2)
    C --> D(Conv1D: filters, kernel_size, activation)
    D --> E(MaxPooling1D: pool_size=2)
    E --> F(Flatten)
    F --> G(Dense: units, activation)
    G --> H(Output: num_classes, softmax)
\`\`\`

## Ignore

The `Legacy` and `Debug` directories are ignored.
