"""
Model Training with Pre-defined Hyperparameters

This module implements the training phase of the facial expression recognition pipeline, using
hyperparameters that were optimized through a previous Hyperband tuning process. The module
handles the complete training workflow from model construction to evaluation and artifact storage.

Components:
1. Model Architecture
   - Configurable CNN with Conv1D layers for landmark sequence processing
   - Flexible normalization strategies (batch, layer, standard)
   - Dense layers with customizable regularization
   - Multiple activation functions and constraint options

2. Training Process
   - Automated dataset loading and validation
   - Comprehensive training with validation split
   - Dynamic batch size handling
   - Training progress monitoring via callbacks
   - Gradient clipping for stable training

3. Regularization Features
   - L1/L2 regularization with configurable strengths
   - Dropout layers with tunable rates
   - Batch normalization options
   - Weight constraints for preventing exploding gradients

4. Performance Optimization
   - Automated hyperparameter application
   - Memory-efficient dataset handling
   - Configurable training epochs
   - Early stopping capabilities

5. Model Management
   - Automatic versioning system
   - Structured artifact organization
   - Comprehensive performance logging
   - Hyperparameter configuration storage

Dependencies:
- keras (>= 2.0): Neural network implementation
- tensorflow (>= 2.0): Backend operations and optimizations
- data_load: Custom dataset management (requires landmarks dataset)
- training_callbacks: Training monitoring and checkpointing
- config: Project-wide configuration settings

Example Usage:
    ```python
    # Load pre-tuned hyperparameters
    with open("best_hyperparameters.json", "r") as f:
        hyperparams = json.load(f)

    # Execute training pipeline
    train_and_save_model("output/models", hyperparams)
    ```

Note: This module requires pre-tuned hyperparameters from the Hyperband optimization
phase (see model_train_hyperband.py). Ensures reproducible model training by using
the same architecture and parameters that achieved optimal performance during tuning.
"""

import json
import os
from datetime import datetime

import keras
import tensorflow as tf

from config import config
from data_load import get_dataset_details, get_datasets, validate_landmark_count
from training_callbacks import get_callbacks


def create_model_from_hyperparams(input_shape, num_classes, hyperparams, sample_label):
    """
    Creates and compiles a facial expression recognition CNN model using pre-tuned hyperparameters.

    This function builds a complete neural network architecture optimized for facial landmark sequence
    processing. The architecture is highly configurable through hyperparameters and includes:

    Architecture Components:
    1. Input Processing
       - Reshapes input landmarks for Conv1D processing
       - Configurable input normalization (standard/layer/batch)

    2. Convolutional Layers
       - Dynamic number of Conv1D layers
       - Customizable filters and kernel sizes
       - Multiple activation function options
       - Optional batch normalization
       - Configurable pooling strategies

    3. Regularization Features
       - L1/L2/combined regularization options
       - Tunable dropout rates
       - Weight constraints (max_norm, unit_norm, min_max_norm)

    4. Dense Layer Configuration
       - Adjustable number of units
       - Multiple activation options
       - Independent regularization settings
       - Optional batch normalization

    5. Training Optimization
       - Gradient clipping options
       - Configurable learning rates
       - Automatic loss function selection

    Args:
        input_shape (tuple): Shape of input data (e.g., (216,) for 68 landmarks * 3 coordinates)
        num_classes (int): Number of output classes (facial expressions to recognize)
        hyperparams (dict): Pre-tuned hyperparameters from Hyperband optimization including:
                          - Normalization settings
                          - Conv1D layer configurations
                          - Dense layer settings
                          - Regularization parameters
                          - Training optimization values
        sample_label (Tensor): Example label tensor to determine appropriate loss function format

    Returns:
        keras.Model: Compiled model with:
                    - Configured architecture based on hyperparameters
                    - Optimization settings
                    - Loss function and metrics

    Note:
        The architecture exactly mirrors the one used in Hyperband optimization to ensure
        consistent performance between tuning and final training phases. Any modifications
        to the architecture should be reflected in both this function and the Hyperband
        tuning script.
    """
    inputs = keras.layers.Input(shape=input_shape)
    # Reshape for Conv1D input
    x = keras.layers.Reshape((input_shape[0], 1))(inputs)

    # Normalization (mimicking model_train_hyperband)
    # Type of input normalization: "standard" (standardize inputs), "layer" (layer normalization), "batch" (batch normalization), or "none"
    norm_type = hyperparams.get("Normalization Type", "batch")
    if norm_type == "standard":
        x = keras.layers.Normalization()(x)
    elif norm_type == "layer":
        x = keras.layers.LayerNormalization()(x)
    elif norm_type == "batch":
        x = keras.layers.BatchNormalization()(x)
    # else "none": do nothing

    # Determine number of Conv1D layers based on keys (using max index found)
    conv_indices = [
        int(key.split("_")[1])
        for key in hyperparams
        if key.startswith("Conv1D_") and "_Filters" in key
    ]
    num_conv_layers = max(conv_indices) if conv_indices else 0

    # Build Convolutional Layers
    for i in range(1, num_conv_layers + 1):
        # Read parameters for this conv layer
        # Number of filters (feature detectors) in this Conv1D layer
        filters = hyperparams[f"Conv1D_{i}_Filters"]
        # Size of the convolution window
        kernel_size = hyperparams[f"Conv1D_{i}_Kernel"]
        # Activation function: relu, tanh, etc.
        activation = hyperparams[f"Conv1D_{i}_Activation"]
        # Type of regularization: none, l1, l2, or l1_l2
        reg_strategy = hyperparams[f"Conv1D_{i}_Regularization"]
        # Weight constraints: none, max_norm, unit_norm, or min_max_norm
        constraint_type = hyperparams[f"Conv1D_{i}_Constraint"]
        # Dropout rate for regularization (0-1)
        dropout_rate = hyperparams[f"Conv1D_{i}_Dropout"]
        # Type of pooling layer: max or average
        pooling_type = hyperparams[f"Conv1D_{i}_Pooling"]
        # Whether to use batch normalization after convolution
        batch_norm = hyperparams[f"Conv1D_{i}_BatchNorm"]

        # Regularization
        regularizer = None
        if reg_strategy != "none":
            # Strength of the regularization penalty (default: 1e-4)
            reg_strength = hyperparams.get(f"Conv1D_{i}_RegStrength", 1e-4)
            if reg_strategy == "l1":
                regularizer = keras.regularizers.L1(reg_strength)
            elif reg_strategy == "l2":
                regularizer = keras.regularizers.L2(reg_strength)
            elif reg_strategy == "l1_l2":
                regularizer = keras.regularizers.L1L2(l1=reg_strength, l2=reg_strength)

        # Constraints
        constraint = None
        if constraint_type != "none":
            # Maximum allowed norm of the weights (default: 2.0)
            constraint_value = hyperparams.get(f"Conv1D_{i}_ConstraintVal", 2.0)
            if constraint_type == "max_norm":
                constraint = keras.constraints.MaxNorm(max_value=constraint_value)
            elif constraint_type == "unit_norm":
                constraint = keras.constraints.UnitNorm()
            elif constraint_type == "min_max_norm":
                constraint = keras.constraints.MinMaxNorm(
                    min_value=0.0, max_value=constraint_value
                )

        # Convolution layer (activation is applied separately)
        x = keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularizer,
            kernel_constraint=constraint,
        )(x)
        # Activation layer
        x = keras.layers.Activation(activation)(x)

        # Optional Batch Normalization
        if batch_norm:
            x = keras.layers.BatchNormalization()(x)

        # Pooling Layer
        if pooling_type == "max":
            x = keras.layers.MaxPooling1D(pool_size=2)(x)
        elif pooling_type == "average":
            x = keras.layers.AveragePooling1D(pool_size=2)(x)

        # Dropout Layer
        x = keras.layers.Dropout(dropout_rate)(x)

    # Flatten before Dense layers
    x = keras.layers.Flatten()(x)

    # Dense layer
    # Number of neurons in the dense layer
    dense_units = hyperparams["Dense_Units"]
    # Activation function for the dense layer
    dense_activation = hyperparams["Dense_Activation"]
    # Regularization strategy for dense layer: none, l1, l2, or l1_l2
    dense_reg_strategy = hyperparams["Dense_Regularization"]
    # Weight constraints for dense layer: none, max_norm, or min_max_norm
    dense_constraint_type = hyperparams["Dense_Constraint"]
    # Whether to use batch normalization after dense layer
    dense_batch_norm = hyperparams["Dense_BatchNorm"]
    # Dropout rate for dense layer (0-1)
    dense_dropout_rate = hyperparams["Dense_Dropout"]

    # Dense Regularization
    dense_regularizer = None
    if dense_reg_strategy != "none":
        # Strength of the dense layer regularization penalty (default: 1e-4)
        dense_reg_strength = hyperparams.get("Dense_RegStrength", 1e-4)
        if dense_reg_strategy == "l1":
            dense_regularizer = keras.regularizers.L1(dense_reg_strength)
        elif dense_reg_strategy == "l2":
            dense_regularizer = keras.regularizers.L2(dense_reg_strength)
        elif dense_reg_strategy == "l1_l2":
            dense_regularizer = keras.regularizers.L1L2(
                l1=dense_reg_strength, l2=dense_reg_strength
            )

    # Dense Constraint
    dense_constraint = None
    if dense_constraint_type != "none":
        # Maximum allowed norm of the dense layer weights (default: 1.5)
        dense_constraint_value = hyperparams.get("Dense_ConstraintVal", 1.5)
        if dense_constraint_type == "max_norm":
            dense_constraint = keras.constraints.MaxNorm(
                max_value=dense_constraint_value
            )
        elif dense_constraint_type == "min_max_norm":
            dense_constraint = keras.constraints.MinMaxNorm(
                min_value=0.0, max_value=dense_constraint_value
            )

    x = keras.layers.Dense(
        dense_units,
        kernel_regularizer=dense_regularizer,
        kernel_constraint=dense_constraint,
    )(x)
    x = keras.layers.Activation(dense_activation)(x)

    if dense_batch_norm:
        x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dropout(dense_dropout_rate)(x)

    # Output layer
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Optimizer with Gradient Clipping
    # Type of gradient clipping: none, value, norm, or global_norm
    clip_type = hyperparams.get("GradientClipType", "none")
    # Maximum gradient value/norm for clipping
    clip_value = hyperparams.get("GradientClipValue", 1.0)
    clip_config = {}
    if clip_type == "value":
        clip_config["clipvalue"] = clip_value
    elif clip_type == "norm":
        clip_config["clipnorm"] = clip_value
    elif clip_type == "global_norm":
        clip_config["global_clipnorm"] = clip_value

    # Learning rate for the Adam optimizer
    learning_rate = hyperparams["learning_rate"]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, **clip_config)

    # Choose loss function based on sample label shape
    if sample_label.shape[-1] > 1:
        loss_function = "categorical_crossentropy"
    else:
        loss_function = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[
            "accuracy",
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.AUC(multi_label=False),
        ],
    )

    return model


def save_model_description(
    output_path, history, num_features, num_classes, train_size, val_size
):
    """
    Generates and saves a comprehensive model training report with performance metrics.

    Creates a detailed description of the trained model, including training configuration,
    dataset statistics, and final performance metrics. This documentation is crucial for:
    - Model versioning and tracking
    - Performance comparison between iterations
    - Experiment documentation
    - Reproducibility verification

    The report includes:
    1. Model Identification
       - Version number (auto-incremented)
       - Creation timestamp
       - Model architecture summary

    2. Dataset Statistics
       - Number of input features
       - Number of output classes
       - Training set size
       - Validation set size

    3. Training Configuration
       - Data augmentation settings
       - Augmentation intensity parameters
       - Other relevant training parameters

    4. Performance Metrics
       - Training and validation loss
       - Accuracy metrics (multiple types)
       - Additional evaluation metrics

    Args:
        output_path (str): Directory path for saving the description file
        history (keras.callbacks.History): Training history containing all metrics
        num_features (int): Number of input features (landmark coordinates)
        num_classes (int): Number of output classes (facial expressions)
        train_size (int): Number of training samples processed
        val_size (int): Number of validation samples evaluated

    Output Files:
        - model_description.txt: Complete model documentation and metrics
    """
    # Extract version and timestamp from directory name
    dir_name = os.path.basename(output_path)  # e.g., "model_v1"
    version = int(dir_name.split("_v")[-1])  # Extracts the numeric part after "_v"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Current datetime as string

    # Save model description
    print("Saving model description")
    description = f"""Model Summary:
    - Version: v{version}
    - Created: {timestamp}
    - Number of features: {num_features}
    - Number of classes: {num_classes}
    - Training samples: {train_size}
    - Validation samples: {val_size}
    - Augmentation applied: {config.SHOULD_AUGMENT}
    - Augmentation intensity: {config.AUGMENTATION_INTENSITY}
    - Final training loss: {history.history["loss"][-1]:.4f}
    - Final training accuracy: {history.history["accuracy"][-1]:.4f}
    - Final training categorical_accuracy: {history.history["categorical_accuracy"][-1]:.4f}
    - Final validation loss: {history.history["val_loss"][-1]:.4f}
    - Final validation accuracy: {history.history["val_accuracy"][-1]:.4f}
    - Final validation categorical_accuracy: {history.history["val_categorical_accuracy"][-1]:.4f}

    """
    with open(os.path.join(output_path, "model_description.txt"), "w") as f:
        f.write(description)

    print("Model artifacts saved in", output_path)


def train_and_save_model(output_dir, hyperparams):
    """
    Executes complete model training pipeline with comprehensive versioning and artifact management.

    This function orchestrates the entire training process from data loading through model
    saving, implementing a robust versioning system for tracking experiments. It handles:

    Workflow Stages:
    1. Data Preparation
       - Loads and validates training/validation datasets
       - Performs input shape validation
       - Verifies dataset integrity and batch sizes

    2. Model Creation
       - Constructs model architecture from hyperparameters
       - Validates model configuration
       - Implements training optimizations

    3. Training Execution
       - Configures training callbacks
       - Manages training epochs
       - Monitors training progress

    4. Artifact Management
       - Implements automatic versioning (model_v1, model_v2, etc.)
       - Creates organized directory structure
       - Saves model checkpoints

    5. Documentation
       - Saves hyperparameter configurations
       - Generates performance reports
       - Records training metrics

    Args:
        output_dir (str): Base directory for storing all model artifacts
        hyperparams (dict): Pre-tuned hyperparameters from optimization, including:
                           - Model architecture parameters
                           - Training configuration
                           - Regularization settings

    Directory Structure:
        output_dir/
        ├── model_v1/
        │   ├── model.keras (trained weights)
        │   ├── hyperparameters.json (configuration)
        │   └── model_description.txt (performance report)
        └── model_v2/
            └── ...

    Raises:
        ValueError: If dataset validation fails or empty batches are detected

    Note:
        This function ensures reproducibility by saving all necessary artifacts
        and configurations needed to recreate the training process.
    """
    # Load datasets
    train_dataset, val_dataset = get_datasets()
    if train_dataset is None or val_dataset is None:
        print("Error: Datasets are empty")
        return

    for i, batch in enumerate(train_dataset):
        if not batch or len(batch[0]) == 0:
            raise ValueError(f"⚠️ Empty batch detected in train_dataset at index {i}!")
        print(f"Train dataset - Batch {i}: Size = {len(batch[0])}")

    sample_train = next(iter(train_dataset))
    input_shape = sample_train[0].shape[1:]

    # Use a sample label batch to determine loss type
    _, sample_label = next(iter(train_dataset))
    num_classes = sample_label.shape[-1]

    # Validate input landmarks (if applicable)
    validate_landmark_count(input_shape, config)

    # Create and compile model
    model = create_model_from_hyperparams(
        input_shape, num_classes, hyperparams, sample_label
    )

    # Define callbacks (excluding TensorBoard)
    callbacks = [
        cb
        for cb in get_callbacks()
        if not isinstance(cb, tf.keras.callbacks.TensorBoard)
    ]

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=hyperparams.get("tuner/epochs", 50),
        # epochs=50,
        callbacks=callbacks,
        verbose=1,
    )

    # Create versioned model directory
    os.makedirs(output_dir, exist_ok=True)
    version = len([d for d in os.listdir(output_dir) if d.startswith("model_v")]) + 1
    model_dir = os.path.join(output_dir, f"model_v{version}")
    os.makedirs(model_dir, exist_ok=True)

    # Save model and hyperparameters
    model_path = os.path.join(model_dir, "model.keras")
    model.save(model_path)

    with open(os.path.join(model_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    num_features, num_classes, total_train_samples, total_val_samples = (
        get_dataset_details(train_dataset, val_dataset)
    )

    # Save model details
    save_model_description(
        model_dir,
        history,
        num_features,
        num_classes,
        total_train_samples,
        total_val_samples,
    )

    print(f"Model saved at: {model_path}")
    print(f"Hyperparameters saved at: {model_dir}/hyperparameters.json")


if __name__ == "__main__":
    # Paths for hyperparameter loading and model saving
    input_dir = "output/model/hyperband_trained/model_v3/best_hyperparameters.json"
    output_dir = "output/model/hyperparam_defined_imp"

    # Load pre-tuned hyperparameters from Hyperband optimization
    with open(input_dir, "r") as f:
        best_hyperparams = json.load(f)

    # Execute training pipeline with versioning
    train_and_save_model(output_dir, best_hyperparams)
