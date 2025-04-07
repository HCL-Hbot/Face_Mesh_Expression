"""
Neural Network Model Training with Hyperband Optimization

This module implements a 1D CNN model training pipeline with advanced features:
- Hyperband optimization for efficient hyperparameter tuning
- XLA (Accelerated Linear Algebra) acceleration for improved CPU performance
- TensorFlow data caching for faster training iterations
- Distributed training strategy for multi-core CPU utilization
- Comprehensive hyperparameter search space including:
  * Network architecture (layers, units, activations)
  * Regularization techniques (dropout, L1/L2)
  * Optimization parameters (learning rate, gradient clipping)
  * Normalization strategies

Dependencies:
- keras: Deep learning model implementation
- keras_tuner: Hyperparameter optimization
- tensorflow: Core ML framework
- data_load: Custom dataset loading utilities
- training_callbacks: Training progress monitoring
- tensorboard_utils: Training visualization
- config: Project configuration parameters

Note: This implementation focuses on facial expression classification
using landmark data processed into 1D sequences.
"""

import json
import os

import keras
import keras_tuner
import tensorflow as tf

from config import config
from data_load import get_datasets
from training_callbacks import get_callbacks

# Enable XLA acceleration for improved CPU operation performance
tf.config.optimizer.set_jit(True)
print("XLA JIT enabled:", tf.config.optimizer.get_jit())


def create_model_tuned(hp, input_shape=(216,), num_classes=5):
    """
    Build a 1D CNN model with comprehensive hyperparameter tuning.

    Constructs a neural network for facial expression classification with:
    - Dynamic layer configuration based on hyperparameters
    - Multiple normalization options (standard, layer, batch)
    - Configurable CNN architecture (1-3 Conv1D layers)
    - Regularization techniques (dropout, L1/L2)
    - Flexible activation functions
    - Gradient clipping options
    - Learning rate optimization

    Args:
        hp (keras_tuner.HyperParameters): Hyperparameter object for tuning
        input_shape (tuple): Shape of input data (default: (216,) for landmark features)
        num_classes (int): Number of output classes (default: 5 facial expressions)

    Returns:
        keras.Model: Compiled model with tuned architecture and parameters

    Note:
        The model architecture is designed for 1D sequence data from facial landmarks,
        with emphasis on preventing overfitting through various regularization techniques.
    """
    # Input layer and reshaping for 1D convolution
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Reshape((input_shape[0], 1))(inputs)  # Convert to sequence format

    # Select input normalization strategy
    normalization = hp.Choice(
        "Normalization Type",
        ["standard", "layer", "batch", "none"],
        default="batch",  # BatchNorm typically works well for CNNs
    )
    if normalization == "standard":
        x = keras.layers.Normalization()(x)
    elif normalization == "layer":
        x = keras.layers.LayerNormalization()(x)
    elif normalization == "batch":
        x = keras.layers.BatchNormalization()(x)

    num_conv_layers = hp.Int(
        "Number of Conv1D Layers", min_value=1, max_value=3, default=2
    )

    for i in range(num_conv_layers):
        filters = hp.Int(
            f"Conv1D_{i + 1}_Filters", min_value=32, max_value=256, step=32, default=64
        )
        kernel_size = hp.Choice(
            f"Conv1D_{i + 1}_Kernel", values=[2, 3, 5, 7], default=3
        )
        activation = hp.Choice(
            f"Conv1D_{i + 1}_Activation",
            ["relu", "leaky_relu", "selu", "tanh", "gelu"],
            default="relu",
        )
        reg_strategy = hp.Choice(
            f"Conv1D_{i + 1}_Regularization",
            ["none", "l1", "l2", "l1_l2"],
            default="none",
        )
        regularizer = None
        if reg_strategy != "none":
            reg_factor = hp.Float(
                f"Conv1D_{i + 1}_RegStrength",
                min_value=1e-5,
                max_value=1e-2,
                sampling="log",
                default=1e-4,
            )
            if reg_strategy == "l1":
                regularizer = keras.regularizers.L1(reg_factor)
            elif reg_strategy == "l2":
                regularizer = keras.regularizers.L2(reg_factor)
            elif reg_strategy == "l1_l2":
                regularizer = keras.regularizers.L1L2(l1=reg_factor, l2=reg_factor)

        constraint_type = hp.Choice(
            f"Conv1D_{i + 1}_Constraint",
            values=config.HP_TUNER_CONSTRAINT_TYPES,
            default="none",
        )
        constraint = None
        if constraint_type != "none":
            constraint_value = hp.Float(
                f"Conv1D_{i + 1}_ConstraintVal",
                min_value=config.HP_TUNER_CONSTRAINT_MIN,
                max_value=config.HP_TUNER_CONSTRAINT_MAX,
                step=config.HP_TUNER_CONSTRAINT_STEP,
            )
            if constraint_type == "max_norm":
                constraint = keras.constraints.MaxNorm(max_value=constraint_value)
            elif constraint_type == "unit_norm":
                constraint = keras.constraints.UnitNorm()
            elif constraint_type == "min_max_norm":
                constraint = keras.constraints.MinMaxNorm(
                    min_value=0.0, max_value=constraint_value
                )

        x = keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularizer,
            kernel_constraint=constraint,
        )(x)
        x = keras.layers.Activation(activation)(x)
        if hp.Boolean(f"Conv1D_{i + 1}_BatchNorm"):
            x = keras.layers.BatchNormalization()(x)
        pooling = hp.Choice(
            f"Conv1D_{i + 1}_Pooling", ["max", "average"], default="max"
        )
        if pooling == "max":
            x = keras.layers.MaxPooling1D(pool_size=2)(x)
        else:
            x = keras.layers.AveragePooling1D(pool_size=2)(x)
        dropout_rate = hp.Float(
            f"Conv1D_{i + 1}_Dropout",
            min_value=0.0,
            max_value=0.5,
            step=0.1,
            default=0.2,
        )
        x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Flatten()(x)

    units = hp.Int("Dense_Units", min_value=16, max_value=128, step=32, default=64)
    dense_activation = hp.Choice(
        "Dense_Activation", ["relu", "selu", "leaky_relu"], default="relu"
    )
    dense_reg = hp.Choice(
        "Dense_Regularization", ["none", "l1", "l2", "l1_l2"], default="none"
    )
    if dense_reg != "none":
        reg_strength = hp.Float(
            "Dense_RegStrength",
            min_value=1e-5,
            max_value=1e-2,
            sampling="log",
            default=1e-4,
        )
        if dense_reg == "l1":
            dense_regularizer = keras.regularizers.L1(reg_strength)
        elif dense_reg == "l2":
            dense_regularizer = keras.regularizers.L2(reg_strength)
        else:
            dense_regularizer = keras.regularizers.L1L2(
                l1=reg_strength, l2=reg_strength
            )
    else:
        dense_regularizer = None

    dense_constraint_type = hp.Choice(
        "Dense_Constraint", values=config.HP_TUNER_CONSTRAINT_TYPES, default="none"
    )
    dense_constraint = None
    if dense_constraint_type != "none":
        dense_constraint_value = hp.Float(
            "Dense_ConstraintVal",
            min_value=config.HP_TUNER_CONSTRAINT_MIN,
            max_value=config.HP_TUNER_CONSTRAINT_MAX,
            step=config.HP_TUNER_CONSTRAINT_STEP,
        )
        if dense_constraint_type == "max_norm":
            dense_constraint = keras.constraints.MaxNorm(
                max_value=dense_constraint_value
            )
        elif dense_constraint_type == "unit_norm":
            dense_constraint = keras.constraints.UnitNorm()
        elif dense_constraint_type == "min_max_norm":
            dense_constraint = keras.constraints.MinMaxNorm(
                min_value=0.0, max_value=dense_constraint_value
            )

    x = keras.layers.Dense(
        units, kernel_regularizer=dense_regularizer, kernel_constraint=dense_constraint
    )(x)
    x = keras.layers.Activation(dense_activation)(x)
    if hp.Boolean("Dense_BatchNorm"):
        x = keras.layers.BatchNormalization()(x)
    dropout_rate = hp.Float(
        "Dense_Dropout", min_value=0.0, max_value=0.5, step=0.1, default=0.2
    )
    x = keras.layers.Dropout(dropout_rate)(x)

    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-1, sampling="log", default=1e-3
    )
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    clip_type = hp.Choice(
        "GradientClipType", values=config.HP_TUNER_CLIP_TYPES, default="none"
    )
    clip_config = {}
    if clip_type != "none":
        clip_value = hp.Float(
            "GradientClipValue",
            min_value=config.HP_TUNER_CLIP_MIN,
            max_value=config.HP_TUNER_CLIP_MAX,
            step=config.HP_TUNER_CLIP_STEP,
        )
        if clip_type == "value":
            clip_config["clipvalue"] = clip_value
        elif clip_type == "norm":
            clip_config["clipnorm"] = clip_value
        elif clip_type == "global_norm":
            clip_config["global_clipnorm"] = clip_value

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, **clip_config)

    model.compile(
        optimizer=optimizer,
        loss=config.HP_TUNER_LOSS_FUNCTION,
        # metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()]
        metrics=[
            "accuracy",  # Overall correctness
            keras.metrics.CategoricalAccuracy(),  # More robust for multi-class classification
            keras.metrics.Precision(),  # Positive predictive value per class
            keras.metrics.Recall(),  # Sensitivity per class
            keras.metrics.AUC(
                multi_label=False
            ),  # Measures how well the model distinguishes classes
        ],
    )

    return model


def main():
    """
    Execute the complete model training pipeline.

    Workflow:
    1. Load and prepare datasets with caching
    2. Configure distributed training strategy
    3. Initialize Hyperband tuner
    4. Execute hyperparameter search
    5. Save best model and hyperparameters

    The function implements performance optimizations:
    - Dataset caching and prefetching
    - Multi-core CPU utilization
    - XLA acceleration
    """
    # Initialize datasets with performance optimizations
    train_dataset, val_dataset = get_datasets()
    if train_dataset is None or val_dataset is None:
        print("Error: Datasets are empty")
        return

    # Apply caching and prefetching for performance
    train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)

    sample_train = next(iter(train_dataset))
    sample_val = next(iter(val_dataset))

    # Log dataset characteristics for debugging and verification
    print("\n=== Dataset Information ===")
    print(f"Train sample shape: {sample_train[0].shape}")
    print(f"Train labels: {sample_train[1]}")
    print(f"Validation sample shape: {sample_val[0].shape}")
    print(f"Validation labels: {sample_val[1]}")

    print("\n=== Sample Analysis ===")
    print(f"Raw sample shape: {sample_train[0].shape}")
    print(f"Sample values (first 10): {sample_train[0][:10]}")
    print(f"Label shape: {sample_train[1].shape}")
    print(f"Label values: {sample_train[1]}")

    input_shape = sample_train[0].shape[1:]
    num_classes = sample_train[1].shape[-1]

    print(f"\nInput shape: {input_shape}")
    print(f"Number of classes: {num_classes}")

    # start_tensorboard()  # Launch TensorBoard for monitoring

    # Configure distributed training for CPU optimization
    strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
    print(f"Number of devices in strategy: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model_builder = lambda hp: create_model_tuned(
            hp, input_shape=input_shape, num_classes=num_classes
        )
        tuner = keras_tuner.Hyperband(
            model_builder,
            objective=config.HP_TUNER_OBJECTIVE,
            max_epochs=config.HP_TUNER_EPOCHS_MAX,
            factor=3,
            directory=config.OUTPUT_DIR,
            project_name="hyperband_tuning",
            overwrite=True,
            seed=config.RANDOM_STATE,
        )
        tuner.search_space_summary()

        # Initialize callbacks, excluding TensorBoard for Hyperband compatibility
        callbacks = [
            cb
            for cb in get_callbacks()
            if not isinstance(cb, tf.keras.callbacks.TensorBoard)
        ]

        # Execute Hyperband optimization
        print("\n=== Starting Hyperband Optimization ===")
        tuner.search(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.HP_TUNER_EPOCHS_MAX,
            callbacks=callbacks,
            verbose=config.VERBOSE,
        )

        print("Hyperparameter search completed. Getting best model...")
        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = model_builder(best_hp)

        print(f"Saving best model to {config.MODEL_BEST_PATH}")
        best_model.save(config.MODEL_BEST_PATH)

        hp_save_path = os.path.join(config.MODEL_DIR, "best_hyperparameters.json")
        print(f"Saving hyperparameters to {hp_save_path}")
        with open(hp_save_path, "w") as f:
            json.dump(best_hp.values, f, indent=2)

    print("Training completed successfully")


if __name__ == "__main__":
    main()
