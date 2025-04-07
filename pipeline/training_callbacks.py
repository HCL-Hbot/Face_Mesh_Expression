"""
Training Callbacks Configuration

This module provides a centralized configuration for Keras training callbacks,
implementing best practices for model training monitoring and optimization.

Key Features:
- Model checkpointing with configurable save strategy
- Early stopping to prevent overfitting
- TensorBoard integration for training visualization
- Learning rate adaptation
- Training metrics logging
- NaN detection and termination

Configuration Parameters (from config.py):
    MODEL_CHECKPOINT_PATH: str
        Path to save model checkpoints
    SAVE_ONLY_LAST_EPOCH: bool
        If True, only saves model from last epoch
    VERBOSE: int
        Logging verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
    LOGS_DIR: str
        Directory for TensorBoard logs
    TENSORBOARD_UPDATE_FREQ: str | int
        'batch', 'epoch', or number of batches
    CSV_LOG_PATH: str
        Path for CSV training logs

Performance Notes:
    - Checkpoint saving frequency affects disk I/O and training time
    - TensorBoard logging with high frequency may impact training performance
    - Early stopping helps optimize training duration
    - Learning rate reduction can help fine-tune model convergence

Example Usage:
    ```python
    from pipeline.training_callbacks import get_callbacks
    from tensorflow import keras

    model = keras.Sequential([...])
    callbacks = get_callbacks()

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        epochs=100
    )
    ```

Dependencies:
    - tensorflow>=2.0: Core ML framework
    - keras: Neural network API
    - tensorboard: Training visualization
"""

import keras
from config import config


def get_callbacks() -> list[keras.callbacks.Callback]:
    """
    Create a comprehensive set of Keras callbacks for model training.

    Returns:
        list[keras.callbacks.Callback]: List of configured callback objects

    Configuration:
        Callback parameters are sourced from config.py, allowing
        centralized management of training behavior.
    """
    callbacks = [
        # 1. Model Checkpointing:
        # - Saves model weights based on validation loss
        # - Configurable save frequency and criteria
        keras.callbacks.ModelCheckpoint(
            config.MODEL_CHECKPOINT_PATH,
            monitor="val_loss",
            verbose=0,
            save_best_only=config.SAVE_ONLY_LAST_EPOCH,
            mode="auto",
            save_freq="epoch",
            initial_value_threshold=0.8,
        ),
        # 2. Early Stopping:
        # - Prevents overfitting by monitoring validation loss
        # - Restores best weights when stopping
        # - Begins monitoring after 10 epochs
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            min_delta=0.01,
            verbose=config.VERBOSE,
            restore_best_weights=True,
            start_from_epoch=10,
        ),
        # 3. TensorBoard Integration:
        # - Real-time training visualization
        # - Performance metrics logging
        # - Training graph visualization
        keras.callbacks.TensorBoard(
            log_dir=config.LOGS_DIR,
            write_steps_per_second=True,
            update_freq=config.TENSORBOARD_UPDATE_FREQ,
        ),
        # 4. Learning Rate Management:
        # - Reduces learning rate on plateau
        # - Helps overcome training stagnation
        # - Configurable reduction factors
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            min_delta=0.1,
            factor=0.1,
            patience=3,
            verbose=config.VERBOSE,
            min_lr=0.000001,
        ),
        # 5. Progress Logging:
        # - CSV logging of training metrics
        # - Supports training analysis and visualization
        keras.callbacks.CSVLogger(
            filename=config.CSV_LOG_PATH, separator=",", append=False
        ),
        # 6. Training Protection:
        # - Terminates training if NaN values detected
        # - Prevents resources waste on failed training
        keras.callbacks.TerminateOnNaN(),
    ]

    return callbacks
