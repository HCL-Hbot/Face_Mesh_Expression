"""
Utility functions for TensorBoard integration in the FaceMesh project.

This module provides functionality to set up and manage TensorBoard for visualizing
training metrics, model graphs, and other deep learning related data. It handles
callback creation, log directory management, and TensorBoard server deployment.

Typical usage:
    ```python
    # Create a TensorBoard callback for model training
    tb_callback = create_tensorboard_callback()
    model.fit(..., callbacks=[tb_callback])

    # Create specific log directory for a model
    log_dir = create_log_dir("/path/to/model")

    # Start TensorBoard server in background
    thread = start_tensorboard()
    ```
"""

import os
import threading
import webbrowser
from datetime import datetime

import keras
from config import config
from tensorboard import program


def create_tensorboard_callback() -> keras.callbacks.TensorBoard:
    """
    Create a TensorBoard callback with project-specific configuration.

    Creates a TensorBoard callback instance configured with settings from the
    project's config file, including log directory, update frequency, and
    visualization options.

    Returns:
        keras.callbacks.TensorBoard: Configured TensorBoard callback ready for
            use in model training.

    Example:
        ```python
        model = create_model()
        tb_callback = create_tensorboard_callback()
        model.fit(x_train, y_train, callbacks=[tb_callback])
        ```
    """
    return keras.callbacks.TensorBoard(
        log_dir=config.LOGS_DIR,
        histogram_freq=config.TENSORBOARD_HISTOGRAM_FREQ,
        write_graph=config.TENSORBOARD_WRITE_GRAPH,
        write_images=config.TENSORBOARD_WRITE_IMAGES,
        update_freq=config.TENSORBOARD_UPDATE_FREQ,
        profile_batch=config.TENSORBOARD_PROFILE_BATCH,
        embeddings_freq=config.TENSORBOARD_EMBEDDINGS_FREQ,
    )


def create_log_dir(model_dir: str) -> str:
    """
    Create and return a timestamped log directory for TensorBoard.

    Creates a unique log directory within the specified model directory using
    the current timestamp. Ensures the directory exists before returning.

    Args:
        model_dir (str): Base directory path where the log directory will be created.

    Returns:
        str: Full path to the created log directory.

    Example:
        ```python
        log_dir = create_log_dir("/models/facemesh_v1")
        # Creates directory like: /models/facemesh_v1/logs/20250311-094200
        ```
    """
    log_dir = os.path.join(model_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def start_tensorboard() -> threading.Thread:
    """
    Start TensorBoard in a separate thread and open it in the browser.

    Launches a TensorBoard server as a daemon thread using the configured log
    directory and port. Automatically opens the TensorBoard interface in the
    default web browser.

    Returns:
        threading.Thread: The daemon thread running the TensorBoard server.

    Example:
        ```python
        # Start TensorBoard server in background
        thread = start_tensorboard()
        # Continue with other tasks while TensorBoard runs
        ```
    """
    log_dir = os.path.join(config.OUTPUT_DIR, config.LOGS_DIR)

    def run_tensorboard():
        tb = program.TensorBoard()
        tb.configure(
            argv=[
                None,
                "--logdir",
                config.LOGS_DIR,
                "--port",
                str(config.TENSORBOARD_PORT),
            ]
        )
        url = tb.launch()
        print(f"\nTensorBoard started at {url}")
        webbrowser.open(url)

    thread = threading.Thread(target=run_tensorboard, daemon=True)
    thread.start()
    return thread
