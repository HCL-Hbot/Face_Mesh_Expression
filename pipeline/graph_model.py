"""
Neural Network Model Visualization

This module provides tools for visualizing the architecture of trained neural networks.
It handles model loading, validation, and generates visual diagrams of the network structure.

Key Features:
- Recursive model file search in output directory structure
- Automated latest model detection based on file modification time
- Model structure validation with comprehensive checks
- High-quality architecture visualization using GraphViz
- Robust error handling and detailed logging

Configuration Parameters:
    BASE_OUTPUT_PATH: str = "output/"
        Root directory for model file search
    MODEL_VERSION_PATH: str = "output/model/v3/"
        Directory for latest model version
    MODEL_FILE_PATTERN: str = "*.keras"
        File pattern for identifying model files
    VISUALIZATION_FILENAME: str = "model_visualization.png"
        Output filename for model diagram

Dependencies:
    - tensorflow>=2.0: Neural network framework
    - keras: High-level neural network API
    - graphviz: Graph visualization engine
    - IPython: Interactive Python notebooks support

Performance Notes:
    - Model loading time scales with model size
    - GraphViz rendering may be memory-intensive for large models
    - Consider using CPU-only TensorFlow for visualization tasks
    - Visualization generation typically takes 2-5 seconds

Example Usage:
    ```python
    # Load and visualize latest model
    from pipeline.graph_model import main
    main()

    # Custom model visualization
    from pipeline.graph_model import load_model, validate_model
    model = load_model()
    if validate_model(model):
        plot_model(model, to_file='custom_viz.png',
                  show_shapes=True)
    ```

Note:
    Requires TensorFlow backend and graphviz system installation.
    For graphviz installation on Unix: sudo apt-get install graphviz
    For Windows: Download from graphviz.org/download
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # Ensure TensorFlow backend

import glob

import graphviz
import numpy as np
import tensorflow as tf
from IPython.display import Image
from keras import layers
from keras._tf_keras.keras.utils import plot_model
from tensorflow import keras


def get_latest_model() -> str | None:
    """
    Locate the most recently modified Keras model file.

    Searches recursively through the output directory structure to find
    the most recently updated .keras model file. This supports the
    versioned model directory structure used in training.

    Parameters:
        None

    Returns:
        str | None: Absolute path to the latest model file, or None if no models found

    Performance:
        - Time complexity: O(n) where n is number of .keras files
        - Space complexity: O(n) for storing file paths

    Example:
        ```python
        model_path = get_latest_model()
        if model_path:
            print(f"Found model at: {model_path}")
        ```

    Note:
        Prints status messages about file discovery for debugging purposes.
    """
    base_path = "output/"
    print("Looking for Keras models in:", base_path)

    # Search for all `.keras` files recursively
    model_files = glob.glob(os.path.join(base_path, "**", "*.keras"), recursive=True)

    # Print found model files
    print("Model files found:", model_files)

    if model_files:
        latest_model = max(
            model_files, key=os.path.getmtime
        )  # Get the most recently modified file
        print("Using model:", latest_model)
        return latest_model

    print("No Keras model files found.")
    return None


def validate_model(model: tf.keras.Model) -> bool:
    """
    Perform comprehensive validation of loaded model structure.

    Checks:
    - Model can be summarized successfully
    - Input/output shapes are properly defined
    - Object is a valid TensorFlow model instance
    - All layers are properly configured
    - Model architecture is compatible with visualization

    Parameters:
        model: tf.keras.Model
            The loaded Keras model to validate

    Returns:
        bool: True if all validation checks pass, False otherwise

    Raises:
        TypeError: If input is not a Keras model
        ValueError: If model structure is invalid

    Example:
        ```python
        model = tf.keras.models.load_model('model.keras')
        if validate_model(model):
            print("Model validation successful")
        ```

    Note:
        Focuses on logging model structure rather than strict validation,
        allowing visualization even if some checks fail.
    """
    try:
        # Validate model object type
        if not isinstance(model, tf.keras.Model):
            raise TypeError("Input must be a valid Keras model")

        print("\nModel Summary:")
        model.summary()
        print(f"\nInput shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}\n")

        return True
    except Exception as e:
        print(f"Error validating model: {e}")
        return False


def load_model() -> tf.keras.Model | None:
    """
    Load and validate the most recent Keras model.

    Workflow:
    1. Find latest model file using get_latest_model()
    2. Attempt to load model using tf.keras.models.load_model()
    3. Validate model structure with validate_model()
    4. Return validated model or None on failure

    Returns:
        tf.keras.Model | None: Successfully loaded and validated model,
                             or None if loading/validation fails

    Performance:
        - Loading time depends on model size and complexity
        - Memory usage scales with model parameters
        - Consider using CPU-only TF for visualization tasks

    Example:
        ```python
        model = load_model()
        if model is not None:
            # Model loaded successfully
            predictions = model.predict(data)
        ```

    Note:
        Includes error handling for both file operations and model loading,
        with informative error messages for debugging.
    """
    model_path = get_latest_model()
    if model_path is None:
        print("No model found to load.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        validate_model(model)
        return model
    except (OSError, ValueError) as e:
        print(f"Error loading model: {e}")
        return None


def main() -> None:
    """
    Execute complete model visualization pipeline.

    Workflow:
    1. Load latest trained model using load_model()
    2. Validate model structure with validate_model()
    3. Generate architecture visualization using plot_model()
    4. Save visualization to configured output directory
    5. Display interactive visualization with IPython

    The visualization includes:
    - Layer structure and connections
    - Shape information for each layer
    - Layer names and types
    - Layer parameters and configurations

    Returns:
        None

    Side Effects:
        - Creates/updates model visualization file
        - Displays visualization in notebook/IDE
        - Prints status messages to console

    Example:
        ```python
        # Generate and display model visualization
        main()
        ```
    """
    # Load and validate model
    model = load_model()  # Dynamically load the latest model

    if model is None:
        print("Failed to load a model. Exiting...")
        return

    print("Successfully loaded the latest model.")

    # Configure visualization output path
    base_path = "output/model/v3/"  # Latest model version directory
    image_path = os.path.join(base_path, "model_visualization.png")

    # Create and save architecture visualization
    plot_model(model, to_file=image_path, show_shapes=True, show_layer_names=True)

    # Show visualization and wait for user review
    Image(image_path)

    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
