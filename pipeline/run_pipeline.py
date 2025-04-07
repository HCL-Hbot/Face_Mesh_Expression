"""
Pipeline Orchestration System

Manages execution of ML pipeline components including:
- Parallel process isolation with subprocess management
- Real-time output streaming for monitoring
- Robust error handling and reporting
- Dynamic configuration state management
- Flexible execution paths for training scenarios

Environment Requirements:
- Python 3.8+
- subprocess support
- Compatible config module
"""

import os
import subprocess
import sys

from config import Config


def run_script(script_name):
    """
    Executes a Python script in an isolated process with real-time output
    streaming.

    Args:
        script_name (str): Path to the Python script to execute

    Behavior:
        - Uses current Python interpreter for consistency
        - Streams stdout in real-time for monitoring
        - Captures and displays stderr on failure
        - Terminates pipeline on any script failure

    Raises:
        SystemExit: If script execution fails (exit code != 0)
    """
    # Use same Python interpreter as parent process for environment consistency
    python_executable = sys.executable
    print(f"Running {script_name} with Python: {python_executable}\n")

    # Configure subprocess with robust error handling and output streaming
    process = subprocess.Popen(
        [python_executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Prevent UnicodeDecodeError on non-ASCII output
        encoding="utf-8",
        # Handle unrecognized characters gracefully
        errors="replace",
    )

    # Stream stdout in real-time
    for line in process.stdout:
        print(line, end="")  # Print output in real-time

    process.wait()  # Wait for the process to finish

    if process.returncode == 0:
        print(f"\n{script_name} executed successfully.\n")
    else:
        print(f"\nError executing {script_name}!\n")

        # Stream stderr output
        for line in process.stderr:
            print(line, end="")

        exit(1)  # Stop execution if an error occurs


if __name__ == "__main__":
    # Extend PYTHONPATH to include pipeline modules for script imports
    pipeline_paths = [os.getcwd(), os.path.join(os.getcwd(), "pipeline")]
    os.environ["PYTHONPATH"] = os.pathsep.join(pipeline_paths)

    # Training mode configuration flag
    # Set to True for image dataset training pipeline
    train_on_image_set = False

    # Define execution pipeline based on training mode
    if train_on_image_set:
        scripts = [
            # Initial image preprocessing
            "pipeline/pipeline_temp/imagePrep.py",
            # Model training with XLA
            "pipeline/model_train_hyperband_XLA_acceleration_caching.py",
            # Hyperparameter optimization
            "pipeline/model_train_hyperparameters.py",
            # Model visualization
            "pipeline/graph_model.py",
        ]
    else:
        scripts = [
            # Uncomment to capture new facemesh data
            # "pipeline/capture_facemesh_data.py"
            # Model training with XLA
            "pipeline/model_train_hyperband_XLA_acceleration_caching.py",
            # Hyperparameter optimization
            "pipeline/model_train_hyperparameters.py",
            # Model visualization
            "pipeline/graph_model.py",
        ]

    for script in scripts:
        run_script(script)

        # Enable data augmentation after initial model training
        # Note: Using pipeline_temp path for legacy compatibility
        model_train_path = (
            "pipeline/pipeline_temp/model_train_hyperband_XLA_acceleration_caching.py"
        )
        if script == model_train_path:
            Config.AUGMENTATION = True
            print("Config.AUGMENTATION set to True")
