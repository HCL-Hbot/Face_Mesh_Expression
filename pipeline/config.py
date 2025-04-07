"""
FaceMesh Configuration Module

This module provides a centralized configuration system for the FaceMesh deep learning application.
It manages all configurable parameters including:
- Model architecture and training parameters
- Data processing and augmentation settings
- File paths and directory structures
- Hyperparameter tuning ranges and settings
- Tensorboard logging configuration
- FaceMesh detection parameters

The configuration is implemented as a singleton class that initializes all necessary
directories and provides consistent access to configuration values across the application.

Usage:
    from pipeline.config import config

    # Access configuration parameters
    batch_size = config.BATCH_SIZE
    model_path = config.TRAIN_MODEL_PATH
"""

import datetime
import os

import keras
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


class Config:
    """
    Centralized configuration class for the FaceMesh deep learning application.

    This class manages all configuration parameters and ensures consistent settings
    across the application. It automatically creates required directories and
    provides easy access to file paths and model parameters.
    """

    def __init__(self):
        """
        Initialize configuration with default values for all parameters.
        Creates necessary directories and sets up paths for model artifacts.
        """
        # General configuration
        # Verbose level for training outputs (0=silent, 1=progress bar, 2=one line per epoch)
        self.VERBOSE = 1
        # Timestamp for unique run identification
        self.TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Data Processing Parameters
        # Input image dimensions for preprocessing
        self.IMAGE_DIMS = (240, 320)
        # Fraction of data to use for validation
        self.VALIDATION_SPLIT = 0.2
        # Seed for reproducible results
        self.RANDOM_STATE = 42

        # Data Augmentation Settings
        # Toggle data augmentation during training
        self.SHOULD_AUGMENT = True
        # Controls the magnitude of augmentation transformations
        self.AUGMENTATION_INTENSITY = 0.10

        # Model Architecture Parameters
        # Number of emotion classes to predict
        self.NUM_CLASSES = 4
        # Input shape for the neural network (number of facial landmarks * 3 coordinates)
        self.INPUT_SHAPE = (216,)
        # Facial features to track with MediaPipe
        self.FACIAL_FEATURES = [
            mp_face_mesh.FACEMESH_LIPS,
            mp_face_mesh.FACEMESH_LEFT_EYEBROW,
            mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
            mp_face_mesh.FACEMESH_LEFT_EYE,
            mp_face_mesh.FACEMESH_RIGHT_EYE,
        ]

        # Training Parameters
        self.EPOCHS = 50  # Maximum number of training epochs
        self.BATCH_SIZE = 32  # Number of samples per training batch
        self.SHOULD_SHUFFLE = True  # Shuffle training data between epochs
        self.SHOULD_BATCH = True  # Use batching during training
        self.MAX_TRIALS = 1000  # Maximum hyperparameter optimization trials

        # Model Checkpoint Configuration
        self.SAVE_ONLY_LAST_EPOCH = True  # Only save model at end of training

        # Tensorboard Logging Configuration
        self.TENSORBOARD_PORT = 6006  # Port for Tensorboard server
        self.TENSORBOARD_UPDATE_FREQ = "epoch"  # Logging frequency
        self.TENSORBOARD_HISTOGRAM_FREQ = 1  # Epochs between histogram updates
        self.TENSORBOARD_WRITE_GRAPH = True  # Log model graph
        self.TENSORBOARD_WRITE_IMAGES = False  # Log model weights as images
        self.TENSORBOARD_PROFILE_BATCH = 2  # Batch to profile for performance
        self.TENSORBOARD_EMBEDDINGS_FREQ = 1  # Epochs between embedding updates

        # Directory Structure Configuration
        # Base project directory paths
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.OUTPUT_DIRECTORY = os.path.join(self.PROJECT_ROOT, "output")
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, "dataset")

        # Input data directory
        self.IMAGE_DIR = os.path.join(self.DATA_DIR)

        # Output directory structure for artifacts
        self.OUTPUT_DIR = self.OUTPUT_DIRECTORY
        self.MODEL_DIR = os.path.join(self.OUTPUT_DIR, "model")
        self.HISTORY_DIR = os.path.join(self.OUTPUT_DIR, "history")
        self.LOGS_DIR = os.path.join(self.OUTPUT_DIR, "logs")
        self.LANDMARKS_DIR = os.path.join(self.OUTPUT_DIR, "landmarks")

        # File paths for model artifacts
        self.TRAIN_MODEL_PATH = os.path.join(self.MODEL_DIR, "model.keras")
        self.MODEL_BEST_PATH = os.path.join(self.MODEL_DIR, "model_best.keras")
        self.MODEL_CHECKPOINT_PATH = os.path.join(
            self.MODEL_DIR, "model_checkpoint.keras"
        )
        self.HYPERPARAMS_PATH = os.path.join(self.MODEL_DIR, "hyperparameters.txt")
        self.TRAIN_HISTORY_PATH = os.path.join(
            self.HISTORY_DIR, "training_history.json"
        )
        self.CSV_LOG_PATH = os.path.join(self.HISTORY_DIR, "log.csv")
        self.LANDMARKS_PATH = os.path.join(self.LANDMARKS_DIR, "landmarks.csv")

        # MediaPipe FaceMesh Configuration
        self.FACE_MESH_STATIC_MODE = False  # Process frames independently
        self.FACE_MESH_MIN_DETECTION_CONFIDENCE = (
            0.1  # Minimum confidence for detection
        )
        self.FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.1  # Minimum confidence for tracking

        # Hyperparameter Tuning Configuration
        # Keras Tuner settings
        self.HP_TUNER_MAX_TRIALS = 1000  # Maximum number of trials
        self.HP_TUNER_OBJECTIVE = "val_loss"  # Metric to optimize
        self.HP_TUNER_DIRECTORY = self.OUTPUT_DIR  # Directory for tuning results
        self.HP_TUNER_EPOCHS_MAX = 50  # Maximum epochs per trial

        # Network architecture search space
        self.HP_TUNER_LAYERS_MIN = 1
        self.HP_TUNER_LAYERS_MAX = 1
        self.HP_TUNER_LAYERS_STEP = 1

        self.HP_TUNER_UNITS_MIN = 32
        self.HP_TUNER_UNITS_MAX = 256
        self.HP_TUNER_UNITS_STEP = 32

        # Model hyperparameters search space
        self.HP_TUNER_ACTIVATIONS = ["relu", "tanh", "sigmoid", "leaky_relu"]
        self.HP_TUNER_OPTIMIZERS = ["adam", "sgd", "rmsprop", "adamw"]
        self.HP_TUNER_LEARNING_RATES = [0.001, 0.01, 0.1]

        # Regularization search space
        self.HP_TUNER_DROPOUT_MIN = 0.1
        self.HP_TUNER_DROPOUT_MAX = 0.5
        self.HP_TUNER_DROPOUT_STEP = 0.1

        self.HP_TUNER_L2_MIN = 0.0001
        self.HP_TUNER_L2_MAX = 0.01
        self.HP_TUNER_L2_STEP = 0.0001

        self.HP_TUNER_L1_MIN = 0.0001
        self.HP_TUNER_L1_MAX = 0.01
        self.HP_TUNER_L1_STEP = 0.0001

        # Weight constraint configuration
        self.HP_TUNER_CONSTRAINT_MIN = 0.1
        self.HP_TUNER_CONSTRAINT_MAX = 3.0
        self.HP_TUNER_CONSTRAINT_STEP = 0.1
        self.HP_TUNER_CONSTRAINT_TYPES = [
            "none",
            "max_norm",
            "unit_norm",
            "min_max_norm",
        ]

        # Gradient clipping configuration
        self.HP_TUNER_CLIP_MIN = 0.1
        self.HP_TUNER_CLIP_MAX = 5.0
        self.HP_TUNER_CLIP_STEP = 0.1
        self.HP_TUNER_CLIP_TYPES = ["none", "value", "norm", "global_norm"]

        # Training configuration search space
        self.HP_TUNER_BATCH_SIZE_MIN = 16
        self.HP_TUNER_BATCH_SIZE_MAX = 64
        self.HP_TUNER_BATCH_SIZE_STEP = 16

        self.HP_TUNER_LOSS_FUNCTION = "categorical_crossentropy"
        self.HP_TUNER_TRIAL_MAX_RETRY = 3

        # Initialize directory structure
        self._create_directories()

    def _create_directories(self):
        """
        Create necessary directory structure for model artifacts and logs.

        Creates output directories if they don't exist:
        - Model directory for saved models
        - History directory for training logs
        - Logs directory for Tensorboard
        - Landmarks directory for facial landmark data
        """
        directories = [
            self.OUTPUT_DIR,
            self.MODEL_DIR,
            self.HISTORY_DIR,
            self.LOGS_DIR,
            self.LANDMARKS_DIR,
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def set_random_seed(self, seed):
        """
        Set global random seed for reproducible results.

        Args:
            seed: Integer seed value for random number generators

        Note:
            This affects both Python's random module and Keras/TensorFlow randomization
        """
        seed = self.RANDOM_STATE
        keras.utils.set_random_seed(seed)


# Create singleton config instance
config = Config()
