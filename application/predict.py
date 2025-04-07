"""
Facial Expression Recognition Module

This module provides functionality for real-time facial expression recognition using MediaPipe and TensorFlow.
It processes facial landmarks to predict emotions using a pre-trained deep learning model.

Key Components:
    - MediaPipe Face Mesh for facial landmark detection
    - TensorFlow/Keras for emotion prediction
    - Support for multiple emotion classes
    - Real-time processing capabilities

Dependencies:
    - tensorflow
    - numpy
    - mediapipe
"""

import mediapipe as mp
import numpy as np
import tensorflow as tf

# Define emotion classes supported by the model
# These labels correspond to the output classes of the trained model
class_labels = ["Smile_with_teeth", "Smile", "Neutral", "Frowning"]

# Initialize MediaPipe face mesh solution
mp_face_mesh = mp.solutions.face_mesh

# Configure face mesh detection with optimized parameters for real-time processing
# - static_image_mode=False: Enables faster tracking between frames
# - max_num_faces=1: Optimized for single face detection
# - refine_landmarks=True: Enhanced landmark accuracy
# - min_detection_confidence=0.3: Lower threshold for faster processing
# - min_tracking_confidence=0.1: Lower threshold to maintain tracking
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.1,
)

# Define relevant facial feature groups for emotion detection
# These feature groups contain the most important landmarks for expression recognition
FACIAL_FEATURES = [
    mp_face_mesh.FACEMESH_LIPS,  # Mouth region
    mp_face_mesh.FACEMESH_LEFT_EYEBROW,  # Left eyebrow movement
    mp_face_mesh.FACEMESH_RIGHT_EYEBROW,  # Right eyebrow movement
    mp_face_mesh.FACEMESH_LEFT_EYE,  # Left eye shape
    mp_face_mesh.FACEMESH_RIGHT_EYE,  # Right eye shape
]


def get_facial_feature_indices():
    """
    Extracts unique landmark indices from predefined MediaPipe FaceMesh facial features.

    This function processes the FACIAL_FEATURES groups to create a deduplicated list
    of landmark indices that are relevant for emotion detection.

    Returns:
        list: Sorted list of unique landmark indices relevant to facial expression recognition.

    Note:
        The returned indices correspond to the MediaPipe FaceMesh coordinate system
        and are used to extract specific facial points for emotion prediction.
    """
    mp_face_mesh = mp.solutions.face_mesh

    # Extract unique landmark indices
    landmark_indices = set()
    for feature in FACIAL_FEATURES:
        for connection in feature:
            landmark_indices.update(connection)  # Add both points in each connection

    return sorted(landmark_indices)  # Sort for consistency


# Generate the list of required landmark indices
expected_landmark_indices = get_facial_feature_indices()


def validate_model(model):
    """
    Validates and logs the structure of a loaded Keras model.

    Args:
        model (tf.keras.Model): The loaded Keras model to validate.

    Returns:
        bool: True if validation passes, False otherwise.

    Raises:
        ValueError: If the model is not a valid Keras model.
    """
    try:
        print("\nModel Summary:")
        model.summary()
        print(f"\nInput shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}\n")

        # Minimal check for TensorFlow model type
        if not isinstance(model, tf.keras.Model):
            raise ValueError("Uploaded file is not a valid Keras model")

        return True
    except Exception as e:
        print(f"Error validating model: {e}")
        return False


def load_model(model_path):
    """
    Load a trained Keras model from the specified file path.

    Args:
        model_path (str): Path to the saved Keras model file (.keras format).

    Returns:
        tf.keras.Model: Loaded Keras model if successful, None otherwise.

    Raises:
        OSError: If the model file cannot be accessed.
        ValueError: If the model file is invalid or corrupted.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        validate_model(model)  # Print model info
        return model
    except (OSError, ValueError) as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_landmarks(landmarks, expected_landmark_indices):
    """
    Converts facial landmarks into a format suitable for model input.

    This function handles two types of landmark inputs:
    1. Tuple format: (x, y, z) coordinates
    2. MediaPipe landmark objects with x, y, z attributes

    Args:
        landmarks (list): List of landmarks in either tuple or MediaPipe format.
        expected_landmark_indices (list): Indices of landmarks to extract.

    Returns:
        numpy.ndarray: Processed landmarks as a float32 array with shape (1, n_features).

    Note:
        The output array is structured as [x1,...,xn, y1,...,yn, z1,...,zn] for n landmarks.
    """
    if isinstance(landmarks[0], tuple):
        # Convert tuple list back into a flattened array
        flattened_landmarks = []
        for lm in landmarks:
            flattened_landmarks.extend([lm[0], lm[1], lm[2]])  # x, y, z
    else:
        # Extract from MediaPipe landmark objects
        selected_landmarks = [landmarks[i] for i in expected_landmark_indices]
        flattened_landmarks = (
            [lm.x for lm in selected_landmarks]
            + [lm.y for lm in selected_landmarks]
            + [lm.z for lm in selected_landmarks]
        )

    return np.array([flattened_landmarks], dtype=np.float32)


def predict_emotion(landmarks, model, expected_landmark_indices):
    """
    Predicts the emotion from facial landmarks using the trained model.

    Args:
        landmarks (list): List of facial landmarks from MediaPipe or tuple format.
        model (tf.keras.Model): The trained emotion recognition model.
        expected_landmark_indices (list): Indices of landmarks expected by the model.

    Returns:
        tuple: (predicted_emotion: str, confidence: float)
            - predicted_emotion: Name of the predicted emotion class
            - confidence: Probability score for the prediction (0-1)

    Example:
        >>> emotion, confidence = predict_emotion(landmarks, model, indices)
        >>> print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
    """
    # Preprocess input landmarks
    X_input = preprocess_landmarks(landmarks, expected_landmark_indices)

    # Predict using the model
    predictions = model.predict(X_input)

    # Convert logits to probabilities using softmax
    probabilities = tf.nn.softmax(predictions).numpy()[0]
    print("Softmax probabilities:", probabilities)

    # Get the class with the highest probability
    predicted_index = np.argmax(probabilities)
    # Use index directly if it's beyond our class_labels length
    predicted_emotion = (
        class_labels[predicted_index]
        if predicted_index < len(class_labels)
        else f"Class_{predicted_index}"
    )
    predicted_probability = probabilities[predicted_index]

    return predicted_emotion, predicted_probability


if __name__ == "__main__":
    # Example usage of the module
    model_path = "models/hyperband_model.keras"

    # Load the model
    model = load_model(model_path)

    if model:
        print("Model loaded successfully.")
    else:
        print("Failed to load the model.")
