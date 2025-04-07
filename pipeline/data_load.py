import glob
import os
import warnings

import cv2
import keras
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore", category=UserWarning)
from config import config

# Global instance of MediaPipe FaceMesh used for facial landmark detection
# Initialized as None and created when needed to avoid unnecessary memory usage
mp_face_mesh = None

# Initialize the MediaPipe FaceMesh detector with configuration parameters
# - static_image_mode=True: Optimized for images rather than video
# - max_num_faces=1: Process only one face per image
# - refine_landmarks=True: Enable more accurate landmark detection
# - min_detection_confidence=0.1: Lower threshold to detect more faces
# - min_tracking_confidence=0.1: Lower threshold for landmark tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1,
)


def load_face_mesh():
    """
    Initialize or return an existing MediaPipe FaceMesh instance.

    This function ensures we only create one instance of MediaPipe FaceMesh
    to optimize memory usage. It uses a global variable to store the instance
    between function calls.

    Returns:
        mediapipe.solutions.face_mesh: Initialized FaceMesh instance for landmark detection

    Example:
        >>> face_mesh = load_face_mesh()
        >>> # Use face_mesh for detection
        >>> results = face_mesh.process(image)

    Notes:
        - Uses global variable for singleton pattern implementation
        - Thread-safe as MediaPipe handles internal synchronization
        - Subsequent calls return the same instance
    """
    global mp_face_mesh
    if mp_face_mesh is None:
        mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh


def get_latest_csv():
    """
    Retrieve the most recently modified CSV file from the landmarks directory.
    This ensures we're always working with the latest landmark data.

    The function searches recursively through the landmarks directory defined in config.LANDMARKS_DIR
    looking for CSV files. It uses file modification time to determine the most recent file.

    Returns:
        str or None: Absolute path to the most recent CSV file, or None if no CSV files found

    Notes:
        - Prints the search directory path and all found CSV files for debugging
        - Uses glob.glob with recursive=True to find all nested CSV files
    """

    # Print in which absolute directory we are looking for the CSV files
    print("Looking for CSV files in: ", config.LANDMARKS_DIR)

    # Search the landmarks directory recursively for CSV files
    csv_files = glob.glob(
        os.path.join(config.LANDMARKS_DIR, "**", "*.csv"), recursive=True
    )

    # print which files are found
    print("CSV Files found: ", csv_files)

    if csv_files:
        latest_csv = max(csv_files, key=os.path.getmtime)
        print("Using CSV: ", latest_csv)
        return latest_csv

    return None


def apply_augmentations(df):
    """
    Apply configurable data augmentations to facial landmark coordinates to increase dataset variety.

    This function helps prevent overfitting by creating additional training samples through
    controlled modifications of the original data. The augmentations simulate real-world
    variations in facial detection.

    Augmentations applied:
    1. Gaussian noise addition
        - Adds random noise to landmark positions
        - Simulates detection variations and sensor noise
        - Scale controlled by config.AUGMENTATION_INTENSITY

    2. Random scaling
        - Uniformly scales all landmarks
        - Simulates different face sizes and distances from camera
        - Scale range controlled by config.AUGMENTATION_INTENSITY

    3. Horizontal flipping
        - Mirrors landmarks across vertical axis
        - Helps model learn facial symmetry
        - Only affects x-coordinates (inverts them around 0.5)

    Args:
        df (pd.DataFrame): DataFrame containing landmark coordinates (x,y,z) and expression labels

    Returns:
        pd.DataFrame: New DataFrame containing both original and augmented samples.
                     Labels are preserved for all samples.
                     Returns original data unchanged if config.SHOULD_AUGMENT is False.

    Notes:
        - All augmented samples are shuffled in the output DataFrame
        - Prints statistics about augmentation results if config.VERBOSE > 0
        - Uses numpy random generator with controlled seed for reproducibility
    """
    if not config.SHOULD_AUGMENT or len(df) == 0:
        return df

    # Keep original data
    augmented = [df.copy()]

    # Get landmark columns (all except label)
    landmark_cols = [col for col in df.columns if col != "label"]

    # 1. Gaussian Noise
    noisy = df.copy()
    noise = np.random.normal(
        scale=config.AUGMENTATION_INTENSITY, size=noisy[landmark_cols].shape
    )
    noisy[landmark_cols] += noise
    augmented.append(noisy)

    # 2. Random Scaling
    scaled = df.copy()
    scale = 1 + np.random.uniform(
        -config.AUGMENTATION_INTENSITY, config.AUGMENTATION_INTENSITY
    )
    scaled[landmark_cols] *= scale
    augmented.append(scaled)

    # 3. Horizontal Flip (only x coordinates)
    flipped = df.copy()
    x_cols = [col for col in landmark_cols if "_x" in col]
    flipped[x_cols] = 1 - flipped[x_cols]
    augmented.append(flipped)

    # Combine and shuffle all augmented data
    result = pd.concat(augmented).sample(frac=1).reset_index(drop=True)

    if config.VERBOSE > 0:
        print("\n=== Data Augmentation ===")
        print(f"Original samples: {len(df)}")
        print(f"Augmented samples: {len(result)}")
        print("Label distribution:")
        print(result["label"].value_counts())

    return result


# def load_landmarks():
#     """
#     Load the most recent landmarks CSV file into a DataFrame
#     If no CSV file is found, process images from the image directory,
#     extract landmarks, and save them to a new CSV file.
#     """
#     if latest_csv := get_latest_csv():
#         df = pd.read_csv(latest_csv)
#         print("\n=== Loading from CSV ===")
#         print(f"Label distribution:\n{df['label'].value_counts()}")

#         # Verify if the landmarks are properly filtered
#         face_mesh = load_face_mesh()

#         # Convert frozensets to lists and combine
#         right_eye = [idx for pair in face_mesh.FACEMESH_RIGHT_EYE for idx in pair]
#         left_eye = [idx for pair in face_mesh.FACEMESH_LEFT_EYE for idx in pair]
#         lips = [idx for pair in face_mesh.FACEMESH_LIPS for idx in pair]

#         # Get unique landmarks and multiply by 3 for x,y,z coordinates
#         expected_features = len(set(right_eye + left_eye + lips)) * 3

#         actual_features = len(df.columns) - 1  # Subtract 1 for label column
#         if actual_features > expected_features:
#             print(f"\nWarning: CSV contains {actual_features} features but expected {expected_features}")
#             print("Reapplying facial feature filter...")
#             df = filter_facial_features(df)
#             landmarks_to_csv(df)  # Save the filtered version
#             print(f"Saved filtered landmarks with {len(df.columns) - 1} features")

#         return df

#     # If no CSV exists, process images and create landmarks
#     print("No existing landmarks CSV found. Processing images...")
#     df = images_to_landmarks(config.IMAGE_DIR)

#     if len(df) > 0:
#         # Clean and save the landmarks
#         print("\n=== Initial Dataset ===")
#         print(f"Label distribution:\n{df['label'].value_counts()}")

#         df = detect_outliers(df)
#         print("\n=== After Outlier Detection ===")
#         print(f"Label distribution:\n{df['label'].value_counts()}")

#         df = filter_zero_landmarks(df)
#         print("\n=== After Zero Filtering ===")
#         print(f"Label distribution:\n{df['label'].value_counts()}")

#         df = filter_facial_features(df)
#         print("\n=== After Facial Feature Filtering ===")
#         print(f"Label distribution:\n{df['label'].value_counts()}")
#         print(f"Number of features: {len(df.columns) - 1}")  # -1 for label column

#         landmarks_to_csv(df)
#         print(f"Successfully processed {len(df)} images and saved landmarks to CSV")
#         return df

#     print("No valid images found for processing")
#     return pd.DataFrame()


def validate_landmark_count(input_shape, config):
    """
    Validate that the input shape matches the expected number of facial landmark features.

    This validation ensures that the model's input layer matches the number of features
    we extract from the facial landmarks. Each landmark has x, y, z coordinates, so the
    total feature count should be 3 times the number of unique landmarks we track.

    Args:
        input_shape (tuple): Shape of the input data from the model or dataset.
        config (object): Configuration object containing FACIAL_FEATURES list that defines
                        which landmark connections we use for expression recognition.

    Raises:
        ValueError: If there's a mismatch between the expected number of features
                   (3 * number of unique landmarks) and the actual input shape.

    Notes:
        - Prints all unique landmark indices for debugging
        - Shows both expected and actual feature counts
        - Useful for catching configuration errors early in the pipeline
    """
    # Extract unique landmark indices from facial feature definitions
    unique_landmarks = set()

    for feature in config.FACIAL_FEATURES:
        for pair in feature:
            unique_landmarks.update(pair)  # Add both points in the connection

    print("Unique landmark indices:", sorted(unique_landmarks))
    print("Total unique landmarks:", len(unique_landmarks))
    # Compute expected feature count (each landmark has x, y, z coordinates)
    expected_landmark_count = len(unique_landmarks) * 3

    if input_shape[0] != expected_landmark_count:
        raise ValueError(
            f"Mismatch: Expected {expected_landmark_count} landmarks, got {input_shape[0]}"
        )

    print(
        f"Validation successful: Expected {expected_landmark_count} landmarks, got {input_shape[0]}"
    )


def get_facial_feature_indices():
    """
    Extract landmark indices for facial features relevant to expression recognition.

    This function processes the facial feature connections defined in config.FACIAL_FEATURES
    to get a unique, sorted list of landmark indices. These indices correspond to key
    points on the face that are most important for detecting expressions.

    The features typically include points around:
    - Eyes and eyebrows (for expressions like surprise, anger)
    - Mouth and lips (for expressions like happiness, sadness)
    - Nose (for expressions like disgust)

    Returns:
        list: Sorted list of unique landmark indices used for expression recognition.
              Each index corresponds to a specific point on the face mesh.

    Notes:
        - Removes duplicate indices since points may appear in multiple connections
        - Sorting provides consistent ordering for feature extraction
        - Used by filter_facial_features() to select relevant coordinates
    """
    mp_face_mesh = mp.solutions.face_mesh

    # Define relevant facial feature groups

    # Extract unique landmark indices
    landmark_indices = set()
    for feature in config.FACIAL_FEATURES:
        for connection in feature:
            landmark_indices.update(connection)  # Add both points in each connection

    return sorted(landmark_indices)  # Sort for consistency


def filter_facial_features(df):
    """
    Filter the DataFrame to only include facial landmarks relevant for expression recognition.
    Keeps only landmarks for key facial features like eyes, eyebrows, mouth, and nose
    defined in config.FACIAL_FEATURES.

    Args:
        df (pd.DataFrame): DataFrame containing all facial landmark coordinates (x,y,z)

    Returns:
        pd.DataFrame: Filtered DataFrame with only the selected facial feature landmarks,
                     reducing dimensionality while preserving expression-relevant information
    """
    if len(df) == 0:
        return df

    # Get dynamically selected landmark indices
    selected_indices = get_facial_feature_indices()

    # Create list of columns to keep (label + selected landmarks)
    filtered_columns = (
        ["label"]
        + [f"landmark{idx:03d}_x" for idx in selected_indices]
        + [f"landmark{idx:03d}_y" for idx in selected_indices]
        + [f"landmark{idx:03d}_z" for idx in selected_indices]
    )

    # Return filtered DataFrame with only the required columns
    return df[filtered_columns]


def load_landmarks():
    """
    Load or create facial landmark data from images or existing CSV.

    This function is the primary entry point for facial landmark data preparation.
    It first attempts to load preprocessed landmarks from the most recent CSV file.
    If no suitable CSV exists, it processes the raw images to extract landmarks.

    Processing pipeline when loading from CSV:
    1. Find and load the most recent landmarks CSV file
    2. Verify the number of features matches expectations
    3. Reapply facial feature filtering if needed

    Processing pipeline when creating new landmarks:
    1. Process all images to extract facial landmarks
    2. Apply outlier detection to remove anomalous samples
    3. Filter out samples with zero-valued landmarks
    4. Apply facial feature filtering to reduce dimensionality
    5. Save the processed landmarks to a new CSV file

    Returns:
        pd.DataFrame: DataFrame containing:
            - Filtered facial landmark coordinates (x,y,z)
            - Expression labels from image directories
            Returns empty DataFrame if no valid data found

    Notes:
        - Prints detailed progress and statistics at each step
        - Automatically saves processed data for future use
        - Verifies data quality through multiple filtering steps
    """
    if latest_csv := get_latest_csv():
        df = pd.read_csv(latest_csv)
        print("\n=== Loading from CSV ===")
        print(f"Label distribution:\n{df['label'].value_counts()}")

        # Get expected number of features dynamically
        expected_features = len(get_facial_feature_indices()) * 3  # x, y, z

        actual_features = len(df.columns) - 1  # Exclude label column
        if actual_features > expected_features:
            print(
                f"\nWarning: CSV contains {actual_features} features but expected {expected_features}"
            )
            print("Reapplying facial feature filter...")
            df = filter_facial_features(df)
            landmarks_to_csv(df)  # Save the filtered version
            print(f"Saved filtered landmarks with {len(df.columns) - 1} features")

        return df

    # If no CSV exists, process images and create landmarks
    print("No existing landmarks CSV found. Processing images...")
    df = images_to_landmarks(config.IMAGE_DIR)

    if len(df) > 0:
        # Clean and save the landmarks
        print("\n=== Initial Dataset ===")
        print(f"Label distribution:\n{df['label'].value_counts()}")

        df = detect_outliers(df)
        print("\n=== After Outlier Detection ===")
        print(f"Label distribution:\n{df['label'].value_counts()}")

        df = filter_zero_landmarks(df)
        print("\n=== After Zero Filtering ===")
        print(f"Label distribution:\n{df['label'].value_counts()}")

        df = filter_facial_features(df)
        print("\n=== After Facial Feature Filtering ===")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        print(f"Number of features: {len(df.columns) - 1}")  # -1 for label column

        landmarks_to_csv(df)
        print(f"Successfully processed {len(df)} images and saved landmarks to CSV")
        return df

    print("No valid images found for processing")
    return pd.DataFrame()


def get_landmarks(img):
    """
    Extract facial landmarks from an input image using MediaPipe FaceMesh.

    This function processes a single image to detect facial landmarks using MediaPipe's
    FaceMesh model. It handles both color and grayscale images by performing appropriate
    color space conversions.

    Args:
        img (numpy.ndarray): Input image in BGR or grayscale format. Shape should be
                           either (height, width) for grayscale or (height, width, 3)
                           for color images.

    Returns:
        list or None: List of detected facial landmarks with their 3D coordinates (x,y,z).
                     Each landmark is a NamedTuple with attributes x, y, z in normalized
                     coordinates (0.0-1.0). Returns None if no face is detected.

    Raises:
        ValueError: If input image is None or has invalid dimensions
        RuntimeError: If face detection fails due to MediaPipe errors

    Example:
        >>> img = cv2.imread('face.jpg')
        >>> landmarks = get_landmarks(img)
        >>> if landmarks:
        ...     # Process first face's landmarks
        ...     face_landmarks = landmarks[0]
        ...     for point in face_landmarks.landmark:
        ...         x, y, z = point.x, point.y, point.z
        ...         # Use coordinates for analysis

    Notes:
        - Uses configuration parameters from config for detection thresholds
        - MediaPipe FaceMesh provides 468 3D landmarks per face
        - Only processes the first detected face (max_num_faces=1)
        - Image is automatically converted to RGB for processing
        - Landmark coordinates are normalized to [0.0, 1.0] range
    """
    face_mesh = load_face_mesh()

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with face_mesh.FaceMesh(
        static_image_mode=config.FACE_MESH_STATIC_MODE,
        min_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONFIDENCE,
    ) as face_detector:
        results = face_detector.process(img)
        landmarks = results.multi_face_landmarks

    return landmarks


def images_to_landmarks(image_dir):
    """
    Process all facial expression images to extract landmarks for training.

    This function handles bulk processing of facial expression images organized in a
    directory structure where each subdirectory name represents an expression label.
    It uses MediaPipe FaceMesh to detect and extract facial landmarks from each image.

    Directory structure expected:
    image_dir/
        happy/
            image1.jpg
            image2.png
            ...
        sad/
            image1.jpg
            ...
        [other_expressions]/
            ...

    Processing steps:
    1. Recursively find all supported image files (jpg, jpeg, png)
    2. For each image:
        - Load using OpenCV
        - Detect face landmarks using MediaPipe FaceMesh
        - Extract x,y,z coordinates for each landmark
        - Use parent directory name as expression label
    3. Combine all processed data into a DataFrame

    Args:
        image_dir (str): Root directory containing subdirectories of expression images

    Returns:
        pd.DataFrame: DataFrame with columns:
            - label: Expression label (from directory name)
            - landmark{N}_{x,y,z}: Coordinates for each detected landmark

    Notes:
        - Skips images that can't be loaded or where no face is detected
        - Processes only the first detected face in each image
        - Maintains order of landmarks for consistency
    """
    # Get list of image files recursively
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(
            glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
        )

    # List to store landmark data for each image
    all_landmarks = []
    processed_count = 0

    # Process each image
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is not None and (landmarks := get_landmarks(img)) is not None:
            face_landmarks = landmarks[0]

            # Create dictionary for this image's data
            landmark_dict = {"label": os.path.basename(os.path.dirname(img_path))}

            # Add coordinates for each landmark
            for idx, landmark in enumerate(face_landmarks.landmark):
                landmark_dict[f"landmark{idx:03d}_x"] = landmark.x
                landmark_dict[f"landmark{idx:03d}_y"] = landmark.y
                landmark_dict[f"landmark{idx:03d}_z"] = landmark.z

            all_landmarks.append(landmark_dict)
            processed_count += 1

    return pd.DataFrame(all_landmarks)


def detect_outliers(df, contamination=0.05):
    """
    Detect and remove outlier samples using the Isolation Forest algorithm.

    This function identifies and removes samples with unusual landmark patterns that may
    represent detection errors or non-typical facial expressions. It uses scikit-learn's
    Isolation Forest implementation for outlier detection.

    The algorithm works by isolating observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Args:
        df (pd.DataFrame): DataFrame containing landmark coordinates and expression labels
        contamination (float): Expected proportion of outliers in the dataset (default 5%)
                             Range: (0, 0.5], higher values remove more samples

    Returns:
        pd.DataFrame: DataFrame with outlier samples removed, maintaining original structure
                     but with potentially fewer rows

    Notes:
        - Only considers landmark coordinates (x,y,z) for outlier detection
        - Requires at least 2 samples for outlier detection
        - Uses random seed 42 for reproducibility
        - Prints detailed statistics about removed samples
    """
    if len(df) < 2:
        print("Not enough samples for outlier detection")
        return df

    print("\n=== Pre-Outlier Detection ===")
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Select only landmark columns
    landmark_columns = [
        col for col in df.columns if "_x" in col or "_y" in col or "_z" in col
    ]
    landmarks = df[landmark_columns]

    # Apply Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso.fit_predict(landmarks)

    # Remove outliers and print info
    num_outliers = (outliers == -1).sum()
    print(f"Number of outliers detected and removed: {num_outliers}")

    filtered_df = df[outliers == 1]
    print("\n=== Post-Outlier Detection ===")
    print(f"Remaining samples: {len(filtered_df)}")
    print(f"Label distribution:\n{filtered_df['label'].value_counts()}")

    return filtered_df


def landmarks_to_csv(df):
    """
    Save facial landmark coordinates and labels to a CSV file.

    This function persists the processed landmark data to disk for future use,
    avoiding the need to reprocess images. The output path is specified in the
    config.LANDMARKS_PATH setting.

    Args:
        df (pd.DataFrame): DataFrame containing:
            - Landmark coordinates in columns named 'landmark{N}_{x,y,z}'
            - Expression labels in a 'label' column

    Returns:
        str: Absolute path of the saved CSV file

    Notes:
        - Creates parent directories if they don't exist
        - Saves without index column to maintain consistency
        - Prints absolute path of saved file for verification
        - File can be reloaded using get_latest_csv() and csv_to_df()
    """
    # Get the full path of the saved file
    output_path = os.path.abspath(config.LANDMARKS_PATH)

    # Save the file
    df.to_csv(config.LANDMARKS_PATH, index=False)

    # Print the saved path
    print(f"Landmarks saved to: {output_path}")

    return output_path


def csv_to_df(csv_path):
    """
    Load facial landmark data from a CSV file into a pandas DataFrame.

    This function provides direct access to saved landmark data, complementing
    the landmarks_to_csv function. It handles basic error checking and provides
    a consistent interface for data loading.

    Args:
        csv_path (str): Path to the CSV file containing landmark data. File should
                       contain columns for landmark coordinates and expression labels.

    Returns:
        pd.DataFrame: DataFrame containing landmark coordinates and labels.
                     Returns empty DataFrame if:
                     - File doesn't exist
                     - File can't be read
                     - File has invalid format

    Notes:
        - Uses pandas read_csv with default settings
        - Assumes landmark naming convention: landmark{N}_{x,y,z}
        - Expects 'label' column for expression classes
        - Prints error message if file not found
    """
    if not os.path.exists(csv_path):
        print("File not found")
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def filter_zero_landmarks(df):
    """
    Remove samples with zero-valued landmark coordinates from the dataset.

    Zero-valued coordinates typically indicate detection failures, where MediaPipe
    FaceMesh failed to correctly locate certain facial landmarks. Removing these
    samples helps ensure data quality for training.

    Filtering process:
    1. Check each sample for any zero values across all landmark coordinates
    2. Remove samples that contain any zeros
    3. Print statistics about removed samples and remaining data distribution

    Args:
        df (pd.DataFrame): DataFrame containing landmark coordinates and labels

    Returns:
        pd.DataFrame: Filtered DataFrame with zero-containing samples removed.
                     Maintains original column structure but with fewer rows.
                     Returns empty DataFrame if input is empty.

    Notes:
        - Prints detailed statistics about removed samples
        - Shows label distribution before and after filtering
        - Helps identify potential systematic detection issues
    """
    if len(df) == 0:
        return df

    print("\n=== Pre-Zero Filtering ===")
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    zero_mask = (df == 0).any(axis=1)
    print(f"Number of samples removed due to zero landmarks: {zero_mask.sum()}")

    filtered_df = df[~zero_mask]
    print("\n=== Post-Zero Filtering ===")
    print(f"Remaining samples: {len(filtered_df)}")
    print(f"Label distribution:\n{filtered_df['label'].value_counts()}")

    return filtered_df


def dataframe_to_dataset(df):
    """
    Convert pandas DataFrame to TensorFlow dataset format for model training.

    This function performs several preprocessing steps to prepare the landmark data
    for training:
    1. Separates features (landmark coordinates) from expression labels
    2. Converts string labels to one-hot encoded format using TensorFlow's StringLookup
    3. Creates a tf.data.Dataset for efficient training pipeline integration

    Args:
        df (pd.DataFrame): DataFrame containing:
            - Landmark coordinates (x,y,z) as feature columns
            - Expression labels in a 'label' column

    Returns:
        tuple: Three elements:
            - tf.data.Dataset: Combined features and labels ready for training
            - numpy.ndarray: One-hot encoded labels for stratification
            - tf.keras.layers.StringLookup: Fitted label encoder for inference
              Returns (None, None, None) if input DataFrame is empty

    Notes:
        - Prints detailed class distribution before and after encoding
        - Uses special StringLookup configuration to prevent OOV tokens
        - Labels must be string type in the input DataFrame
        - Features should be floating point values
    """
    if len(df) == 0:
        return None, None, None

    # Extract features and labels
    features = df.drop("label", axis=1).values
    labels = df["label"].values

    print("\n=== Label Distribution ===")
    print(df["label"].value_counts())

    # Convert string labels to integers using StringLookup
    # label_lookup = keras.layers.StringLookup(output_mode='one_hot')
    label_lookup = keras.layers.StringLookup(
        output_mode="one_hot", mask_token=None, oov_token=None, num_oov_indices=0
    )
    label_lookup.adapt(np.expand_dims(labels, -1))  # Adapt to our label vocabulary

    # Convert labels to one-hot encoded format
    labels_one_hot = label_lookup(np.expand_dims(labels, -1)).numpy()

    # Print one-hot encoded distribution
    print("\n=== One-Hot Encoded Distribution ===")
    label_counts = np.sum(labels_one_hot, axis=0)
    for i, count in enumerate(label_counts):
        print(f"Class {i}: {int(count)} samples")

    # Convert to tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels_one_hot))
    return dataset, labels_one_hot, label_lookup


def split_dataset(dataset, labels):
    """
    Split dataset into training and validation sets while maintaining class distribution.
    Uses stratified splitting to ensure each expression class is proportionally
    represented in both training and validation sets.

    Args:
        dataset: tf.data.Dataset containing features and labels
        labels: One-hot encoded expression labels for stratification

    Returns:
        tuple: (train_dataset, val_dataset) with config.VALIDATION_SPLIT ratio
               Each returned dataset maintains the original class distribution
    """
    if dataset is None:
        return None, None

    # Calculate split sizes
    dataset_size = sum(1 for _ in dataset)
    if dataset_size == 0:
        print("Error: Empty dataset")
        return None, None

    # Convert dataset to numpy for stratification
    features = np.array(list(dataset.map(lambda x, _: x)))
    labels_np = np.array(list(dataset.map(lambda _, y: y)))

    # Get class indices for stratification
    class_indices = np.argmax(labels_np, axis=1)

    print("\n=== Class Distribution Before Split ===")
    unique, counts = np.unique(class_indices, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples")

    # Ensure minimum samples per class for validation
    min_samples_per_class = np.min(counts)
    if min_samples_per_class < 2:
        print(
            f"Warning: Some classes have fewer than 2 samples (min: {min_samples_per_class})"
        )

    # Create validation indices ensuring class representation
    val_indices = []
    for class_idx in unique:
        class_samples = np.where(class_indices == class_idx)[0]
        val_size = max(1, int(len(class_samples) * config.VALIDATION_SPLIT))
        val_indices.extend(np.random.choice(class_samples, val_size, replace=False))

    # Create masks for splitting
    all_indices = np.arange(dataset_size)
    train_indices = np.setdiff1d(all_indices, val_indices)

    # Print split details
    print("\n=== Split Distribution ===")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (features[train_indices], labels_np[train_indices])
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (features[val_indices], labels_np[val_indices])
    )

    # Print class distribution for both sets
    train_labels = np.argmax(labels_np[train_indices], axis=1)
    val_labels = np.argmax(labels_np[val_indices], axis=1)

    print("\n=== Training Set Distribution ===")
    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    for cls, count in zip(train_unique, train_counts):
        print(f"Class {cls}: {count} samples")

    print("\n=== Validation Set Distribution ===")
    val_unique, val_counts = np.unique(val_labels, return_counts=True)
    for cls, count in zip(val_unique, val_counts):
        print(f"Class {cls}: {count} samples")

    # Shuffle datasets
    train_dataset = train_dataset.shuffle(buffer_size=len(train_indices))
    val_dataset = val_dataset.shuffle(buffer_size=len(val_indices))

    return train_dataset, val_dataset


def shuffle_dataset(dataset, buffer_size):
    """
    Apply configurable shuffling to a TensorFlow dataset.

    This function conditionally shuffles the dataset based on configuration settings.
    Shuffling is important for preventing the model from learning unintended patterns
    from the order of training samples.

    Args:
        dataset (tf.data.Dataset): Dataset to potentially shuffle
        buffer_size (int): Size of the shuffle buffer. Larger values give more
                          randomization but require more memory. Should typically
                          be similar to the dataset size for best randomization.

    Returns:
        tf.data.Dataset: Shuffled dataset if config.SHOULD_SHUFFLE is True,
                        otherwise returns the original dataset unchanged

    Notes:
        - Uses TensorFlow's shuffle operation with a fixed buffer size
        - Only shuffles if enabled in configuration (config.SHOULD_SHUFFLE)
        - Buffer size affects both memory usage and randomization quality
        - Returns original dataset if shuffling is disabled
    """
    if config.SHOULD_SHUFFLE:
        return dataset.shuffle(buffer_size=buffer_size)
    return dataset


def batch_dataset(dataset):
    """
    Apply configurable batching to a TensorFlow dataset.

    This function conditionally batches the dataset based on configuration settings.
    Batching is essential for efficient training and properly utilizing hardware
    acceleration (GPU/TPU).

    Args:
        dataset (tf.data.Dataset): Dataset to potentially batch

    Returns:
        tf.data.Dataset: Batched dataset if config.SHOULD_BATCH is True,
                        using config.BATCH_SIZE for the batch size.
                        Returns original dataset if batching is disabled.

    Notes:
        - Uses TensorFlow's batch operation with fixed batch size
        - Only batches if enabled in configuration (config.SHOULD_BATCH)
        - Drops remainder samples if they don't fill a complete batch
        - Important for optimal GPU/TPU utilization during training
    """
    if config.SHOULD_BATCH:
        return dataset.batch(config.BATCH_SIZE, drop_remainder=True)
    return dataset


def get_datasets():
    """
    Main function to prepare training and validation datasets.
    Complete pipeline:
    1. Load landmark data from CSV or process images if no CSV exists
    2. Apply data augmentation if configured
    3. Convert to TensorFlow dataset format
    4. Split into training and validation sets
    5. Apply batching and other processing

    Returns:
        tuple: (train_dataset, val_dataset) ready for model training,
               or (None, None) if data preparation fails
    """
    # Load and preprocess landmarks
    df = load_landmarks()
    if len(df) == 0:
        return None, None

    # Apply data augmentation
    df = apply_augmentations(df)

    # Convert to TensorFlow dataset
    dataset, labels, label_lookup = dataframe_to_dataset(df)
    if dataset is None:
        return None, None

    # Split dataset while maintaining class distribution
    train_dataset, val_dataset = split_dataset(dataset, labels)
    if train_dataset is None or val_dataset is None:
        return None, None

    # Process datasets
    train_dataset = batch_dataset(train_dataset)
    val_dataset = batch_dataset(val_dataset)

    # Get shape information
    feature_shape = next(iter(train_dataset))[0].shape
    label_shape = next(iter(train_dataset))[1].shape

    print("\n=== Dataset Details ===")
    print(f"Feature shape: {feature_shape}")
    print(f"Label shape: {label_shape}")

    # Update config
    config.INPUT_SHAPE = feature_shape[1]  # Remove batch dimension
    config.NUM_CLASSES = label_shape[1]

    # Print final dataset sizes
    print(f"\nTraining samples: {sum(1 for _ in train_dataset)}")
    print(f"Validation samples: {sum(1 for _ in val_dataset)}")

    return train_dataset, val_dataset


def get_dataset_details(train_dataset, val_dataset):
    """
    Extract key information about the prepared datasets.

    This function analyzes the training and validation datasets to provide
    important statistics about their structure and composition. It's useful
    for verifying dataset preparation and model configuration.

    Information extracted:
    1. Feature dimensionality (number of landmark coordinates)
    2. Number of expression classes
    3. Total number of training samples
    4. Total number of validation samples

    Args:
        train_dataset (tf.data.Dataset): Prepared training dataset
        val_dataset (tf.data.Dataset): Prepared validation dataset

    Returns:
        tuple: (num_features, num_classes, total_train_samples, total_val_samples)
            - num_features: Number of input features per sample
            - num_classes: Number of unique expression classes
            - total_train_samples: Total training samples across all batches
            - total_val_samples: Total validation samples across all batches

    Notes:
        - Examines first batch to determine feature and class dimensions
        - Counts total samples by iterating through all batches
        - Useful for model architecture decisions and training configuration
    """
    # Get a sample batch for shape information
    sample_features, sample_labels = next(iter(train_dataset))
    feature_shape = sample_features.shape
    label_shape = sample_labels.shape

    # Feature amount (features per sample)
    num_features = feature_shape[1]
    # Class amount (number of classes)
    num_classes = label_shape[1]

    # Get total sample counts if datasets are batched:
    total_train_samples = sum(features.shape[0] for features, _ in train_dataset)
    total_val_samples = sum(features.shape[0] for features, _ in val_dataset)

    return num_features, num_classes, total_train_samples, total_val_samples


def preview_dataset(df, dataset=None):
    """
    Generate a comprehensive preview of the preprocessed dataset.

    This function provides a detailed overview of the dataset's characteristics,
    including sample distribution, feature statistics, and example data. It's
    useful for data quality assessment and verification of preprocessing steps.

    Preview includes:
    1. Sample distribution across expression classes
    2. Dataset dimensions and feature counts
    3. Statistical summary of landmark coordinates
    4. Example features from a random sample

    Args:
        df (pd.DataFrame): DataFrame containing the landmark coordinates and labels
        dataset (tf.data.Dataset, optional): Dataset version if already converted
                                           Used for additional TensorFlow-specific info

    Notes:
        - Provides insights into data balance across classes
        - Shows feature value ranges and distributions
        - Helps identify potential preprocessing issues
        - Useful for verifying data preparation pipeline
    """
    if len(df) == 0:
        print("No data to preview")
        return

    print("\n=== Dataset Preview ===")

    # Show sample distribution
    print("\nSample distribution:")
    print(df["label"].value_counts())

    # Show dataset shape and features
    print(f"\nTotal samples: {len(df)}")
    landmark_columns = [
        col for col in df.columns if "_x" in col or "_y" in col or "_z" in col
    ]
    print(
        f"Number of landmarks: {len(landmark_columns) // 3}"
    )  # Divide by 3 for x,y,z coordinates

    # Show dataset statistics
    print("\nLandmark Statistics:")
    stats = df[landmark_columns].describe()
    print(f"Mean range: {stats.loc['mean'].min():.3f} to {stats.loc['mean'].max():.3f}")
    print(
        f"Std Dev range: {stats.loc['std'].min():.3f} to {stats.loc['std'].max():.3f}"
    )

    # Show example features for a random sample
    print("\nExample features from a random sample:")
    random_sample = df.sample(n=1)
    random_landmarks = {
        col: random_sample[col].values[0]
        for col in random_sample.columns
        if "_x" in col or "_y" in col or "_z" in col
    }
    print(f"Label: {random_sample['label'].values[0]}")
    print("First 5 landmarks:")
    for i, (col, val) in enumerate(
        list(random_landmarks.items())[:15]
    ):  # First 5 landmarks (x,y,z each)
        print(f"{col}: {val:.4f}")


def main():
    """
    Main entry point for facial landmark processing pipeline.

    This function demonstrates the complete landmark processing workflow:
    1. Initialize MediaPipe FaceMesh for facial detection
    2. Extract landmarks from all images in the configured directory
    3. Clean the data through multiple filtering steps:
        - Remove outlier samples
        - Filter out zero-valued landmarks
        - Select only relevant facial features
    4. Convert the processed data to TensorFlow format
    5. Split into training and validation sets
    6. Save the processed landmarks for future use

    The function primarily serves as an example of the full pipeline and
    can be used to regenerate the landmark dataset from raw images.
    """
    load_face_mesh()
    df = images_to_landmarks(config.IMAGE_DIR)
    print("\n=== Raw Dataset ===")
    print(f"Total samples: {len(df)}")

    df = detect_outliers(df)
    df = filter_zero_landmarks(df)
    df = filter_facial_features(df)

    print("\n=== Processed Dataset ===")
    print(f"Remaining samples: {len(df)}")
    print(f"Features shape: {df.shape[1] - 1}")  # Subtract 1 for label column

    dataset, labels, _ = dataframe_to_dataset(df)
    print("\n=== Dataset Details ===")
    print(f"Unique classes: {len(labels[0])}")
    print(f"Class distribution: {np.sum(labels, axis=0)}")
    print(f"Dataset features shape: {dataset.element_spec[0].shape}")
    print(f"Dataset labels shape: {dataset.element_spec[1].shape}")

    train_dataset, val_dataset = split_dataset(dataset, labels)
    print("\n=== Split Dataset ===")
    print(f"Training samples: {sum(1 for _ in train_dataset)}")
    print(f"Validation samples: {sum(1 for _ in val_dataset)}")

    landmarks_to_csv(df)


if __name__ == "__main__":
    main()
