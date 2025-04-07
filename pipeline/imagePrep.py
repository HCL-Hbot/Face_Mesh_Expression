"""
Image preparation and facial landmark detection utilities for the FaceMesh project.

This module provides functions for processing images and extracting facial landmarks
using MediaPipe's Face Mesh solution. It handles various image preprocessing steps
including grayscale conversion, face detection, cropping, normalization, and landmark
visualization.

The module supports the following key operations:
- Image preprocessing (grayscale conversion, cropping, normalization)
- Face detection using Haar Cascades
- Facial landmark detection and visualization
- Landmark data export to CSV format

Typical usage:
    ```python
    import cv2
    from imagePrep import grayscale_image, crop_face, normalize_image

    # Load and process an image
    img = cv2.imread("face_image.jpg")
    gray = grayscale_image(img)
    face = crop_face(gray)
    if face is not None:
        normalized = normalize_image(face)
        # Process normalized face image...
    ```
"""

import os
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from data_load import get_landmarks
from fileHandler import create_facemesh_csv, create_output_dir, load_files
from mediapipe.framework.formats import landmark_pb2


def grayscale_image(img: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.

    Args:
        img (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Grayscale version of the input image.

    Example:
        ```python
        img = cv2.imread("color_image.jpg")
        gray = grayscale_image(img)
        ```
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop_face(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect and crop a face from a grayscale image.

    Uses OpenCV's Haar Cascade classifier to detect faces and crops the first
    detected face. The cropped face is resized to 512x512 pixels.

    Args:
        img (np.ndarray): Grayscale input image.

    Returns:
        Optional[np.ndarray]: Cropped and resized face image (512x512 pixels),
            or None if no face is detected.

    Raises:
        IOError: If the Haar Cascade classifier file cannot be loaded.

    Example:
        ```python
        gray_img = grayscale_image(cv2.imread("face.jpg"))
        face = crop_face(gray_img)
        if face is not None:
            cv2.imshow("Cropped Face", face)
        ```
    """
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Check if the cascade file is loaded correctly
    if face_cascade.empty():
        raise IOError(
            "Cannot load cascade classifier 'haarcascade_frontalface_default.xml'"
        )

    # Detect faces
    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    # Crop the face
    for x, y, w, h in faces:
        face = img[y : y + h, x : x + w]
        face = cv2.resize(face, (512, 512))  # Scale cropped face to 512x512
        return face

    return None  # Return None if no face is found


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to normalize
    image intensity.

    This normalization helps improve image contrast and reduce lighting variations
    across different images.

    Args:
        img (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: Normalized image with enhanced contrast.

    Example:
        ```python
        face = crop_face(grayscale_image(cv2.imread("face.jpg")))
        if face is not None:
            normalized = normalize_image(face)
            cv2.imshow("Normalized Face", normalized)
        ```
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def preview_landmarks(
    img: np.ndarray, landmarks: List[landmark_pb2.NormalizedLandmarkList]
) -> None:
    """
    Visualize facial landmarks detected by MediaPipe Face Mesh.

    Draws facial landmark connections and highlights important facial features
    (eyes and lips) on the input image.

    Args:
        img (np.ndarray): Input image (BGR format).
        landmarks (List[landmark_pb2.NormalizedLandmarkList]): List of facial
            landmarks detected by MediaPipe Face Mesh.

    Raises:
        ValueError: If the input image doesn't have 3 color channels.

    Example:
        ```python
        img = cv2.imread("face.jpg")
        landmarks = get_landmarks(img)
        if landmarks:
            preview_landmarks(img, landmarks)
            cv2.waitKey(0)
        ```
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    if len(img.shape) < 3 or img.shape[2] != 3:
        raise ValueError("Image does not have 3 channels (RGB/BGR)")

    for face_landmarks in landmarks:
        # Draw face mesh tessellation
        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        # Highlight eyes and mouth with green
        for feature in [
            mp_face_mesh.FACEMESH_LEFT_EYE,
            mp_face_mesh.FACEMESH_RIGHT_EYE,
            mp_face_mesh.FACEMESH_LIPS,
        ]:
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=feature,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                ),
            )
    cv2.imshow("Facemesh Landmarks", img)


def save_facemesh(
    filename: str,
    class_name: str,
    landmarks: List[landmark_pb2.NormalizedLandmarkList],
    facemesh_csv_path: str,
) -> None:
    """
    Save facial landmarks to a CSV file with associated metadata.

    Exports landmark coordinates (x, y, z) along with filename, hash, and class
    information to a CSV file. Creates the file if it doesn't exist or appends
    to it if it does.

    Args:
        filename (str): Path to the source image file.
        class_name (str): Classification label for the face.
        landmarks (List[landmark_pb2.NormalizedLandmarkList]): Detected facial
            landmarks.
        facemesh_csv_path (str): Path to the output CSV file.

    Example:
        ```python
        landmarks = get_landmarks(img)
        if landmarks:
            save_facemesh(
                "face.jpg",
                "happy",
                landmarks,
                "facial_landmarks.csv"
            )
        ```
    """
    if landmarks:
        for face_landmarks in landmarks:
            # Prepare row data
            row = [filename, hash(filename), class_name]
            for landmark in face_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])

            # Define column names
            columns = ["filename", "hash", "class"]
            for i in range(len(face_landmarks.landmark)):
                columns.extend(
                    [
                        f"landmark{i + 1:03d}_x",
                        f"landmark{i + 1:03d}_y",
                        f"landmark{i + 1:03d}_z",
                    ]
                )

            # Write to CSV
            df = pd.DataFrame([row], columns=columns)
            df.to_csv(
                facemesh_csv_path,
                mode="a",
                index=False,
                header=not os.path.exists(facemesh_csv_path),
            )


if __name__ == "__main__":
    # Load the list of image files
    file_list = load_files()
    # Create the output directory
    output_dir = create_output_dir()
    # Create the facemesh CSV file
    facemesh_csv_path = create_facemesh_csv(output_dir)

    # Process each image file
    for filename in file_list:
        print(filename)
        img = cv2.imread(filename)
        print(f"Resolution: {img.shape[1]}x{img.shape[0]}")

        # Save facemesh landmarks to CSV before any image manipulation
        landmarks = get_landmarks(img)
        class_name = os.path.basename(os.path.dirname(filename))
        if landmarks is not None:
            save_facemesh(filename, class_name, landmarks, facemesh_csv_path)
        else:
            print(f"No face detected in {filename}")
            continue

        # Convert the image to grayscale
        gray = grayscale_image(img)
        # Use full grayscale image instead of cropping
        face = gray

        if face is not None:
            # Normalize the face image
            norm = normalize_image(face)
            # Save the normalized image
            cv2.imwrite(os.path.join(output_dir, os.path.basename(filename)), norm)
        else:
            print(f"No face detected in {filename}")

    # Close all OpenCV windows
    cv2.destroyAllWindows()
