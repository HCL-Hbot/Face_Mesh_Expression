"""
FaceMesh Data Capture Tool

This script captures facial landmark data using MediaPipe's FaceMesh for facial expression classification.
It processes webcam input in real-time to detect and track facial landmarks, allowing users to:
- View real-time facial landmark detection
- Classify facial expressions (neutral, frowning, smile, smile_with_teeth)
- Save landmark coordinates with classifications for machine learning training
- Track data collection progress with sample counts per expression

Usage:
    Run the script and use keyboard controls to capture and classify facial expressions:
    - 'q': Quit application
    - 'f': Freeze/unfreeze frame
    - 's': Save current landmarks with classification
    - 'n': Set classification to "neutral"
    - 'r': Set classification to "frowning"
    - 'm': Set classification to "smile"
    - 'b': Set classification to "smile_with_teeth"

Output:
    Saves facial landmark coordinates (x, y, z) with classifications to CSV file
    Format: [label, landmark000_x, landmark000_y, landmark000_z, ..., landmarkN_z]
"""

import csv
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_LEFT_IRIS,
    FACEMESH_LIPS,
    FACEMESH_NOSE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_RIGHT_IRIS,
)

# Initialize MediaPipe FaceMesh with configuration for detecting and tracking face landmarks
# static_image_mode=True processes each frame independently
# max_num_faces=1 limits detection to one face for efficiency
# refine_landmarks=True enables more precise landmark detection including iris
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1,
)

# Initialize state variables
freeze = False  # Controls whether to process new frames or keep current frame
coordinates = []  # Stores detected facial landmark coordinates
classification = "neutral"  # Current emotion classification label
classification_counts = {}  # Tracks number of saved samples per emotion

# List of facial feature connections for visualization
# These define the lines to draw between landmarks for each facial feature
FACIAL_FEATURES = [
    mp_face_mesh.FACEMESH_LIPS,
    mp_face_mesh.FACEMESH_LEFT_EYEBROW,
    mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
    mp_face_mesh.FACEMESH_LEFT_EYE,
    mp_face_mesh.FACEMESH_RIGHT_EYE,
]


# Draw lines between facial landmarks to visualize facial features
def draw_landmarks(frame, landmarks, connections):
    """
    Draw lines between facial landmarks to visualize feature connections.

    Args:
        frame: numpy.ndarray - Video frame/image to draw on
        landmarks: list - MediaPipe facial landmark points
        connections: list - Pairs of landmark indices to connect with lines

    Returns:
        numpy.ndarray: Frame with landmark connections drawn
    """
    for connection in connections:
        start_idx, end_idx = connection
        pt1 = (
            int(landmarks[start_idx].x * frame.shape[1]),
            int(landmarks[start_idx].y * frame.shape[0]),
        )
        pt2 = (
            int(landmarks[end_idx].x * frame.shape[1]),
            int(landmarks[end_idx].y * frame.shape[0]),
        )
        frame = cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    return frame


# Visualize individual facial landmark points
def draw_points(frame, landmarks, color=(0, 255, 0), radius=2):
    """
    Draw facial landmark points on the frame as colored circles.

    Args:
        frame: numpy.ndarray - Video frame/image to draw on
        landmarks: list - MediaPipe facial landmark points
        color: tuple - RGB color for points (default: green)
        radius: int - Point radius in pixels (default: 2)

    Returns:
        numpy.ndarray: Frame with landmark points drawn
    """
    for landmark in landmarks:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), radius, color, -1)
    return frame


# Save facial landmark data to CSV for machine learning training
def save_coordinates_to_csv(
    coordinates, classification, filename="facemesh_coordinates.csv"
):
    """
    Save facial landmark coordinates and classification to CSV file.
    Creates header row on first write, then appends subsequent data rows.

    Data format:
    - First column: Emotion classification label
    - Subsequent columns: x, y, z coordinates for each landmark

    Args:
        coordinates: list - Tuples of (x,y,z) coordinates for each landmark
        classification: str - Emotion classification label
        filename: str - Output CSV filename (default: facemesh_coordinates.csv)
    """
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            header = ["label"]
            for i in range(len(coordinates)):
                header.extend(
                    [f"landmark{i:03d}_x", f"landmark{i:03d}_y", f"landmark{i:03d}_z"]
                )
            writer.writerow(header)
        row = [classification]
        for coord in coordinates:
            row.extend([coord[0], coord[1], coord[2]])
        writer.writerow(row)


# Future feature: Adjust landmark positions using trackbar values
def adjust_landmarks(landmarks):
    """
    Placeholder for future landmark position adjustment functionality.
    Will use trackbar values to modify landmark positions for data augmentation.

    Args:
        landmarks: list - MediaPipe facial landmark points

    Returns:
        list: Adjusted landmark points (currently returns unmodified landmarks)
    """
    return landmarks


# Callback function for trackbar value changes
# Currently unused but required for trackbar creation
def on_trackbar(val):
    pass


# Create window and trackbars for future landmark adjustment feature
cv2.namedWindow("Frame")
cv2.createTrackbar("Eyebrow", "Frame", 0, 100, on_trackbar)
cv2.createTrackbar("Smile", "Frame", 0, 100, on_trackbar)

# Initialize webcam capture (index 0 for default camera)
cap = cv2.VideoCapture(0)

# Main application loop
# Process webcam frames to detect and visualize facial landmarks in real-time
# Allows for expression classification and data collection through keyboard controls
while cap.isOpened():
    if not freeze:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                landmarks = adjust_landmarks(landmarks)
                # Instead of drawing only specified features, draw all points for now.
                frame = draw_points(frame, landmarks)
                coordinates = [(lm.x, lm.y, lm.z) for lm in landmarks]

    # Display the current classification on the frame
    cv2.putText(
        frame,
        f"Classification: {classification}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # Display a list of saved classification counts
    y_offset = 70
    for cls, count in classification_counts.items():
        text = f"{cls}: {count}"
        cv2.putText(
            frame,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_offset += 30

    cv2.imshow("Frame", frame)

    # Keyboard controls for application:
    # q: Quit the application
    # f: Freeze/unfreeze the current frame
    # s: Save current facial landmarks with classification
    # n: Set classification to "neutral"
    # r: Set classification to "frowning"
    # m: Set classification to "smile"
    # b: Set classification to "smile_with_teeth"
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("f"):
        freeze = not freeze
    elif key == ord("s"):
        save_coordinates_to_csv(coordinates, classification)
        # Increment counter for current emotion classification
        classification_counts[classification] = (
            classification_counts.get(classification, 0) + 1
        )
    elif key == ord("n"):
        classification = "neutral"
    elif key == ord("r"):
        classification = "frowning"
    elif key == ord("m"):
        classification = "smile"
    elif key == ord("b"):
        classification = "smile_with_teeth"

# Cleanup: Release webcam, close windows and face mesh processor
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
