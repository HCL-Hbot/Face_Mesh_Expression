import cv2
import mediapipe as mp
import numpy as np
import logging
import time

from predict import predict_emotion, load_model
from predict import get_facial_feature_indices
from predict import FACIAL_FEATURES

# expected_landmark_indices = [384, 385, 386, 387, 388, 133, 390, 7, 263, 0, 267, 13, 398, 14, 144, 145, 
#  17, 146, 269, 270, 402, 405, 153, 154, 155, 409, 157, 158, 159, 160, 33, 
#  161, 163, 291, 37, 415, 39, 40, 173, 178, 308, 181, 310, 311, 312, 185, 
#  314, 61, 317, 191, 318, 321, 324, 78, 80, 81, 466, 82, 84, 87, 88, 91, 
#  95, 375, 362, 373, 246, 374, 249, 380, 381, 382]

# expected_landmark_indices = list(range(478))

# Init media pipe face mesh
mp_face_mesh = mp.solutions.face_mesh

# face_mesh = mp_face_mesh.FaceMesh()
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.1, min_tracking_confidence=0.1)

# Get and print dynamically extracted indices
expected_landmark_indices = get_facial_feature_indices()
print("Extracted Landmark Indices:", expected_landmark_indices)

last_log_time   = 0
log_interval    = 2

def normalize_landmarks(landmarks):
    min_x = min(lm.x for lm in landmarks)
    min_y = min(lm.y for lm in landmarks)
    max_x = max(lm.x for lm in landmarks)
    max_y = max(lm.y for lm in landmarks)

    return [( (lm.x - min_x) / (max_x - min_x),
              (lm.y - min_y) / (max_y - min_y),
              lm.z ) for lm in landmarks]

def draw_landmarks(frame, landmarks, connections):
    for connection in connections:
        start_idx, end_idx = connection
        pt1 = (int(landmarks[start_idx].x * frame.shape[1]), int(landmarks[start_idx].y * frame.shape[0]))
        pt2 = (int(landmarks[end_idx].x * frame.shape[1]), int(landmarks[end_idx].y * frame.shape[0]))
        frame = cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    return frame

def draw_points(frame, landmarks, color=(0, 255, 0), radius=2):
    for landmark in landmarks:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), radius, color, -1)
    return frame

def manipulate_frame(frame, model=None):
    global last_log_time
    # Use mediapipe to detect face landmarks
    results = face_mesh.process(frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            # Draw the landmarks for the specified facial features
            
			
            frame = draw_points(frame, landmarks)
            
            for feature in FACIAL_FEATURES:
                frame = draw_landmarks(frame, landmarks, feature)
            # Initialize default values
            predicted_emotion = "No model loaded"
            confidence = 0.0

            # Only predict emotion if model is loaded
            if model is not None:
                try:
                    # landmarks = normalize_landmarks(landmarks)
                    predicted_emotion, confidence = predict_emotion(landmarks, model, expected_landmark_indices)
                except Exception as e:
                    predicted_emotion = "Prediction error"
                    confidence = 0.0
                    logging.warning(f"Error predicting emotion: {str(e)}")

            # Get text size
            text1 = f"Prediction: {predicted_emotion}"
            text2 = f"Confidence: {confidence:.2f}" if confidence > 0 else "Confidence: N/A"
            
            (text_width, text_height), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            # Define text position
            x, y = 50, 50  # Position where text will be placed

            # Draw a black rectangle background
            cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + baseline + 5), (0, 0, 0), -1)

            # Overlay the text
            cv2.putText(frame, text1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text2, (x, y + text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    else:
        current_time = time.time()
        if current_time - last_log_time >= log_interval:
            logging.warning('No face detected in the frame.')
            last_log_time = current_time

    return frame

if __name__ == "__main__":
    
    # The following code is for testing the functions in this file, you can modify it as needed.
    # set the model here if you want to test the manipulate_frame function
    model_path = "models/current_model.keras"
    model = load_model(model_path)
    # Test the manipulate_frame function
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = manipulate_frame(frame, model)  # Pass manually set model becuase it is for testing
        
        # Overlay text: Press Q to exit
        # frame = cv2.putText(frame, "Press Q to exit", (50, 100), 
                            # cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
        
    cap.release()
    
    cv2.destroyAllWindows()
    
    # Release the face mesh model
    face_mesh.close()
    
    logging.info("Face mesh model released.")
