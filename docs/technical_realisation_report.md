

**Embedded Systems Engineering**

**Product report**

![A face of a person

Description automatically generated](data:image/png;base64...)

*Facemesh facial expression detection*

**Embedded Systems Engineering
Academy Engineering and Automotive
HAN University of Applied Sciences**

Authors

| ***154814821888881671261 1659237*** | ***Jaap Jan GroenedijkKevin van HoeijenEnes ÇaliskanWilliam Hak*** |
| --- | --- |

Date

| ***January 2025*** |
| --- |

Version

| **0.2** |
| --- |

# Revisions

| **Version** | **When** | **Who** | **What** |
| --- | --- | --- | --- |
| 0.1 | 14-12-2024 | J.J.Groenendijk | Technical Design init |
| 0.2 | 12-1-2025 | K van Hoeijen | Front page, introduction and functional design |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

# Preface

*Looking back on the course of the project, how did the project group experience the project, what did the group learn from it, what will the group do better next time; in short, the preface can contain all kinds of personal reflections on the project and its course.*

During the project, the team encountered several challenges. Some team members did not contribute adequately, leading to an uneven distribution of workload. The high workload during the semester also contributed to this issue, with the majority of the work falling on one or two group members. In some instances, team members committed to completing code but failed to deliver, further hindering progress.

The team also learned several lessons about embedded systems development. One key takeaway was the importance of having a clear, defined vision and goal for the project to achieve results faster. Some team members also discovered that Python and machine learning can be enjoyable. [UNCLEAR: Some team members expressed a preference for working solo rather than in groups, but this may not be appropriate to include in a formal report.]

In future projects, the team would prioritize removing members who do not contribute adequately to ensure a more balanced and productive workload distribution.

# Summary

*Description of the starting points, goals to be achieved, what was and was not achieved,* ***results achieved****; the summary should give an overall impression of the whole assignment and is maximum 1 A4.*

The original success criteria/KPIs for the project were defined in the requirements (see Chapter 2). Due to a lack of time, the team was unable to achieve all of the unmet objectives.

# Testing

[UNCLEAR: This section needs to be filled in with details about the testing process, including the testing framework, procedures, and results. Did the tests meet the functional and technical specifications? What test setup was used and what were the final results? The results are accompanied by a clear description of any remaining problems and how they might be explained. Were any 'work arounds' performed during testing? The tests must be described in such a way that each test can be reproduced by others.] The software runs fine on an old Macbook, but other technical requirements remain to be tested.

# Contents

[Revisions 2](#_Toc190277315)

[Preface 3](#_Toc190277316)

[Summary 4](#_Toc190277317)

[Contents 5](#_Toc190277318)

[1 Introduction 7](#_Toc190277319)

[1.1 Reason 7](#_Toc190277320)

[1.2 Objective 7](#_Toc190277321)

[1.3 Report structure 7](#_Toc190277322)

[2 Functional design 8](#_Toc190277323)

[2.1 Functional specifications 8](#_Toc190277324)

[2.2 Technical specifications 9](#_Toc190277325)

[2.3 User interface 9](#_Toc190277326)

[3 Technical Design 10](#_Toc190277327)

[3.1 Host hardware 10](#_Toc190277328)

[3.2 Host software 10](#_Toc190277329)

[3.3 Architecture 10](#_Toc190277330)

[3.3.1 Application container 10](#_Toc190277331)

[3.4 System Flow 10](#_Toc190277332)

[3.4.1 Technology Stack 10](#_Toc190277333)

[3.4.2 Project Structure 11](#_Toc190277334)

[3.5 Interfaces 12](#_Toc190277335)

[3.5.1 Power Supply 12](#_Toc190277336)

[3.5.2 USB Interface: PC-Camera 12](#_Toc190277337)

[3.5.3 ROS 2 Communication 13](#_Toc190277338)

[4 Realisation 14](#_Toc190277339)

[4.1 Hardware 14](#_Toc190277340)

[4.2 Software 14](#_Toc190277341)

[5 Testing 15](#_Toc190277342)

[6 Conclusions and recommendations 16](#_Toc190277343)

[7 References 17](#_Toc190277344)

[Appendix A 19](#_Toc190277345)

[Appendix B 20](#_Toc190277346)

[Appendix n 21](#_Toc190277347)

# Introduction

This project is conducted to realise a product that is able to process image data and predict what kind of expression is shown at a person's face.

## Reason

Whilst the shortage of trained professionals rises, the demand for good quality care keeps rising. To reduce the workload of care professionals our team was asked to provide a solution that requires less human intervention, so that professional care can be given to whom who needs it the most while mitigating the loss of costly human checkups.

## Objective

Our objective is to feed back a single stream of human facial expressions back to an interface so that that information, and that information only, can be used to trigger a checkup on a patient. This wile guaranteeing the privacy of the patient in question, by not storing any visual data of the video stream.

This goal is achieved by using 3D-coordinates provided by the FaceMesh model by Mediapipe.

## Report structure

The project document is oorganized as followed;

**Chapter 2:** Provides an overview of the functional design, what the end product should and should not do. Specified by the SMART and MoSCoW method.

**Chapter 3:** Outlines the technical design, detailing the hardware and software specifications, system architecture, technology stack, project structure, and interfaces.

**Chapter 4:** Describes the implementation phase, focusing on the development process, tools, and methodologies used to build the system.

**Chapter 5:** Explains the system testing and validation process, including the testing framework, procedures, and results.

**Chapter 6:** Discusses the evaluation of the project, highlighting performance analysis, system limitations, and areas for improvement.

#

# Functional design

The function design chapter servers as a blueprint for the development of the system. Here lies the functional and technical aspects of what the system must, would, should and will not do.

The requirements are, when applicable, noted in the SMART method.

In this chapter, we define detailed functionalities as well as optional features that provide usability and integration. By sorting the requirements hierarchically it provides a structural approach of realising the project whilst maintaining flexibility for future enhancements.

## Functional specifications

| **SMART functional specification** | | |
| --- | --- | --- |
| **#** | **MoSCoW** | **Description** |
| **F1** | **M** | **The system will process a live video stream in real-time.** |
| F1.1 | M | The system will render a Face Mesh on the detected face during analysis. |
| F1.2 | M | The system will detect and process eyebrows and mouth movements to recognize facial expressions. |
| **F2** | **M** | **The system will detect and track changing facial expressions sequentially within a single video stream automatically.** |
| F2.1 | M | The system will recognize: “Neutral” “Smiling” “Frowning” and “Big smile with visible teeth” |
| F2.1.1 | M | The system will recognize different specified facial expressions with an accuracy of at least 90% |
| **F3** | **M** | **The system logs facial expressions in a file** |
| **F4** | **M** | **The system logs facial expressions through an API** |
| F4.1  F4.2 | S  S | The system uses Flask as API  The system uses ROS-2 as API |
| **F4** | **M** | **The system will detect when there is no face present** |
| **F5** | **M** | **The system will project the FaceMesh on the face in the video stream for live feedback** |
| **F6** | **M** | **The system provides the accuracy of the prediction in the videostream and in the output** |
| **F7** | **C** | **The system detects when a face is not straight for the camera** |
| **F7** | **W** | **The system provides a possibility to save the videostream** |

## Technical specifications

| **SMART technical specification** | | |
| --- | --- | --- |
| **#** | **MoSCoW** | **Description** |
| **T1** | **M** | **The system will work on a generic laptop** |
| T1.1 | M | The system will work on Ubuntu operating system |
| T1.2 | M | The system will work on OUMAX Mini PC Intel N100 16GB 500GB or Max N95 8GB 256GB hardware. |
| **T2** | **M** | **The system will operate effectively up to 2 meters away from the camera.** |
| **T3** | **M** | **The system will operate effectively up to 2 meters away from the camera.** |
| **T4** | **M** | **The system will work under diffused LED lighting conditions.** |
| **T5** | **M** | **The system will remain operational when a face is tilted up to 45 degrees relative to the camera.** |
| **T6** | **M** | **The system will process facial expression changes with a delay of no more than 1000 ms.** |
| **T7** | **M** | **The system will visually display emotion scores with a delay of no more than 1000 ms.** |
| **T8** | **S** | **The system should utilize hardware acceleration via a TPU** |
| **T9** | **S** | **The system should utilize TensorFlow Lite for microcontroller implementations** |
| **T10** | **C** | **The system could support various input resolutions for training and live feed** |
| **T11** | **C** | **The system could be developed in C++** |

## User interface

Debug user interface for checking video streaming and prediction of emotions. Inputs are video streams only from the built-in or plugged in webcam of the operating system it is running on. From here the only user interface that is usable on UI level is as is below:

![A screenshot of the web application showing the original and manipulated video feeds.](data:image/png;base64...)
The web application displays the original video stream from the webcam on the left and the manipulated video feed with real-time facial landmarks overlaid on the right.

The used face mesh landmarks are drawn on the face in the video input, and the emotion and the emotion score is displayed

![A screenshot of the web application displaying the detected emotion and confidence score.](data:image/png;base64...)
The web application displays the detected emotion and confidence score in the top-right corner of the manipulated video feed.

#

# Technical Design

## Host hardware

The client for the Facemesh project already defined which hardware is preferred. The system will run on a single board computer (SBC) with a low power CPU. System specifications of the host is as follows:

| CPU | Intel N100, 4 cores/threads, max turbo: 3.4GHz, Maximum Memory Speed  4800 MHz |
| --- | --- |
| PSU | 12V, 3.75A |
| RAM | 8 GB |
| Storage | 128 GB |

## Host software

Ubuntu LTS will be used as the host OS. The latest stable version of the docker engine is used as container engine.

## Architecture

The Facemesh solution consists of multiple microservices that are hosted in containers. Docker is used as container engine, and the orchestration of the services is done with a single Docker Compose file. Docker is used to make the software more portable. Splitting software in multiple services is not an active goal in the Facemesh project.

### Application container

The application container is based on a minimal python container. The application has the following functions

* Hosts a Flask server.
* Streams webcam video feeds to the backend.
* Processes video frames using OpenCV.
* Displays processed frames on the client-side web interface

The api container does the following:

* Connect to the application container
* Collect system logs
* Display the logs on a website hosted within the api container.
* Publish the logs on ROS2 protocol

## System Flow

Steps in the system:

The user accesses the Flask app through http://localhost:9009.

Webcam video feeds are streamed and processed in real-time using OpenCV and FaceMesh.

System logs are generated by the API container and displayed at http://localhost:9010.

### Technology Stack

* Programming Languages: Python
* Frameworks: Flask, OpenCV, Mediapipe
* Containerization: Docker, Docker Compose
* Protocols: HTTP, ROS2 for logging

### Project Structure

The project is organized into several directories and files:

1. **.vscode**
   * tasks.json: Defines tasks for building and running Docker containers.
2. **api**
   * app.py: Main API application file.
   * Dockerfile: Dockerfile for building the API container.
   * requirements.txt: Python dependencies for the API.
   * ros\_entrypoint.sh: Entrypoint script for the Docker container.
   * templates/
     + index.html: Main HTML template.
3. **application**
   * app.py: Main application file.
   * Dockerfile: Dockerfile for building the application container.
   * frame\_editor.py: Contains functions for manipulating video frames.
   * requirements.txt: Python for the application.
   * templates/
     + index.html: Main HTML template.
4. **Data**
   * Contains data-related files and directories (ignored in this documentation).
5. **dataset**
   * Contains dataset-related files and directories (ignored in this documentation).
6. **docker-compose.yml**
   * Docker Compose configuration file.
7. **output**
   * Contains output files and directories.
8. **pipeline**
   * fileHandler.py: Handles file operations.
   * imagePrep.py: Contains functions for image preprocessing and landmark detection.
9. **README.md**
   * Project README file.

## Interfaces

For each interface in the system, we describe its electrical and data communication properties. These specifications are determined based on technical requirements and design choices. Below, we outline the specifications and design considerations for the power supply, USB interface (PC-Camera), and ROS 2 communication.

## 3.5.1 Power Supply

The system is powered by a mini-PC with a built-in Intel Alder Lake N100 processor and an ASRock N100DC-ITX motherboard, designed for low power consumption and stable operation. The power supply ensures sufficient and reliable power delivery to all components.

**Specification:**

* **Voltage Requirements:**
  + AOOSTAR Mini-PC: 19V DC input via external power adapter.
  + Peripheral Devices: 5V or 3.3V (USB-powered or onboard power regulation).
* **Power Source:**
  + The system uses an external EU power adapter with a 19V DC output for the mini-PC.
  + Peripheral devices such as the USB camera draw power directly from the mini-PC’s USB 2.0 or 3.0 ports.
* **Maximum Current:**
  + The total system power budget must accommodate up to 65W (based on the mini-PC's power rating and connected peripherals).

## 3.5.2 USB Interface: PC-Camera

The USB interface connects the camera to the PC, enabling real-time video capture for the Face Mesh application. This interface also serves as the primary communication channel for transferring image data to the microcontroller for processing.

**Specification:**

* **Communication Protocol:** USB 2.0 High Speed (480 Mbps).
* **Driver Requirements:** The USB driver ensures reliable data transfer between the camera and the PC, enabling consistent frame rates for video streaming.
* **Voltage Levels:** The USB interface operates at 5V, delivering both data and power to the camera module.
* **Frame Rate and Resolution:** The camera supports up to 30 FPS at a resolution of 1280x720 pixels, meeting the requirements of the Face Mesh algorithm.

## 3.5.3 ROS 2 Communication

The system utilizes ROS 2 (Robot Operating System) to facilitate data exchange between the microcontroller and the PC. ROS 2 serves as the middleware for processing and publishing the detected facial expression data to other modules.

**Specification:**

* **Communication Medium:** Ethernet or Wi-Fi, depending on the system’s deployment environment.
* **Message Format:** The data is exchanged using ROS 2 messages in a standard format. For example, detected facial landmarks and expressions are published as sensor\_msgs or custom-defined message types.
* **Frequency:** The microcontroller publishes data at a rate of 10 Hz, ensuring near real-time updates.
* **Middleware Layer:** ROS 2’s DDS (Data Distribution Service) ensures reliable communication and supports Quality of Service (QoS) policies to handle varying network conditions.

## 3.5.4 Frontend

The frontend interface provides a real-time display of facial expression detection through a locally hosted web application (accessible via http://localhost:9009) running in Docker. The interface is designed to visualize both the original and manipulated video feeds side-by-side while displaying detected emotions.

1. **Interface Layout**

* **Left Panel:** Displays the original video stream from the webcam, processed via WebRTC.
* **Right Panel:** Shows the manipulated video feed with real-time facial landmarks overlaid using MediaPipe FaceMesh.
* **Emotion Label:** The detected emotion (e.g., 'Smiling,', ‘Smiling with teeth’, 'Frowning,' or 'Neutral') appears in the top-right corner of the manipulated video.
* **Accuracy Score:** Displays the confidence score of the detected expression.

1. **Functionality**

* **Real-Time Landmark Detection:** The application processes the video stream frame-by-frame using MediaPipe FaceMesh.
* **Dual Video Feed:** Displays both original and processed streams simultaneously.
* **Live WebSocket Logs:** Forwards detection logs to the API service (http://localhost:9010) using WebSockets and ROS2.
* **API Integration:** Logs are published to the ROS2 network and made available via REST endpoints.

#

# Realisation

*Details of the realized hardware and software with accompanying explanations and calculations (such as power consumption, values of components, etc.). Complete and detailed diagrams of the hardware and listings of the software are included in the appendices.*

## Hardware

## Software

*The realized software is explained by means of code snippets. Make sure the code is easy to read by using syntax highlighting. Use code snippets that are no longer than 20 lines and that each line of code fits on one line in the report. Not all of the realized code needs to be explained. Choose two or three of the most relevant subsystems. The full code is included as an appendix. Also pay attention to the software development environment. In doing so, ask yourself what is important information for a fellow engineer who will be using the same development environment for the first time.*

### Flask App Setup (`application/app.py`)

This code snippet shows how the Flask app is set up to receive video frames and send them to the `manipulate_frame` function:

```python
from flask import Flask, request, Response, render_template
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/video_feed', methods=['POST'])
def video_feed():
    # Receive the image from the request
    image = request.data
    if not image:
        return Response(status=400, response="Empty image data")

    nparr = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return Response(status=400, response="Failed to decode image")

    # Manipulate the frame
    manipulated_frame = manipulate_frame(frame)

    # Encode the manipulated frame as JPEG
    _, buffer = cv2.imencode('.jpg', manipulated_frame)
    response_data = buffer.tobytes()

    return Response(response_data, mimetype='image/jpeg')
```

This code defines a Flask route `/video_feed` that receives image data from a POST request, decodes it using OpenCV, manipulates it using the `manipulate_frame` function, and returns the manipulated frame as a JPEG image.

### HTTPHandler for Logging (`application/app.py`)

This code snippet shows how the HTTPHandler is used for logging:

```python
import logging
import requests

class HTTPHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        try:
            response = requests.post('http://api:9010', data=log_entry)
        except Exception as e:
            print(f"Failed to send log entry: {e}")

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)

http_handler = HTTPHandler()
http_handler.setLevel(logging.WARNING)
root_logger.addHandler(http_handler)
```

This code defines a custom logging handler that sends log entries to a remote API endpoint using HTTP POST requests.

### Frame Manipulation (`application/frame_editor.py`)

This code snippet shows the `manipulate_frame` function, which uses MediaPipe to detect facial landmarks and predict emotions:

```python
import cv2
import mediapipe as mp
from predict import predict_emotion

# Init media pipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize a list of facial feature classes from MediaPipe face mesh solution
FACIAL_FEATURES = [
    mp_face_mesh.FACEMESH_LIPS,
    mp_face_mesh.FACEMESH_LEFT_EYEBROW,
    mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
]

def draw_landmarks(frame, landmarks, connections):
    for connection in connections:
        start_idx, end_idx = connection
        pt1 = (int(landmarks[start_idx].x * frame.shape[1]), int(landmarks[start_idx].y * frame.shape[0]))
        pt2 = (int(landmarks[end_idx].x * frame.shape[1]), int(landmarks[end_idx].y * frame.shape[0]))
        frame = cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    return frame

def manipulate_frame(frame):
    # Use mediapipe to detect face landmarks
    results = face_mesh.process(frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            # Draw the landmarks for the specified facial features
            for feature in FACIAL_FEATURES:
                frame = draw_landmarks(frame, landmarks, feature)
            
            # Predict emotion using the landmarks
            predicted_emotion, confidence = predict_emotion(landmarks)

            # Get text size
            text1 = f"Prediction: {predicted_emotion}"
            text2 = f"Confidence: {confidence:.2f}"
            
            (text_width, text_height), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            # Define text position
            x, y = 50, 50  # Position where text will be placed

            # Draw a black rectangle background
            cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + baseline + 5), (0, 0, 0), -1)

            # Overlay the text
            cv2.putText(frame, text1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text2, (x, y + text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame
```

This code uses the MediaPipe library to detect facial landmarks in a video frame and then draws those landmarks on the frame. It also calls the `predict_emotion` function to predict the emotion being expressed in the frame.

### Landmark Drawing (`application/frame_editor.py`)

This code snippet shows the `draw_landmarks` function, which draws the landmarks on the frame:

```python
def draw_landmarks(frame, landmarks, connections):
    for connection in connections:
        start_idx, end_idx = connection
        pt1 = (int(landmarks[start_idx].x * frame.shape[1]), int(landmarks[start_idx].y * frame.shape[0]))
        pt2 = (int(landmarks[end_idx].x * frame.shape[1]), int(landmarks[end_idx].y * frame.shape[0]))
        frame = cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    return frame
```

This function takes a frame, a list of landmarks, and a list of connections between landmarks as input. It then iterates over the connections and draws a line between the corresponding landmarks on the frame.

### Model Creation (`pipeline/model_train.py`)

This code snippet shows the `create_model` function, which defines the neural network architecture:

```python
import keras
from keras import layers

def create_model(feature_columns, num_classes):
    # Create a input layer for each feature
    inputs = {}
    for feature in feature_columns:
        inputs[feature] = layers.Input(name=feature, shape=(1,))

    # Concatenate the input layers
    concatenated = layers.concatenate(list(inputs.values()))

    # Apply normalization after concatenation
    x = layers.Normalization()(concatenated)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    # Output layer for multi-class classification
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
```

This function defines a neural network model for multi-class classification. It takes a list of feature columns and the number of classes as input, and returns a compiled Keras model.

### Dataset Preparation (`pipeline/model_train.py`)

This code snippet shows the `prepare_datasets` function, which prepares the training and validation datasets:

```python
import tensorflow as tf
import keras
from keras import layers

def prepare_datasets(df, batch_size=16, train_split=0.8, random_seed=42):
    df = df.copy()

    # Create and adapt label encoder
    label_encoder = layers.StringLookup(output_mode="int")
    label_encoder.adapt(df["class"])

    # Extract labels and encode them
    labels = df.pop("class")
    labels = label_encoder(labels)
    
    # Create initial dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    # Split the dataset using Keras utility
    train_ds, val_ds = keras.utils.split_dataset(
        dataset,
        left_size=train_split,
        shuffle=True,
        seed=random_seed
    )
    
    # Now batch the datasets
    train_ds = (
        train_ds
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    return train_ds, val_ds
```

This function prepares the training and validation datasets for the model. It takes a Pandas DataFrame, batch size, train split ratio, and random seed as input. It then encodes the labels, splits the dataset into training and validation sets, and batches the datasets.

### Image Normalization (`pipeline/imagePrep.py`)

This code snippet shows the `normalize_image` function, which applies adaptive histogram equalization to the image:

```python
import cv2

def normalize_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(img)
    
    return norm
```

This function applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to normalize the image. CLAHE is used to improve the contrast of the image, which can help to improve the accuracy of the facial expression recognition model.

### Saving Facemesh Landmarks (`pipeline/imagePrep.py`)

This code snippet shows the `save_facemesh` function, which saves the facemesh landmarks to a CSV file:

```python
import os
import pandas as pd

def save_facemesh(filename, class_name, landmarks, facemesh_csv_path):
    if landmarks:
        for face_landmarks in landmarks:
            row = [filename, hash(filename), class_name]
            for i, landmark in enumerate(face_landmarks.landmark):
                row.extend([landmark.x, landmark.y, landmark.z])
            
            # Define column names
            columns = ['filename', 'hash', 'class']
            for i in range(len(face_landmarks.landmark)):
                columns.extend([f'landmark{i+1:03d}_x', f'landmark{i+1:03d}_y', f'landmark{i+1:03d}_z'])
            
            # Use pandas to write to CSV
            df = pd.DataFrame([row], columns=columns)
            df.to_csv(facemesh_csv_path, mode='a', index=False, header=not os.path.exists(facemesh_csv_path))
```

This function saves the detected facial landmarks to a CSV file. The CSV file is used to train the facial expression recognition model.

### Hardware

The system utilizes the following hardware components:

*   **Mini-PC:** AOOSTAR Mini-PC with Intel Alder Lake N100 processor, 8GB RAM, and 128GB storage. This serves as the main processing unit for the facial expression recognition system.
*   **USB Camera:** A standard USB camera is used to capture video frames of the user's face.

### Model Tuning

[UNCLEAR: Add details about the model tuning process, including the hyperparameters that were tuned and the results of the tuning process.]

# Testing

[UNCLEAR: This section needs to be filled in with details about the testing process, including the testing framework, procedures, and results. Did the tests meet the functional and technical specifications? What test setup was used and what were the final results? The results are accompanied by a clear description of any remaining problems and how they might be explained. Were any 'work arounds' performed during testing? The tests must be described in such a way that each test can be reproduced by others.]

# Conclusions and recommendations

*Reflection on the goals of the project. What are the results? What has been achieved and what has not been achieved? What can be added to, expanded on, improved?*

# References

Adrián Sánchez Cano. (2013, 3 5). *Using RTC module on FRDM-KL25Z.* Retrieved from https://community.nxp.com/docs/DOC-94734

ARM. (2022, 04 26). *µVision® IDE*. Retrieved from https://www2.keil.com/mdk5/uvision/

ARM Developer. (2022, 04 26). *KAN232 - MDK V5 Lab for Freescale Freedom KL25Z Board*. Retrieved from https://developer.arm.com/documentation/kan232/latest/

Berckel, M. v.-v. (2017). *Schrijven voor technici.* Noordhoff Uitgevers B.V.

contributors, W. (2022, 07 06). *MoSCoW method*. (Wikipedia, The Free Encyclopedia) Retrieved 07 06, 2022, from https://en.wikipedia.org/w/index.php?title=MoSCoW\_method&oldid=1091822315

contributors, W. (2022, 05 25). *SMART criteria*. (Wikipedia, The Free Encyclopedia) Retrieved 07 07, 2022, from https://en.wikipedia.org/w/index.php?title=SMART\_criteria&oldid=1089766780

ELECFREAKS. (2022, 04 19). Retrieved from Ultrasonic Ranging Module HC - SR04: https://cdn.sparkfun.com/datasheets/Sensors/Proximity/HCSR04.pdf

Freescale Semiconductor, I. (n.d.). *FRDM-KL25Z Pinouts (Rev 1.0).* Retrieved 3 31, 2023, from https://www.nxp.com/document/guide/get-started-with-the-frdm-kl25z:NGS-FRDM-KL25Z

Freescale Semiconductor, Inc. (2012, 9). *KL25 Sub-Family Reference Manual, Rev. 3.*

Freescale Semiconductor, Inc. (2013, 10 24). *FRDM-KL25Z User's Manual, Rev. 2.0.* Retrieved from https://www.nxp.com/document/guide/get-started-with-the-frdm-kl25z:NGS-FRDM-KL25Z

Freescale Semiconductor, Inc. (2014, 08). *Kinetis KL25 Sub-Family, 48 MHz Cortex-M0+ Based Microcontroller with USB, Rev 5.*

Hmneverl. (2015, 11 18). *De beslismatrix, het maken van keuzes*. (Info.NU.nl) Retrieved 07 06, 2022, from https://mens-en-samenleving.infonu.nl/diversen/164525-de-beslismatrix-het-maken-van-keuzes.html

NXP. (2022, 04 19). *Kinetis® KL2x-72/96 MHz, USB Ultra-Low-Power Microcontrollers (MCUs) based on Arm® Cortex®-M0+ Core*. Retrieved from https://www.nxp.com/products/processors-and-microcontrollers/arm-microcontrollers/general-purpose-mcus/kl-series-cortex-m0-plus/kinetis-kl2x-72-96-mhz-usb-ultra-low-power-microcontrollers-mcus-based-on-arm-cortex-m0-plus-core:KL2x?tab=Buy\_Parametric\_Tab#/

NXP. (2022). *OpenSDA Serial and Debug Adapter*. Retrieved from https://www.nxp.com/design/software/development-software/sensor-toolbox-sensor-development-ecosystem/opensda-serial-and-debug-adapter:OPENSDA?&tid=vanOpenSDA

Solomon Systech Limited. (2008, 4). *SSD1306: Advanced Information.* Retrieved from https://cdn-shop.adafruit.com/datasheets/SSD1306.pdf

Vishay Semiconductors. (2017, 8 9). *TCRT5000(L), Reflective Optical Sensor with Transistor Output, Rev. 1.7*.

# Appendix A

[UNCLEAR: Complete and detailed diagrams of the hardware.]

# Appendix B

Complete code listings for the key software modules:
