# API Service for FaceMesh Data

This directory contains the API service responsible for receiving FaceMesh data, publishing it to a ROS2 network, and providing access to logs.

## Overview

The API is built using Flask and Flask-SocketIO. It exposes a REST endpoint for receiving POST requests containing FaceMesh data. Upon receiving data, the API publishes it to a ROS2 topic named `facemesh_data` and emits a WebSocket event for real-time log updates.

## Files

*   `Dockerfile`: Defines the Docker image for the API service.
*   `app.py`: Contains the Flask application, ROS2 node, and WebSocket logic.
*   `requirements.txt`: Lists the Python dependencies for the API service.
*   `ros_entrypoint.sh`: Entrypoint script for the Docker container, setting up the ROS2 environment.
*   `templates/index.html`: A basic HTML template for testing the API.

## Docker Instructions

### Building the Image

```bash
docker build -t facemesh_api api/
```

### Running the Container

```bash
docker run -d -p 9010:9010 facemesh_api
```

## API Endpoints

### `/` (POST)

*   **Method:** POST
*   **Description:** Receives FaceMesh data and publishes it to the ROS2 network.
*   **Request Body:** Raw text data containing FaceMesh information.
*   **Response:** "Data published to ROS"

### `/logs` (GET)

*   **Method:** GET
*   **Description:** Retrieves the logs of received FaceMesh data.
*   **Response:** JSON array containing the log data.

## WebSocket

The API uses WebSocket to provide real-time log updates. Clients can connect to the WebSocket endpoint to receive updates whenever new data is published to the ROS2 network.

## ROS2 Integration

The API publishes data to the `facemesh_data` topic using the `std_msgs/String` message type. Make sure your ROS2 environment is properly configured to receive these messages.
