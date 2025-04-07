from flask import Flask, request, Response, render_template, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import logging
import requests
import os
from frame_editor import manipulate_frame
from predict import load_model, validate_model

# Create upload directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model instance
current_model = None

class HTTPHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        try:
            print(f"HTTPHandler: Sending log entry: {log_entry}")  # Debugging statement
            response = requests.post('http://api:9010', data=log_entry)  # Use the service name 'api'
            print(f"HTTPHandler: Response status code: {response.status_code}")  # Debugging statement
            print(f"HTTPHandler: Response text: {response.text}")  # Debugging statement
        except Exception as e:
            print(f"HTTPHandler: Failed to send log entry: {e}")  # Debugging statement

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)

http_handler = HTTPHandler()
http_handler.setLevel(logging.WARNING)
root_logger.addHandler(http_handler)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_model', methods=['POST'])
def upload_model():
    global current_model
    
    if 'model' not in request.files:
        return jsonify({'error': 'No model file provided'}), 400
    
    file = request.files['model']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.keras'):
        return jsonify({'error': 'Invalid file format. Please upload a .keras file'}), 400
    
    # Save the uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, 'current_model.keras')
    file.save(filepath)
    
    # Try to load and validate the model
    try:
        model = load_model(filepath)
        
        if not validate_model(model):
            os.remove(filepath)
            return jsonify({'error': 'Invalid model structure'}), 400
        
        current_model = model
        return jsonify({'message': 'Model uploaded successfully'}), 200
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error loading model: {str(e)}'}), 400


@socketio.on('video_frame')
def handle_frame(frame_data):
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return

    # Manipulate the frame
    manipulated_frame = manipulate_frame(frame, current_model)

    # Encode the manipulated frame as JPEG
    _, buffer = cv2.imencode('.jpg', manipulated_frame)
    response_data = buffer.tobytes()

    # Emit the processed frame back to the client
    emit('processed_frame', response_data)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=9009, debug=True, allow_unsafe_werkzeug=True)
