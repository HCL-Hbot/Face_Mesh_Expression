"""
FaceMesh API Server Module

This module implements a Flask-based web server that integrates with ROS2 to handle FaceMesh data.
It provides a bridge between web clients and ROS2 nodes, enabling real-time facial mesh data
transmission and visualization.

Features:
- RESTful API endpoints for data submission and log retrieval
- WebSocket support for real-time updates
- ROS2 publisher node for data distribution
- Web interface for visualization

Dependencies:
- ROS2 (rclpy)
- Flask
- Flask-SocketIO
- logging

Usage:
    Run the server:
    $ python app.py

    The server will start on port 9010 and create a ROS2 publisher node
    that publishes received facial mesh data to the 'facemesh_data' topic.
"""

# Standard library imports
import logging
import threading

# Third-party imports
import rclpy
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit
from rclpy.node import Node
from std_msgs.msg import String

app = Flask(__name__)
socketio = SocketIO(app)
logs = []


class ROS2Node(Node):
    """A ROS2 node for publishing FaceMesh data.

    This node publishes facial mesh data received from the web interface
    to the ROS2 network, allowing other ROS2 nodes to consume and process
    the data.

    Attributes:
        publisher_ (Publisher): ROS2 publisher for the 'facemesh_data' topic
    """

    def __init__(self) -> None:
        """Initialize the ROS2 node and set up the publisher."""
        super().__init__("facemesh_publisher")
        self.publisher_ = self.create_publisher(String, "facemesh_data", 10)

    def publish_data(self, data: str) -> None:
        """Publish data to the ROS2 network.

        Args:
            data: The facial mesh data to publish
        """
        msg = String()
        msg.data = data
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{data}"')


def ros2_spin(node: Node) -> None:
    """Spin the ROS2 node in a separate thread.

    Args:
        node: The ROS2 node to spin
    """
    rclpy.spin(node)


@app.route("/", methods=["POST", "GET"])
def handle_request():
    """Handle both GET and POST requests to the root endpoint.

    GET: Returns the main visualization interface
    POST: Accepts facial mesh data and publishes it to ROS2

    Returns:
        str: Success message for POST requests
        template: Rendered HTML template for GET requests
    """
    if request.method == "POST":
        data = request.get_data(as_text=True)
        if not data:
            app.logger.warning("No data received")
            return "No data received", 400

        app.logger.info(f"Received data: {data}")
        logs.append(data)
        ros2_node.publish_data(data)
        socketio.emit("log_update", data)
        return "Data published to ROS", 200

    return render_template("index.html")


@app.route("/logs", methods=["GET"])
def get_logs():
    """Retrieve the history of received facial mesh data.

    Returns:
        Response: JSON array of historical log entries
    """
    return jsonify(logs), 200


@socketio.on("connect")
def handle_connect():
    """Handle new WebSocket connections.

    Sends all historical log entries to newly connected clients
    to synchronize their state with the server.
    """
    for log in logs:
        emit("log_update", log)


if __name__ == "__main__":
    # Configure logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    # Initialize ROS2 node and start it in a separate thread
    rclpy.init(args=None)
    ros2_node = ROS2Node()
    ros2_thread = threading.Thread(target=ros2_spin, args=(ros2_node,), daemon=True)
    ros2_thread.start()

    try:
        # Start the Flask-SocketIO server
        socketio.run(app, host="0.0.0.0", port=9010, allow_unsafe_werkzeug=True)
    finally:
        # Ensure proper cleanup of ROS2 resources
        ros2_node.destroy_node()
        rclpy.shutdown()
