# Use the official ROS image
FROM ros:iron-ros-base

# Label image name
LABEL image_name="ros-iron-ros-base"

# Set the working directory in the container
WORKDIR /application

# Copy only requirements first so dependencies are cached
COPY requirements.txt /application/

# Install python3
RUN apt-get update && apt-get install -y \
python3-pip dos2unix && \
rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /application
COPY . .

# setup entrypoint
COPY ./ros_entrypoint.sh /

# Make the entrypoint script executable
RUN chmod +x /ros_entrypoint.sh
RUN dos2unix /ros_entrypoint.sh

# Expose the port
EXPOSE 9010

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["python3", "app.py"]