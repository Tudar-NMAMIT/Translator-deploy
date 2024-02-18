# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the application code into the container
COPY . .

# Install any dependencies needed by the application
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
