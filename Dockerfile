# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files
COPY requirements.txt .
COPY main.py .
COPY aruco_triangle_detection.py .
COPY triangle_detection.py .

RUN pip install uv
# Install Python dependencies
RUN uv pip install --no-cache-dir -r requirements.txt --system

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GEMINI_API_KEY="AIzaSyBMZshYv40LxHWivZdoRQfR1Z6aGZddzC8"

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
