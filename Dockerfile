FROM python:3.11-slim

# Install system dependencies for OpenCV and X11
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/

# Copy models (weights)
COPY yolov8n.pt .
COPY yolov8n-pose.pt .
# Assume LSTM and PPE models are in models/
COPY models/ ./models/

# Build frontend and copy dist
# (In a production CI, this would be a multi-stage build, 
# but for now we copy the already built frontend from the host)
COPY frontend/dist/ ./frontend/dist/

# Create data directories
RUN mkdir -p data/clips data/events.db

EXPOSE 8000

# Environment variables for default behavior
ENV VIDEO_SOURCE=0
ENV DETECTION_CONFIDENCE=0.5

CMD ["python", "backend/main.py", "--source", "0", "--port", "8000"]
