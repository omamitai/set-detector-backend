# Backend-only Dockerfile for SET Game Detector API
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create model directories
RUN mkdir -p /app/models/Card/16042024 \
    /app/models/Characteristics/11022025 \
    /app/models/Shape/15052024

# Copy model files (these need to exist in the build context)
COPY models/Card/16042024/best.pt models/Card/16042024/data.yaml /app/models/Card/16042024/
COPY models/Characteristics/11022025/fill_model.keras models/Characteristics/11022025/shape_model.keras /app/models/Characteristics/11022025/
COPY models/Shape/15052024/best.pt models/Shape/15052024/data.yaml /app/models/Shape/15052024/

# Copy backend code
COPY main.py .

# Set the command to run the FastAPI app
CMD ["python", "main.py"]
