# Backend-only Dockerfile for SET Game Detector API
FROM python:3.10

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

# Install Python dependencies in stages to avoid memory issues
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel setuptools

# Install dependencies in separate steps to help with debugging
RUN pip install --no-cache-dir numpy==1.25.2 pandas==2.1.1 pydantic==2.4.2
RUN pip install --no-cache-dir fastapi==0.104.1 python-multipart==0.0.6 uvicorn[standard]==0.23.2 aiofiles==23.2.1
RUN pip install --no-cache-dir pillow==10.0.1 opencv-python-headless==4.8.1.78
RUN pip install --no-cache-dir tensorflow-cpu==2.13.0
RUN pip install --no-cache-dir torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir ultralytics==8.0.211

# Create model directories
RUN mkdir -p /app/models/Card/16042024 \
    /app/models/Characteristics/11022025 \
    /app/models/Shape/15052024

# Copy model files
COPY models/Card/16042024/best.pt models/Card/16042024/data.yaml /app/models/Card/16042024/
COPY models/Characteristics/11022025/fill_model.keras models/Characteristics/11022025/shape_model.keras /app/models/Characteristics/11022025/
COPY models/Shape/15052024/best.pt models/Shape/15052024/data.yaml /app/models/Shape/15052024/

# Copy backend code
COPY main.py .

# Set the command to run the FastAPI app
CMD ["python", "main.py"]
