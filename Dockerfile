FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code and models
COPY . .

# Expose the port
ENV PORT=8000
EXPOSE 8000

# Start the application
CMD ["python", "app.py"]
