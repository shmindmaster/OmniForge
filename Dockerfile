# OmniForge: YOLOv11 Segmentation Template - Production Ready
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY config/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/test_cases data/results models/production

# Expose port for FastAPI (when implemented)
EXPOSE 8000

# Default command - run demo or start interactive shell
CMD ["python", "scripts/pragmatic_nail_finetune.py", "--mode", "quick", "--epochs", "1"]
