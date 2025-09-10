# Multi-stage build for im2fit FastAPI application
# Stage 1: Base Python image with system dependencies
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM base as production

# Copy application code
COPY app/ ./app/
COPY model/ ./model/

# Create necessary directories
RUN mkdir -p /app/static /app/templates

# Copy startup script
COPY app/startup.txt ./startup.txt

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start command using startup.txt configuration
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 4 --worker-class uvicorn.workers.UvicornWorker --timeout 120 --preload app.main:app"]

# Development stage
FROM base as development

# Install additional development dependencies
RUN pip install --no-cache-dir pytest pytest-cov

# Copy all source code for development
COPY . .

# Set development environment
ENV FLASK_ENV=development
ENV DEBUG=True

# Development command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]