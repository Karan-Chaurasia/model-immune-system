# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL org.opencontainers.image.title="Model Immune System"
LABEL org.opencontainers.image.description="Training-time AI security and resilience system"
LABEL org.opencontainers.image.version="1.0.0"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mis

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary runtime directories
RUN mkdir -p data/raw data/processed data/poisoned checkpoints logs \
    && chown -R mis:mis /app

USER mis

# Expose ports
EXPOSE 8000 8501

# Default command: run training then serve the API
CMD ["python", "training/train.py", "--config", "configs/config.yaml"]
