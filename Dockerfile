# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL org.opencontainers.image.title="Model Immune System"
LABEL org.opencontainers.image.description="Training-time AI security and resilience system"
LABEL org.opencontainers.image.version="1.0.0"

RUN useradd --create-home --shell /bin/bash mis

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/raw data/processed data/poisoned checkpoints logs plots \
    && chown -R mis:mis /app

USER mis
EXPOSE 8000 8501
CMD ["python", "training/train.py", "--config", "configs/config.yaml"]