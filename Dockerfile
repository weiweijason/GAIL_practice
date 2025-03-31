# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir numpy torch torchvision torchaudio gym tqdm

# Copy source files
COPY . /app/src/

# Create directories for models and results
RUN mkdir -p /app/models /app/results

# Set environment variables
ENV PYTHONPATH=/app

# Set the working directory to the src directory
WORKDIR /app/src
