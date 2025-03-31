# Use Python 3.9 as base image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    vim \
    tmux \
    curl \
    wget \
    python3-dev \
    python3-pip \
    python3-setuptools \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

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
