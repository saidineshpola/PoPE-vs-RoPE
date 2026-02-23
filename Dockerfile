# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Default: CPU build. For GPU support, modify this section to install CUDA-enabled PyTorch
# Example for GPU (CUDA 11.8):
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
#     pip install --no-cache-dir numpy transformers datasets tiktoken wandb tqdm
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directories for data and output
RUN mkdir -p data out

# Default command (can be overridden in docker-compose)
CMD ["python", "train.py", "--help"]
