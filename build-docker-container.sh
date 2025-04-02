#!/bin/bash

# Detect CUDA compute capability using nvidia-smi
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)

# Check if detection was successful
if [ -z "$COMPUTE_CAP" ]; then
    echo "Error: Could not detect CUDA compute capability."
    echo "Using default value of 89 (RTX 4090)."
    CUDA_ARCH="89"
else
    # Remove the dot from the compute capability (e.g., 8.9 -> 89)
    CUDA_ARCH=${COMPUTE_CAP//./}
    echo "Detected CUDA architecture: $COMPUTE_CAP (using $CUDA_ARCH for build)"
fi

# Build the Docker image with the detected architecture
docker build \
    --build-arg CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -t llama-cpp-connector:latest \
    -f Dockerfile .

echo "Build complete!"
echo "You can run the container with: docker run --gpus all -it llama-cpp-connector:latest"