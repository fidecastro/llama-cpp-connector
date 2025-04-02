# Stage 1: Build environment
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake python3 python3-pip python3-venv git pkg-config libcurl4-openssl-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone llama.cpp repository instead of copying local files
RUN git clone https://github.com/ggml-org/llama.cpp.git .

# This ARG will be passed at build time
ARG CUDA_ARCHITECTURES="89"

# Configure build with CUDA architecture passed at build time
RUN mkdir -p build && \
    cd build && \
    cmake .. \
        -DLLAMA_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    cmake --build . --config Release -j $(nproc)

# Collect build artifacts
RUN mkdir -p /app/artifacts/libs && \
    find build -name "*.so*" -exec cp {} /app/artifacts/libs \; && \
    cp build/bin/* /app/artifacts/

# Stage 2: Runtime environment
FROM ${BASE_CUDA_RUN_CONTAINER}

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y libcurl4-openssl-dev libgomp1 python3 python3-pip python3-venv python3-requests && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Add virtualenv activation to .bashrc so it's activated in interactive shells
RUN echo 'source /opt/venv/bin/activate' >> /root/.bashrc

# Copy requirements file into the image
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies from requirements.txt in the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt # Clean up requirements file

# Copy artifacts from build stage
COPY --from=build /app/artifacts/libs/* /usr/local/lib/
COPY --from=build /app/artifacts/* /usr/local/bin/

# Set up library path
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Create workspace directory structure
RUN mkdir -p /workspace/config /workspace/models /workspace/examples/test_images

# Copy Python connector files, config folder, and examples to workspace
COPY llama_vision_connector.py llama_server_connector.py /workspace/
COPY config/ /workspace/config/
COPY models/ /workspace/models/
COPY examples/ /workspace/examples/

# Set workspace as working directory
WORKDIR /workspace

# Create a shell script to activate the virtual environment when the container starts
RUN echo '#!/bin/bash\necho "Activating Python virtual environment..."\nsource /opt/venv/bin/activate\nPS1="(venv) \u@\h:\w\\$ "\nexec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh", "/bin/bash", "-l"]