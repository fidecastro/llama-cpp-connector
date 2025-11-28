# Stage 1: Build environment
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake python3 python3-pip python3-venv git pkg-config libcurl4-openssl-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ensure latest code is fetched and submodules are updated
RUN rm -rf .git && \
    echo "Cloning latest llama.cpp..." && \
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git . && \
    echo "Updating submodules..." && \
    git submodule update --init --recursive

# This ARG will be passed at build time
ARG CUDA_ARCHITECTURES="89"

# Configure and build llama.cpp
RUN echo "Configuring and building llama.cpp..." && \
    mkdir -p build && \
    cd build && \
    cmake .. \
    -DLLAMA_CUDA=ON \
    # You generally don't need BUILD_SHARED_LIBS=ON for llama.cpp unless you have specific needs
    # The necessary .so files seem to be built by default now when CUDA is ON.
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    cmake --build . --config Release -j $(nproc)

# --- Final Artifact Collection ---
# Copy all artifacts from the build/bin directory
RUN echo "Collecting build artifacts from /app/build/bin/..." && \
    rm -rf /app/artifacts && \
    mkdir -p /app/artifacts && \
    cp -v /app/build/bin/* /app/artifacts/

# Stage 2: Runtime environment
FROM ${BASE_CUDA_RUN_CONTAINER}

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y libcurl4 libgomp1 python3 python3-pip python3-venv python3-requests && \
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

# --- Final Artifact Copying ---
# Copy shared libraries (.so) to /usr/local/lib
COPY --from=build /app/artifacts/*.so* /usr/local/lib/
# Copy everything else (executables) to /usr/local/bin
COPY --from=build /app/artifacts/* /usr/local/bin/
# Clean up any .so files mistakenly copied to /usr/local/bin
RUN find /usr/local/bin -maxdepth 1 -name '*.so*' -delete

# Set up library path and ensure ldconfig recognizes new libs
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
RUN ldconfig

# Create workspace directory structure
RUN mkdir -p /workspace/config /workspace/models /workspace/examples/test_images

# Copy Python connector files, config folder, and examples to workspace
COPY llama_cli_connector.py llama_server_connector.py /workspace/
COPY config/ /workspace/config/
COPY models/ /workspace/models/
COPY examples/ /workspace/examples/

# Set workspace as working directory
WORKDIR /workspace

# Create a shell script to activate the virtual environment when the container starts
RUN echo '#!/bin/bash\necho "Activating Python virtual environment..."\nsource /opt/venv/bin/activate\nPS1="(venv) \u@\h:\w\\$ "\nexec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh", "/bin/bash", "-l"]
