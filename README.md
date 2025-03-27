# llama-cpp-connector

Super simple Python connectors for llama.cpp, including vision models (Gemma 3, Qwen2-VL, QVQ). Compile llama.cpp and run!

## Overview

This project provides lightweight Python connectors to easily interact with llama.cpp models, supporting both standard text models and multimodal vision models (currently Gemma 3, Qwen2-VL, and QVQ). It creates a simple framework to build applications on top of llama.cpp while handling the complexity of model configuration, server management, and inference.

The idea behind it is simple: to offer a minimalistic environment for Python coders to directly interact with llama.cpp without intermediaries like Ollama or LMStudio. Just go to HuggingFace, download your models, use the connectors and have fun!

## Features

- 🚀 **Easy to use**: Only two Python classes to interact with local LLMs: `LlamaServerConnector` and `LlamaVisionConnector`
- 🖼️ **Vision model support**: Ready-to-use connectors for Gemma 3, Qwen2-VL, and QVQ vision models
- 🔄 **OpenAI-compatible API**: Use the `LlamaServerConnector` with the OpenAI Python client
- ⚙️ **Configurable**: simple JSON-based configuration for all model parameters
- 🐳 **Docker ready**: Build once, prepare your container, run `docker commit` and your LLM-powered app is done
- 🧠 **Great for pros**: A perfect sandbox for those familiar with llama.cpp!

## Components

### 1. LlamaServerConnector (`llama_server_connector.py`)

This component provides an OpenAI-compatible server interface for text-based models:

- Provides methods to start, manage and kill llama-server instances directly via Python
- Automatically starts and manages a llama.cpp server
- Finds available ports dynamically
- Configures model parameters from JSON
- Provides a simple API to send prompts and get responses
- Compatible with the OpenAI Python client

### 2. LlamaVisionConnector (`llama_vision_connector.py`)

For multimodal vision models (Gemma 3, Qwen2-VL, QVQ):

- Process images with text prompts
- Automatically handles configuration for different vision models
- Asynchronous API for efficient processing
- Supports custom prompts or prompt files

### 3. Docker Build System

Simplifies dependency management and deployment:

- Multi-stage build for optimal image size
- Automatically detects CUDA architecture
- Sets up all required dependencies
- Creates a ready-to-use environment with Python and libraries

## How to Use

### Installation

#### 1. Docker Container (Recommended)

The easiest way to get started is with the Docker container. 

A big part of why I built this was to have a _very_ simple llama.cpp/Python sandbox that could follow the new releases of llama.cpp repository faster than bindings such as llama-cpp-python does.

The simple Dockerfile here should do the trick. (Important note: the Dockerfile assumes you are using NVIDIA GPUs.) 

```bash
# Clone the repository
git clone https://github.com/yourusername/llama-cpp-connector.git
cd llama-cpp-connector

# Run the build script
chmod +x build-docker-container.sh
./build-docker-container.sh

# Run the container with GPU access
docker run --gpus all -it llama-cpp-connector:latest
```

The `build-docker-container.sh` script will automatically detect the CUDA compute capability of your GPU and compile llama.cpp specifically for your architecture, so the compilation is as fast as possible and the container is of minimal size.

#### 2. Manual Installation

If you prefer not to use Docker:

1. Compile llama.cpp from source and make sure all binaries are in your PATH
   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   mkdir build && cd build
   cmake .. -DLLAMA_CUDA=ON
   cmake --build . --config Release
   # Add binaries to your PATH
   ```

2. Install Python dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Using the Server Connector

```python
from llama_server_connector import LlamaServerConnector

# Initialize the server with a text model
connector = LlamaServerConnector(
    config_path="config/models.json",
    model_key="DEEPSEEK-R1-QWEN-14B"
)

# Get a response
##   Note: the get_response method only provides the response string.
##   If you want a OpenAI completions object, just send a openai request to the server at http://{host}:{self.urlport}/v1
response = connector.get_response("Explain quantum computing in simple terms")
print(response)

# When done
connector.kill_server()
```

### Using the Vision Connector

```python
import asyncio
from llama_vision_connector import LlamaVisionConnector

async def process_image():
    # Initialize with a vision model
    vision = LlamaVisionConnector(
        config_path="config/models.json",
        model_key="GEMMA3_12B"
    )
    
    # Process an image with default or custom prompt
    ##   Note: the get_response method only provides the response string.
    ##   Vision models in llama.cpp are currently handled via the CLI interface, so this is NOT an openai-compatible interaction.
    description = await vision.get_response(
        "path/to/image.jpg", 
        prompt="Describe this image in detail"
    )
    
    print(description)

# Run the async function
asyncio.run(process_image())
```

## Configuration

Models are configured in `config/models.json`:

```json
{
    "MODELS": {
        "QVQ_72B_PREVIEW": {
            "CLI_CMD": "llama-qwen2vl-cli",
            "MODEL_PATH": "models/QVQ-72B-Preview-GGUF",      
            "MMPROJ_PATH": "models/mmproj-QVQ-72B-Preview-f16.gguf",
            "TEMPERATURE": 0.3,
            "NUM_LAYERS_TO_GPU": 99
            // Other parameters...
        },
        "GEMMA3_12B": {
            "CLI_CMD": "llama-gemma3-cli",
            "MODEL_PATH": "models/gemma-3-12b-it-Q6_K_L.gguf",      
            "MMPROJ_PATH": "models/mmproj-gemma3-12b-it-f32.gguf",
            "TEMPERATURE": 0.3,
            "NUM_LAYERS_TO_GPU": 99
            // Other parameters...
        },
        // Additional models...
    }
}
```

## Docker Container

While optional, the Docker container is the heart of this project, providing:

1. **Automatic CUDA detection**: Builds with the optimal settings for your GPU
2. **Compiled binaries**: Builds llama.cpp from source with all optimizations
3. **Python environment**: Pre-configured with all (minimal) required libraries
4. **Ready-to-use framework**: Just add your models and start developing

### Why Use the Docker Container?

- **Eliminates dependency issues**: All libraries and tools are pre-installed
- **GPU-optimized**: Builds specifically for your GPU architecture
- **Reproducible**: Same environment on any machine
- **Easy model management**: Just mount your model folder
- **Portable**: Run anywhere Docker and CUDA are supported
- **Always pulls latest llama.cpp**: No more waiting for llama-cpp-python to update!

To use your own models, just mount your models directory:

```bash
docker run --gpus all -v /path/to/your/models:/workspace/models -it llama-cpp-connector:latest
```

## Adding Models

1. Place your GGUF model files in the `models/` directory
2. Update `config/models.json` with your model configuration
3. Use the appropriate connector based on model type (text or vision)

## Why llama-cpp-connector?

- **Simplified integration**: No need to directly interface with llama.cpp's C++ code
- **Simplified configuration**: Change model behavior just by editing a JSON script
- **Docker-first approach**: Consistent environment across development and deployment
- **Minimal dependencies**: Just have llama.cpp binaries and you're done
- **OpenAI-compatible**: Use familiar APIs with local models

## License

[MIT License](LICENSE)
