# llama-cpp-connector

Super simple Python connectors for llama.cpp, including vision models (Gemma 3, Qwen2-VL, QVQ). Compile llama.cpp and run!

## Overview

This project provides lightweight Python connectors to easily interact with llama.cpp models, supporting both standard text models (via `llama-server`) and multimodal vision models (via their specific CLI tools, e.g., `llama-gemma3-cli`). It creates a simple framework to build applications on top of llama.cpp while handling the complexity of model configuration, server management, and basic inference.

The idea behind it is simple: to offer a minimalistic environment for Python coders to directly interact with llama.cpp without intermediaries like Ollama or LMStudio. Just go to HuggingFace, download your models, use the connectors and have fun!

## Features

- 🚀 **Easy to use**: Just two main Python classes to interact with local LLMs: `LlamaServerConnector` and `LlamaVisionConnector`.
- 🖼️ **Vision model support**: Specifically handles vision models like Gemma 3, Qwen2-VL, and QVQ using their dedicated CLI tools.
- 🔄 **OpenAI-compatible API (Text Models)**: The `LlamaServerConnector` manages a standard `llama-server` process, providing an OpenAI-compatible endpoint (`/v1/...`) for text models.
- 👁️ **OpenAI-compatible API (Vision Models)**: The `LlamaVisionConnector` can *optionally* run an internal FastAPI server (`auto_start=True`) providing an OpenAI-compatible `/v1/chat/completions` endpoint that accepts multimodal requests (text + image URLs). This internally calls the vision CLI tool.
- ⚙️ **Configurable**: Simple JSON-based configuration (`config/models.json`) for all model parameters (paths, CLI command, GPU layers, temperature, etc.).
- 🛠️ **Server Management**: Connectors handle starting, stopping, and finding available ports for the underlying `llama-server` or internal FastAPI server processes.
- 🐳 **Docker Ready**: Build once, prepare your container, run `docker commit` and your LLM-powered app is done.
- 🐛 **Debug Mode**: Optional `debug_server=True` flag provides detailed logging for server process management.
- 🧠 **Great for pros**: A perfect sandbox for those familiar with llama.cpp!

## Components

### 1. LlamaServerConnector (`llama_server_connector.py`)

This component provides an OpenAI-compatible server interface for text-based models.

- Starts and stops the `llama-server` subprocess.
- Configures server parameters (model path, GPU layers, context size, chat template, etc.) based on `config/models.json`.
- Finds an available port and provides the OpenAI-compatible base URL (e.g., `http://127.0.0.1:8080/v1`).
- Offers a simplified `get_response()` method for basic prompts.
- Primarily intended for use with a standard OpenAI client (Python or other) pointed at its managed server URL.

### 2. LlamaVisionConnector (`llama_vision_connector.py`)

Interacts with **multimodal vision models** using their specific CLI tools (e.g., `llama-gemma3-cli`, `llama-qwen2vl-cli`).

- **Direct Mode:** Provides an async `get_response(image_path, prompt)` method that directly runs the configured CLI tool (specified in `config/models.json`) with the image and prompt, parsing the text output.
- **Server Mode (Optional):** If initialized with `auto_start=True`, it runs its *own* internal FastAPI/Uvicorn server process.
    - This internal server exposes an OpenAI-compatible `/v1/chat/completions` endpoint.
    - This endpoint accepts multimodal requests (text and `data:` image URLs).
    - When a request is received, it saves the image temporarily, calls the model's CLI tool via the `get_response` logic, and formats the output as an OpenAI ChatCompletion response.
    - Useful for providing a consistent OpenAI-like interface even for vision models that rely on CLI tools.
- Handles model-specific configuration (CLI command, model paths, mmproj path) from `config/models.json`.
- Parses CLI output to extract the model's response text.

### 3. Configuration (`config/models.json`)

Central JSON file to define model settings.

```json
{
    "MODELS": {
        "MY_TEXT_MODEL": {
            // Parameters for LlamaServerConnector (llama-server)
            "MODEL_PATH": "models/my-text-model.gguf",
            "NUM_LAYERS_TO_GPU": 99,
            "TEMPERATURE": 0.3,
            "NUM_TOKENS_OF_CONTEXT": 8192,
            "CHAT_TEMPLATE": "chatml" // Optional
            // ... other llama-server compatible params ...
        },
        "MY_VISION_MODEL": {
            // Parameters for LlamaVisionConnector (CLI tool)
            "CLI_CMD": "llama-gemma3-cli", // The specific CLI executable
            "MODEL_PATH": "models/my-vision-model.gguf",
            "MMPROJ_PATH": "models/my-mmproj-model.gguf", // Path to multimodal projector
            "TEMPERATURE": 0.3,
            "NUM_LAYERS_TO_GPU": 99
            // ... other params supported by the specific CLI_CMD ...
        }
        // ... add more models ...
    }
}
```

### 4. Examples (`examples/`)

Contains `test_server_connector.py` and `test_vision_connector.py` demonstrating basic usage, including server management, direct calls, and OpenAI client interaction.

### 5. Docker Build System

Simplifies dependency management and deployment (see below).

## How to Use

### Installation

#### 1. Docker Container (Recommended)

The easiest way to get started is with the Docker container. 

A big part of why I built this was to have a _very_ simple llama.cpp/Python sandbox that could follow the new releases of llama.cpp repository faster than bindings such as llama-cpp-python does.

The simple Dockerfile here should do the trick. (Important note: the Dockerfile assumes you are using NVIDIA GPUs.) 

```bash
# Clone the repository
git clone https://github.com/fidecastro/llama-cpp-connector.git
cd llama-cpp-connector

# Run the build script
chmod +x build-docker-container.sh
./build-docker-container.sh

# Run the container with GPU access, mounting your models
docker run --gpus all -v /path/to/your/models:/workspace/models -it llama-cpp-connector:latest
```

The `build-docker-container.sh` script will automatically detect the CUDA compute capability of your GPU and compile llama.cpp specifically for your architecture, so the compilation is as fast as possible and the container is of minimal size.

#### 2. Manual Installation

1.  **Compile llama.cpp:** Clone the [llama.cpp repository](https://github.com/ggerganov/llama.cpp), build it with your desired backend (e.g., CUDA), and ensure the necessary binaries (`llama-server`, `llama-gemma3-cli`, etc.) are accessible in your system's `PATH`.
2.  **Install Python Dependencies:**
    ```bash
    # Core dependencies (needed by connectors)
    pip install requests openai

2. Install Python dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Using the Server Connector (Text Models)

```python
import openai
from llama_server_connector import LlamaServerConnector

# Initialize connector - starts llama-server
connector = LlamaServerConnector(
    config_path="config/models.json",
    model_key="MY_TEXT_MODEL", # Your text model key from config
    initial_port=8080,
    debug_server=False # Set to True for verbose logging
)

# --- Option 1: Simplified get_response (returns only text) ---
print("--- Using connector.get_response ---")
response_text = connector.get_response("Explain quantum computing in simple terms")
print(f"Simple Response: {response_text}")

# --- Option 2: Standard OpenAI Client (Recommended) ---
print("\n--- Using OpenAI client ---")
client = openai.OpenAI(
    base_url=connector.get_server_url(), # Use URL provided by connector
    api_key="not-needed"
)

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    model=connector.model_key, # Can use key from connector
    max_tokens=150
)
print(f"OpenAI Response: {chat_completion.choices[0].message.content}")

# Server is killed automatically on exit, or manually:
# connector.kill_server()
```

### Using the Vision Connector (Multimodal Models)

```python
import asyncio
import base64
import requests
from llama_vision_connector import LlamaVisionConnector

async def process_image():
    # --- Option 1: Direct CLI Interaction --- 
    print("--- Using Direct CLI Interaction ---")
    connector_direct = LlamaVisionConnector(
        config_path="config/models.json",
        model_key="MY_VISION_MODEL", # Your vision model key
        auto_start=False, # DO NOT start internal server
        debug_server=False
    )
    
    try:
        description = await connector_direct.get_response(
            "examples/test_images/sample.jpg", 
            prompt="Describe this image in detail"
        )
        print(f"Direct Response: {description}")
    except Exception as e:
        print(f"Direct call failed: {e}")
    # No server to kill for connector_direct

    # --- Option 2: Internal Server Mode (OpenAI-like endpoint) ---
    print("\n--- Using Internal Server Mode ---")
    # Reset singleton before creating server instance
    LlamaVisionConnector._instance = None 
    LlamaVisionConnector._server_process = None

    connector_server = LlamaVisionConnector(
        config_path="config/models.json",
        model_key="MY_VISION_MODEL",
        auto_start=True, # START internal server
        server_port=8001,
        debug_server=False
    )

    try:
        print("Waiting for server...")
        await asyncio.sleep(5) # Give server time to start

        if not connector_server.is_server_running():
            print("Server failed to start. Exiting.")
            return

        server_url = f"http://{connector_server.server_host}:{connector_server.server_port}/v1/chat/completions"
        print(f"Sending request to internal server: {server_url}")

        # Prepare image data
        with open("examples/test_images/sample.jpg", "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            image_url_data = f"data:image/jpeg;base64,{image_data}"

        request_data = {
            "model": connector_server.model_key,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image?"},
                        {"type": "image_url", "image_url": {"url": image_url_data}}
                    ]
                }
            ]
        }

        # Use requests (or an async HTTP client) to send the request
        response = await asyncio.to_thread(requests.post, server_url, json=request_data, timeout=60)

        if response.status_code == 200:
            print(f"Server Response: {response.json()['choices'][0]['message']['content']}")
        else:
            print(f"Server Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Server mode failed: {e}")

    # Server is killed automatically on exit
    # connector_server.kill_server()

# Run the async function
asyncio.run(process_image())
```
### Using with [Open WebUI](https://github.com/open-webui/open-webui/)

It's super simple to connect either connector (server or vision) to Open WebUI:

1. Compile the container for llama-cpp-connector as usual.
2. Run `pip install open-webui` inside the container.
3. Run either `python llama_server_connector.py` (for text models), or `python llama_vision_connector.py server` (for vision models). 
4. Run `open-webui serve --port 3000` in a terminal inside the container.
5. Log in to the Open WebUI HTTP page (in 127.0.0.1:3000) > Settings > Admin Settings > disable all connections but OpenAI API, and set it to point to `http://localhost:8080/v1` (for llama_server_connector) or `http://localhost:8001/v1` (for llama_vision_connector); also add a dummy API key ("not-needed")

This provides a complete UI for interacting with your local models through a familiar chat interface.

## Configuration Details (`config/models.json`)

*   **`MODELS`**: Dictionary containing configurations for different models, keyed by a unique name.
*   **Model Entry**: Each model has its own dictionary.
    *   **`MODEL_PATH`**: (Required) Path to the main GGUF model file.
    *   **`CLI_CMD`**: (Required for Vision Models) The specific llama.cpp CLI executable (e.g., `llama-gemma3-cli`, `llama-qwen2vl-cli`). Used by `LlamaVisionConnector`.
    *   **`MMPROJ_PATH`**: (Required for Vision Models) Path to the multimodal projector file. Used by `LlamaVisionConnector`.
    *   Other parameters are passed either to `llama-server` (by `LlamaServerConnector`) or the specific CLI tool (by `LlamaVisionConnector`). Common examples:
        *   `NUM_LAYERS_TO_GPU` (`-ngl`)
        *   `TEMPERATURE` (`--temp`)
        *   `NUM_TOKENS_OF_CONTEXT` (`-c`, for `llama-server`)
        *   `CHAT_TEMPLATE` (`--chat-template`, for `llama-server`)
        *   Refer to llama.cpp documentation for parameters supported by `llama-server` or the specific vision CLI tool.

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

Mount your models directory:
```bash
docker run --gpus all -v /path/to/your/models:/workspace/models -it llama-cpp-connector:latest
```

## Why llama-cpp-connector?

- **Simplified integration**: No need to directly interface with llama.cpp's C++ code
- **Simplified configuration**: Change model behavior just by editing a JSON script
- **Docker-first approach**: Consistent environment across development and deployment
- **Minimal dependencies**: Just have llama.cpp binaries and you're done
- **OpenAI-compatible**: Use familiar APIs with local models

## License

[MIT License](LICENSE)
