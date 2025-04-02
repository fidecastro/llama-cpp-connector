# Example Scripts for Llama Connectors

This directory contains example scripts demonstrating the usage of both the `LlamaServerConnector` (for standard text models via `llama-server`) and `LlamaVisionConnector` (for multimodal models via their specific CLIs).

## Prerequisites

1.  Ensure you have a valid `config/models.json` file with your model configurations (see main project README).
2.  For `test_server_connector.py`, ensure the `openai` Python library is installed:
    ```bash
    pip install openai
    ```
3.  For `test_vision_connector.py`, ensure `requests` is installed. If you want to test the server mode (which is the default), you also need `fastapi` and `uvicorn`:
    ```bash
    pip install requests fastapi uvicorn
    ```
    *Note: `LlamaVisionConnector` itself handles the case where FastAPI/Uvicorn are not installed if you only use direct CLI calls (`auto_start=False`).*
4.  For vision testing (`test_vision_connector.py`), create a `test_images` directory within `examples/` and add a sample image (e.g., `sample.jpg`).
5.  Ensure you have a `vision-prompt.txt` file in the *root* of the project directory if you want to test the file-based prompt functionality in `test_vision_connector.py`.

## Test Scripts

### 1. `test_server_connector.py`

This script demonstrates using `LlamaServerConnector` to interact with a text generation model running via the standard `llama-server` process managed by the connector.

It shows:
*   Initializing the connector (which starts `llama-server`).
*   Getting responses using the simplified `connector.get_response()` method.
*   Getting responses using a standard `openai` client pointed at the connector's managed server URL.
*   Testing chat completions and streaming.

To run:
```bash
python examples/test_server_connector.py
```

### 2. `test_vision_connector.py`

This script demonstrates using `LlamaVisionConnector` to interact with multimodal vision models.

It tests two main modes:
1.  **Direct Interaction:** Calls the specific vision model's CLI tool directly using `connector.get_response()`.
    *   Tests with a prompt string.
    *   Tests using the `prompt_file` argument, pointing to the `vision-prompt.txt` located in the project root.
2.  **Server Mode:** Initializes the connector with `auto_start=True`, which runs the connector's internal FastAPI/Uvicorn server in a background process.
    *   Sends an OpenAI-compatible request (including a base64 encoded image) to the connector's `/v1/chat/completions` endpoint.

To run:
```bash
python examples/test_vision_connector.py
```

## Directory Structure Example

```
.
├── config/
│   └── models.json
├── examples/
│   ├── README.md
│   ├── test_server_connector.py
│   ├── test_vision_connector.py
│   └── test_images/
│       └── sample.jpg      # Add your test image here
├── llama_server_connector.py
├── llama_vision_connector.py
├── vision-prompt.txt       # Example prompt file for vision models
└── ... (other project files)
```

## Notes

*   Both connectors use a Singleton pattern, meaning only one instance (and potentially one managed server process per class) exists at a time.
The test scripts demonstrate how to reset this for sequential tests if needed.
*   The `debug_server=True` flag is enabled by default in both test scripts. This provides verbose logging for the server process startup and shutdown, which is helpful for debugging.
*   The scripts automatically find available ports starting from the `initial_port` specified (defaults are 8080 for server, 8001 for vision).
*   Server processes are managed by the connector instances and are automatically terminated when the main script exits (using `atexit`).

## Customization

You can modify the following parameters when initializing the connectors in the scripts:
*   `config_path`: Path to your models configuration file.
*   `model_key`: Specific model to use (if `None`, uses the first suitable model found).
*   `initial_port` / `server_port`: Starting port number for the server.
*   `host` / `server_host`: Host address for the server.
*   `debug_server`: Set to `False` to disable verbose server logging.
*   Image paths and test prompts within the scripts. 