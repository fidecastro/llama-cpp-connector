import subprocess
import asyncio
import os
import json
from typing import Dict, Any, Optional, List, Union
import base64
import tempfile
import re
import uuid
import time
import shutil
import sys
import requests
import argparse
from pydantic import BaseModel, Field, ValidationError
import select # For non-blocking reads
import fcntl # For non-blocking reads
import threading # <-- Import threading

# --- FastAPI / Server Imports ---
# Wrap in try-except to allow running without FastAPI installed if server isn't started
try:
    from fastapi import FastAPI, HTTPException, Body, Request
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Define dummy classes if FastAPI is not installed, so type hints don't break
    class BaseModel: pass
    Body = lambda **kwargs: None # Define Body as a dummy lambda
    HTTPException = Exception # Map to base Exception
    print("Warning: FastAPI or Uvicorn not installed. Server functionality disabled.")
# --- End FastAPI Imports ---


# --- Pydantic Models for FastAPI ---
# Only define if FastAPI is available
if FASTAPI_AVAILABLE:
    class ImageUrl(BaseModel):
        url: str

    class ContentItem(BaseModel):
        type: str
        text: Optional[str] = None
        image_url: Optional[ImageUrl] = None

    class Message(BaseModel):
        role: str
        content: Union[str, List[ContentItem]]

    class ChatCompletionRequest(BaseModel):
        model: Optional[str] = None # Can be used to verify against connector's model
        messages: List[Message]
        # Add other OpenAI parameters if needed (temperature, max_tokens, etc.)
        # These would need to be passed down or handled if overriding CLI args

    # We can reuse the dict structure for response, or define a Pydantic model too
    class ChatCompletionChoice(BaseModel):
        index: int
        message: Message
        finish_reason: str

    class Usage(BaseModel):
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        total_tokens: Optional[int] = None

    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[ChatCompletionChoice]
        usage: Usage
        system_fingerprint: Optional[str] = None

    class ModelsList(BaseModel):
        object: str
        data: List[Dict[str, Any]]
# --- End Pydantic Models ---


class LlamaVisionConnector:
    """
    Connector for interacting with llama.cpp vision models via their CLI tools.
    Provides methods for direct interaction (`get_response`, `get_openai_response`)
    and can optionally run as a FastAPI server endpoint (`/v1/chat/completions`)
    for OpenAI-compatible HTTP interactions.
    """
    _instance = None
    _server_process = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LlamaVisionConnector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self,
                 config_path: str = "config/models.json",
                 model_key: str = None,
                 auto_start: bool = True,
                 server_host: str = "0.0.0.0",
                 server_port: int = 8001,
                 debug_server: bool = False):
        """
        Initialize the connector, load config, and optionally start the server subprocess.
        
        Args:
            config_path (str): Path to the JSON configuration file.
            model_key (str, optional): Key of the model in config. Uses first if None.
            auto_start (bool): If True, automatically starts the server as a separate
                               process during initialization. Defaults to True.
            server_host (str): Host address for the FastAPI server process.
            server_port (int): Port for the FastAPI server process.
            debug_server (bool): If True, enable detailed debug printing for the server subprocess management.
        """
        if not hasattr(self, 'initialized'):
            # --- Core Initialization ---
            print(f"Initializing LlamaVisionConnector...")
            self.debug_server = debug_server # STORE the flag
            self.config_path = os.path.abspath(config_path) # Store absolute path
            self._load_config(self.config_path)
            self._initialize_model(model_key)

            # --- Server Process Attributes ---
            self.server_host = server_host
            self.initial_port = server_port  # Store initial port
            self.server_port = self.find_available_port(self.initial_port, self.server_host)  # Find available port
            self._server_process: Optional[subprocess.Popen] = None # Holds the server subprocess handle
            self.app: Optional[FastAPI] = None # App instance created *by the subprocess*
                                                # or if running the server blocking method directly.

            # --- Auto-Start Server Process ---
            if auto_start:
                print("auto_start is True. Attempting to start server process...")
                self.start_server_process()
            else:
                print("auto_start is False. Server process not started automatically.")

            # --- Register Cleanup --- Ensure server process is killed on exit ---
            import atexit
            atexit.register(self.kill_server) # Register kill_server to be called on script exit
            
            self.initialized = True

    def find_available_port(self, initial_port: int, host: str) -> int:
        """
        Find an available port starting from initial_port.
        
        Args:
            initial_port (int): The initial port to try
            host (str): The host to check ports on
            
        Returns:
            int: An available port number
        """
        port = initial_port
        while True:
            try:
                response = requests.get(f'http://{host}:{port}/health')
                if response.status_code != 200:
                    return port
                else:
                    port += 1
            except requests.ConnectionError:
                return port
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load or reload configuration from the JSON file.
        
        Args:
            config_path (str): Path to the configuration JSON file
            
        Returns:
            dict: Configuration as a dictionary
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file: {config_path}")
            
        # Basic config validation after loading
        self.processing_folder = self.config.get("OUTPUT_FOLDER", "output")
        self.vision_prompt_filename = self.config.get("VISION_PROMPT_FILENAME", "vision-prompt.txt")
        self.valid_image_extensions = self.config.get("VALID_IMAGE_EXTENSIONS", ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'])
        
        return self.config

    def _initialize_model(self, model_key: Optional[str]):
        """
        Sets the initial model configuration based on the provided key or defaults.
        
        Args:
            model_key (str, optional): The key of the model to use from the config.
                                       If None, uses the first model found.
        """
        models = self.config.get("MODELS", {})
        if not models:
            raise ValueError("No models defined in the configuration file.")

        if model_key:
            if model_key in models:
                print(f"Using specified model key: '{model_key}'")
                self.set_model(model_key)
            else:
                raise ValueError(f"Specified model key '{model_key}' not found in configuration.")
        else:
            # Default to the first model in the dictionary
            first_model_key = next(iter(models))
            print(f"No model key specified. Defaulting to first model: '{first_model_key}'")
            self.set_model(first_model_key)

    def set_model(self, model_key: str) -> None:
        """
        Change the model configuration.
        
        Args:
            model_key (str): Key of the model in the configuration
        """
        if model_key not in self.config.get("MODELS", {}):
            raise ValueError(f"Model '{model_key}' not found in configuration")
            
        self.model_config = self.config["MODELS"][model_key]
        self.model_key = model_key
        
    def _parse_cli_output(self, cli_output: str, prompt: str) -> str:
        """
        Parses the raw stdout from the llama.cpp vision CLI to extract the model's response.
        Uses logic refined from testing and incorporates several strategies.
        May require adjustments for different CLI tools or versions.
        
        Args:
            cli_output (str): The raw standard output from the CLI process.
            prompt (str): The prompt sent to the CLI (for context if needed).
        
        Returns:
            str: The cleaned model response text.
        """
        print("--- Attempting to parse CLI Output ---")
        # Uncomment to see the full raw output during debugging:
        # print("--- Raw CLI Output Start ---")
        # print(cli_output)
        # print("--- Raw CLI Output End ---")

        if not cli_output:
             print("Warning: CLI output was empty.")
             return ""

        output_lines = cli_output.strip().splitlines()
        # Initialize response_start_line_index to -1 (meaning no marker found yet)
        response_start_line_index = -1
        last_marker_found_at = -1 # Track the line index where the last marker was found
        extracted_response = cli_output # Default to full output

        # --- Strategy 1: Look for known markers indicating end of setup/start of response ---
        # Iterate through *all* lines to find the *last* occurrence of any known marker.
        print("Executing Strategy 1: Searching for known markers...")
        known_markers = [
            "Image decoded in ",                         # Gemma3 marker
            "encode_image_with_clip: image encoded in ", # Qwen2-VL marker
            # Add other potential markers here
        ]

        for i, line in enumerate(output_lines):
            line_strip = line.strip()
            for marker in known_markers:
                if line_strip.startswith(marker):
                    # Found a marker. Record the line index *after* this one as the potential start.
                    # Update this *every time* a marker is found to ensure we use the last one.
                    response_start_line_index = i + 1
                    last_marker_found_at = i # Keep track of the line the marker was on
                    print(f"  Found marker '{marker}...' at line {i}. Potential response start updated to line {response_start_line_index}.")
                    # Don't break inner loop yet, check other markers on the same line?
                    # (Current logic assumes one marker per line is sufficient)
                    break # Break inner loop (markers) and continue to next line

        # After checking all lines, if a marker was found (response_start_line_index is not -1)
        if response_start_line_index != -1:
            print(f"Strategy 1 finished. Last known marker was found at line {last_marker_found_at}. Extracting response from line {response_start_line_index}.")
            if response_start_line_index < len(output_lines):
                # Extract from the line *after* the last marker found
                extracted_response = '\n'.join(output_lines[response_start_line_index:]).strip()
            else:
                # Last marker was on the very last line
                print("Warning: Last known marker was on the final line. Assuming no response text followed.")
                extracted_response = ""
            # Proceed directly to cleanup for the response extracted by Strategy 1

        # --- Strategy 2: Look for chat markers (ONLY if Strategy 1 failed) ---
        else: # No known markers were found by Strategy 1
            print("Strategy 1 found no known markers. Executing Strategy 2: Searching for 'ASSISTANT:' marker...")
            assistant_marker = "ASSISTANT:"
            found_assistant = False
            # Search backwards for the last occurrence
            for i in range(len(output_lines) - 1, -1, -1):
                 line_strip = output_lines[i].strip()
                 if line_strip.startswith(assistant_marker):
                      marker_pos = output_lines[i].find(assistant_marker)
                      partial_line = output_lines[i][marker_pos + len(assistant_marker):].strip()
                      remaining_lines = output_lines[i+1:]
                      extracted_response = "\n".join([partial_line] + remaining_lines).strip()
                      print(f"  Strategy 2 found '{assistant_marker}' marker at line {i}. Extracted response.")
                      found_assistant = True
                      break # Found the last relevant marker

            # --- Strategy 3: Fallback - Try removing echoed prompt (if Strategies 1 & 2 failed) ---
            if not found_assistant:
                 print("Strategy 2 found no 'ASSISTANT:' marker. Executing Strategy 3: Attempting prompt removal fallback...")
                 # Use rfind to find the last occurrence of the prompt
                 prompt_marker_pos = cli_output.rfind(prompt.strip())
                 if prompt_marker_pos != -1:
                      # Find the start of the text *after* the prompt
                      potential_start = prompt_marker_pos + len(prompt.strip())
                      # Skip potential newlines or whitespace right after the prompt
                      while potential_start < len(cli_output) and cli_output[potential_start].isspace():
                           potential_start += 1

                      if potential_start < len(cli_output):
                          extracted_response = cli_output[potential_start:].strip()
                          print("  Strategy 3: Removed echoed prompt to extract response.")
                      else:
                          # Prompt found, but nothing follows? Use full output.
                          print("  Strategy 3 Warning: Prompt found, but no text followed. Using full CLI output.")
                          extracted_response = cli_output # Keep default
                 else:
                      # No markers, no prompt found -> Use full output.
                      print("  Strategy 3 Warning: No known markers or prompt found. Using full CLI output.")
                      extracted_response = cli_output # Keep default

        # --- Final Cleanup Steps (Applied to the extracted_response from the successful strategy) ---
        print("Applying final cleanup to extracted response...")
        final_response = extracted_response

        # Remove potential trailing timing info
        timing_marker = "llama_print_timings:"
        timing_pos = final_response.rfind(timing_marker)
        if timing_pos != -1:
            # Heuristic check if timing info is likely the last block
            lines_after_timing = final_response[timing_pos:].count('\n')
            chars_after_timing = len(final_response[timing_pos:])
            if chars_after_timing < 300 and lines_after_timing < 6:
                print("Detected potential timing info at the end, truncating.")
                final_response = final_response[:timing_pos].strip()

        # Add any other general cleanup rules here if needed

        print(f"Final parsed response length: {len(final_response)}")
        return final_response

    async def _run_cli(self, image_path: str, prompt: str) -> str:
        """
        Internal helper to build/run the CLI command and parse the output.

        Args:
            image_path (str): Path to the image file.
            prompt (str): The prompt to send to the CLI.

        Returns:
            str: The cleaned model response text.
        """
        cmd = [self.model_config["CLI_CMD"]]
        cmd.extend(["-m", self.model_config["MODEL_PATH"]])
        cmd.extend(["-fa", "-sm", "row"])
        if "MMPROJ_PATH" in self.model_config:
            cmd.extend(["--mmproj", self.model_config["MMPROJ_PATH"]])
        # Add other relevant parameters from config
        if "TEMPERATURE" in self.model_config:
            cmd.extend(["--temp", str(self.model_config["TEMPERATURE"])])
        if "NUM_LAYERS_TO_GPU" in self.model_config:
            cmd.extend(["-ngl", str(self.model_config["NUM_LAYERS_TO_GPU"])])
        # Add more parameters as needed from your config structure

        cmd.extend(["--image", image_path])
        cmd.extend(["-p", prompt])

        print(f"Running command: {' '.join(cmd)}") # For debugging

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        # Decode using utf-8, ignoring errors which might occur with unexpected output
        stdout_decoded = stdout.decode('utf-8', errors='ignore').strip()
        stderr_decoded = stderr.decode('utf-8', errors='ignore').strip()

        # --- Error Handling ---
        if process.returncode != 0:
            # CLI process failed
            error_message = f"Vision CLI ('{cmd[0]}') failed with return code {process.returncode}."
            if stderr_decoded:
                error_message += f"\nStderr:\n{stderr_decoded}"
            # Sometimes errors are printed to stdout instead of stderr
            elif stdout_decoded:
                error_message += f"\nStdout:\n{stdout_decoded}"
            print(f"ERROR: {error_message}")
            raise RuntimeError(error_message)

        # Print stderr as a warning even if the process succeeded (return code 0)
        # It might contain useful info or non-fatal warnings from llama.cpp
        if stderr_decoded:
            print(f"--- CLI Stderr (Warning, Return Code 0) ---")
            print(stderr_decoded)
            print(f"--- End CLI Stderr ---")

        # --- Parse Output --- Call the dedicated method ---
        # The parsing logic is now fully contained within _parse_cli_output
        parsed_output = self._parse_cli_output(stdout_decoded, prompt)
        # --- End Parse Output ---

        return parsed_output


    async def get_response(self, image_path: str, prompt: str = None, prompt_file: str = None) -> str:
        """
        Process an image with a prompt using the configured llama.cpp vision CLI.
        This is the method for direct interaction, returning the parsed text output.
        It does NOT require the FastAPI server to be running.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: The parsed text output from the CLI.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if prompt is None and prompt_file is None:
            # Use default prompt if available in config, otherwise raise error
            if "DEFAULT_PROMPT" in self.model_config:
                final_prompt = self.model_config["DEFAULT_PROMPT"]
            elif os.path.exists("vision-prompt.txt"):
                 print("Using default prompt from vision-prompt.txt")
                 with open("vision-prompt.txt", "r") as f:
                      final_prompt = f.read().strip()
            else:
                raise ValueError("No prompt provided and no default prompt configured or found.")
        elif prompt:
            final_prompt = prompt
        else: # prompt_file provided
            if not os.path.exists(prompt_file):
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
            with open(prompt_file, "r") as f:
                final_prompt = f.read().strip()

        return await self._run_cli(image_path, final_prompt)


    async def get_openai_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes OpenAI-formatted multimodal input and returns a response dictionary
        in the OpenAI ChatCompletion format.
        Raises HTTPException(400) if no image_url is found in the user message.

        Args:
            messages: A list of message dictionaries (OpenAI format).

        Returns:
            dict: The response dictionary in the OpenAI ChatCompletion format.
        """
        text_prompt = ""
        image_data_url = None

        # 1. Parse input messages to find text prompt and image url
        print("--> [Server Process] Parsing messages in get_openai_response...")
        user_message_content = None
        for message in messages:
            if message.get("role") == "user":
                user_message_content = message.get("content")
                break # Assuming only one user message block contains the multimodal input

        if isinstance(user_message_content, list):
            print("--> [Server Process] User message content is a list. Processing items...")
            for item in user_message_content:
                if item.get("type") == "text":
                    text_prompt += item.get("text", "") + "\n"
                elif item.get("type") == "image_url":
                    image_data_url = item.get("image_url", {}).get("url")
        elif isinstance(user_message_content, str):
             # Handle case where content might just be a string (Open-WebUI post-processing?)
             print("--> [Server Process] User message content is a string. Assuming text-only.")
             text_prompt = user_message_content # Use the string as prompt
             # image_data_url remains None

        text_prompt = text_prompt.strip()
        print(f"--> [Server Process] Parsed text_prompt (len: {len(text_prompt)}): '{text_prompt[:100]}...'")
        print(f"--> [Server Process] Parsed image_data_url is {'set' if image_data_url else 'None'}")

        # --- Check if image_data_url was actually found --- #
        if not image_data_url:
            print("--> [Server Process] ERROR: No image_data_url found. Raising 400.")
            raise HTTPException(
                 status_code=400,
                 detail="Request failed: No image_url found in user message content. This endpoint requires multimodal input."
            )
        # --- End Check --- #

        if not text_prompt:
            # Use default if no specific prompt given
             if "DEFAULT_PROMPT" in self.model_config:
                 text_prompt = self.model_config["DEFAULT_PROMPT"]
             elif os.path.exists("vision-prompt.txt"):
                 print("Using default prompt from vision-prompt.txt for OpenAI request")
                 with open("vision-prompt.txt", "r") as f:
                      text_prompt = f.read().strip()
             else:
                 text_prompt = "Describe the image." # Default fallback

        if not image_data_url:
            raise ValueError("No image_url found in user message content.")

        # 2. Handle Image Data
        # Expected format: "data:image/{format};base64,{data}"
        match = re.match(r"data:image/(\w+);base64,(.*)", image_data_url)
        if not match:
            raise ValueError("Invalid image data URL format. Expected 'data:image/{format};base64,{data}'.")

        image_format = match.group(1)
        base64_data = match.group(2)
        image_bytes = base64.b64decode(base64_data)

        temp_image_file = None
        temp_image_path = ""
        try:
            # 3. Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_format}") as temp_file:
                temp_file.write(image_bytes)
                temp_image_path = temp_file.name
            print(f"--> [Server Process] Temporary image saved to: {temp_image_path}")

            # 4. Execute CLI using the helper method
            print("--> [Server Process] Calling _run_cli...")
            cli_output_text = await self._run_cli(temp_image_path, text_prompt)
            print(f"--> [Server Process] _run_cli finished. Output length: {len(cli_output_text)}")

            # 5. Format Output
            response_id = f"chatcmpl-vision-{uuid.uuid4()}"
            created_timestamp = int(time.time())
            model_name = self.model_key # Or self.model_config.get("MODEL_NAME", self.model_key)

            openai_response = {
                "id": response_id,
                "object": "chat.completion",
                "created": created_timestamp,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": cli_output_text, # Already stripped in _run_cli
                        },
                        "finish_reason": "stop", # CLI runs to completion
                    }
                ],
                "usage": { # Token counts are not available from CLI
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None
                }
            }
            print("--> [Server Process] Formatting OpenAI response...")
            return openai_response

        except HTTPException: # Re-raise HTTPExceptions directly
             raise
        except Exception as e:
             print(f"--> [Server Process] Error during OpenAI-compatible vision processing: {e}")
             # Raise as 500 for unexpected errors during processing
             raise HTTPException(status_code=500, detail=f"Internal server error during vision processing: {e}")
        finally:
            # 6. Cleanup Temporary File
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                    print(f"--> [Server Process] Cleaned up temp image: {temp_image_path}")
                except OSError as e:
                    print(f"Error deleting temporary file {temp_image_path}: {e}")

    def is_valid_image_file(self, filename: str) -> bool:
        """
        Check if a filename has a valid image extension.
        
        Args:
            filename (str): The filename to check
            
        Returns:
            bool: True if the filename has a valid image extension, False otherwise
        """
        # Check if the filename ends with any of the valid extensions
        return any(filename.lower().endswith(ext) for ext in self.valid_image_extensions)


    # --- Server Methods (Optional Functionality) ---

    def _create_fastapi_app(self):
        """
        Creates the FastAPI application instance and defines endpoints.
        This is only needed if you intend to run the connector as an HTTP server.
        Requires FastAPI and Uvicorn to be installed.
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is not installed. Cannot create server app.")

        app = FastAPI(
            title="Llama Vision Connector",
            description="OpenAI-compatible endpoint for llama.cpp vision models via CLI",
            version="0.1.0"
        )

        # --- Define Endpoint ---
        # Note: Using POST for chat completions endpoint
        @app.post("/v1/chat/completions",
                  # response_model=ChatCompletionResponse, # Enable if using Pydantic response model
                  summary="Handle Multimodal Chat Completion Request",
                  tags=["Chat"])
        async def handle_chat_completion(
            body: dict = Body(...), # Accept raw dict first
            raw_request: Request = None
        ):
            """
            Accepts OpenAI-compatible chat completion requests with multimodal content.
            Processes the request using the configured llama.cpp vision model CLI.
            """
            print("--> [Server Process] Entered handle_chat_completion")
            # --- DEBUG: Print the raw body received by FastAPI ---
            print("--> [Server Process] Raw Dict Body Received by FastAPI:")
            import json
            print(json.dumps(body, indent=2))
            # print("--- End Raw Dict Body ---") # REMOVED for brevity
            # --- END DEBUG ---

            # --- Manually Validate with Pydantic ---
            request_payload = None # Initialize
            try:
                print("--> [Server Process] Attempting Pydantic validation...")
                request_payload = ChatCompletionRequest(**body)
                print("--> [Server Process] Pydantic validation successful.")
            except ValidationError as e:
                print("!!! [Server Process] Pydantic Validation Error !!!")
                print(e.json(indent=2))
                raise HTTPException(status_code=422, detail=e.errors())
            except Exception as e:
                 print(f"--> [Server Process] Error during manual Pydantic validation: {e}")
                 raise HTTPException(status_code=400, detail=f"Error processing request body: {e}")
            # --- END Manual Validation ---

            print(f"--> [Server Process] Received request for model: {request_payload.model or 'default'}")

            response_dict = None # Initialize
            try:
                # Convert Pydantic messages back to dicts for get_openai_response
                messages_dict_list = [msg.dict() for msg in request_payload.messages]

                print("--> [Server Process] Calling get_openai_response...")
                response_dict = await self.get_openai_response(messages=messages_dict_list)
                print("--> [Server Process] get_openai_response finished.")

            except (ValueError, FileNotFoundError) as e:
                print(f"--> [Server Process] Input Error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except RuntimeError as e:
                print(f"--> [Server Process] Runtime Error: {e}")
                raise HTTPException(status_code=500, detail=f"Backend CLI execution failed: {e}")
            except Exception as e:
                print(f"--> [Server Process] Unexpected Server Error: {e}")
                raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

            print("--> [Server Process] Returning response.")
            return response_dict

        # --- Add endpoint for listing models ---
        @app.get("/v1/models",
                 # response_model=ModelsList, # Optional: Define Pydantic model for response
                 summary="List Available Models",
                 tags=["Management"])
        async def list_models():
            """
            Provides information about the currently loaded model, mirroring
            the format used by llama-server's /v1/models endpoint (without metadata).
            """
            if not self.model_config or "MODEL_PATH" not in self.model_config:
                raise HTTPException(status_code=500, detail="Model configuration not loaded properly.")

            # Use the model path as the ID, consistent with the user's example output
            model_id_to_use = self.model_config.get("MODEL_PATH", "unknown")
            current_time = int(time.time())

            # Construct the response data
            # We cannot easily get the 'meta' field without parsing the GGUF file
            model_data = {
                "id": model_id_to_use,
                "object": "model",
                "created": current_time,
                "owned_by": "LlamaVisionConnector" # Indicate the owner
                # "meta": {} # Omitting meta field
            }

            response = {
                "object": "list",
                "data": [model_data]
            }
            return response

        # --- Add other endpoints if needed (e.g., health check) ---
        @app.get("/health", summary="Health Check", tags=["Management"])
        async def health_check():
            return {"status": "ok", "model_key": self.model_key}

        self.app = app
        return app

    def run_server(self, host: str = None, port: int = None):
        """
        Starts the Uvicorn server to serve the FastAPI application.
        This method provides the optional HTTP endpoint for OpenAI-compatible interaction.
        It requires FastAPI and Uvicorn to be installed.

        NOTE: This method is BLOCKING. It will run the server indefinitely
              until interrupted (e.g., Ctrl+C).

        Args:
            host (str, optional): Host to bind the server to. Defaults to `self.server_host`.
            port (int, optional): Port to bind the server to. Defaults to `self.server_port`.
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI/Uvicorn not installed, cannot start server.")

        _host = host if host is not None else self.server_host
        _port = port if port is not None else self.server_port

        if not self.app:
             self._create_fastapi_app() # Create app instance if not already done

        print(f"Starting Uvicorn server on http://{_host}:{_port}")
        # Consider adding reload=True for development, but disable for production
        uvicorn.run(self.app, host=_host, port=_port, log_level="info")
        # This line will block until the server is stopped (e.g., Ctrl+C)
        print("Uvicorn server stopped.")

    # Optional: Add an async start method if needed for non-blocking startup
    # async def start_server_async(self, host: str = None, port: int = None):
    #     ... run uvicorn in a separate thread or use asyncio features ...

    def is_server_running(self, timeout: float = 1.0) -> bool:
        """
        Checks if the managed server *process* is running and if the server
        responds to the /health endpoint.

        Args:
            timeout (float): HTTP request timeout for the health check.

        Returns:
            bool: True if the process exists and the server responds, False otherwise.
        """
        # 1. Check if the process handle exists and the process hasn't terminated
        if self._server_process is None or self._server_process.poll() is not None:
            if self._server_process and self._server_process.poll() is not None:
                 print(f"Server process (PID {self._server_process.pid}) has terminated with code {self._server_process.poll()}. Clear the handle.")
                 self._server_process = None # Clear the handle if terminated
            else:
                 print("Server process is not running (no active process handle).")
            return False

        # 2. If process seems alive, perform the HTTP health check
        if not FASTAPI_AVAILABLE:
            print("FastAPI not available, cannot perform HTTP health check.")
            # Process is running, but can't confirm health via HTTP
            # Consider returning True here with a warning, or False?
            # Returning False as we can't confirm the API is responsive.
            return False

        health_url = f"http://{self.server_host}:{self.server_port}/health"
        # print(f"Checking server health at: {health_url}") # Less verbose check
        try:
            response = requests.get(health_url, timeout=timeout)
            if 200 <= response.status_code < 300:
                 # print(f"Server responded OK ({response.status_code}).") # Less verbose
                 return True
            else:
                 print(f"Server process (PID {self._server_process.pid}) responded to health check with status {response.status_code}.")
                 return False
        except requests.exceptions.RequestException as e:
            # Don't print full error every time, just indicate failure
            # print(f"Health check failed: {e}")
            print(f"Server process (PID {self._server_process.pid}) did not respond to health check ({e.__class__.__name__}).")
            return False

    def kill_server(self):
        """
        Terminates the background server process if it is running.
        """
        if self._server_process and self._server_process.poll() is None:
            pid = self._server_process.pid
            print(f"Attempting to terminate server process (PID {pid})...")
            try:
                self._server_process.terminate() # Send SIGTERM
                try:
                    # Wait for a short period for the process to terminate
                    self._server_process.wait(timeout=5.0)
                    print(f"Server process (PID {pid}) terminated successfully.")
                except subprocess.TimeoutExpired:
                    print(f"Server process (PID {pid}) did not terminate gracefully after 5s. Sending SIGKILL...")
                    self._server_process.kill() # Send SIGKILL if terminate failed
                    self._server_process.wait(timeout=1.0) # Wait briefly after kill
                    print(f"Server process (PID {pid}) killed.")
            except Exception as e:
                print(f"Error during server process termination: {e}")
            finally:
                self._server_process = None # Clear the handle regardless
                LlamaVisionConnector._instance = None  # Clear the singleton instance
        elif self._server_process:
             print(f"Server process (PID {self._server_process.pid}) already terminated.")
             self._server_process = None # Clear handle if already dead
             LlamaVisionConnector._instance = None  # Clear the singleton instance
        else:
            print("No server process to kill (process handle is None).")

        # Also ensure the atexit registration doesn't try to kill again
        # This is tricky, maybe atexit should check self._server_process itself
        # Or we try to unregister? (Requires storing the registration function)
        # Safest is just letting atexit call kill_server, which checks if process is None

    def restart_server(self):
        """
        Stops the current server process (if running) and starts a new one.
        """
        print("Attempting to restart server process...")
        try:
            self.kill_server() # Attempt to stop the existing process first
        except Exception as e:
             print(f"Ignoring error during kill phase of restart: {e}")

        # Add a small delay before starting again
        time.sleep(1)

        try:
            self.start_server_process() # Start a new process
            print("Server restart initiated successfully.")
        except Exception as e:
             print(f"ERROR: Failed to start new server process during restart: {e}")

    def start_server_process(self, wait_time: float = 5.0):
        """
        Starts the FastAPI/Uvicorn server as a separate background process.
        Uses subprocess.Popen to launch the current script with the
        '--internal-run-server' argument.

        Args:
            wait_time (float): How long to wait (in seconds) after launching
                               for the server to become responsive.
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI/Uvicorn not available, cannot start server process.")

        if self.is_server_running(timeout=0.5): # Quick check first
            print("Server process appears to be running already.")
            return

        print("Starting FastAPI server process in the background...")

        # Command to re-run this script with the internal server flag
        # sys.executable ensures we use the same python interpreter
        # -u makes output unbuffered
        command = [
            sys.executable,
            "-u",
            os.path.abspath(__file__), # Use absolute path to this script
            "--internal-run-server",
            "--host", self.server_host,
            "--port", str(self.server_port),
            "--config", self.config_path,
        ]
        if self.model_key:
            command.extend(["--model-key", self.model_key])

        print(f"Subprocess command: {' '.join(command)}")

        try:
            # Use Popen to start the process without blocking
            # Redirect stdout/stderr if desired, otherwise they might mix with main script
            if self.debug_server: print(">>> DEBUG: Attempting subprocess.Popen...")
            self._server_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE, # Capture stdout
                stderr=subprocess.PIPE, # Capture stderr
                text=True, # Decode stdout/stderr as text
                # Close file descriptors in child process on Unix-like systems
                close_fds=True
            )
            if self.debug_server: print(f">>> DEBUG: subprocess.Popen succeeded. PID: {self._server_process.pid}")
            print(f"Server process launched with PID: {self._server_process.pid}")

            # Wait briefly and check if the server became responsive
            print(f"Waiting up to {wait_time} seconds for server to respond...")
            time.sleep(1) # Initial short sleep before polling
            responsive = False
            stdout = "" # Initialize stdout variable
            stderr = "" # Initialize stderr variable
            for i in range(int(wait_time)): # Check roughly once per second
                if self.debug_server: print(f">>> DEBUG: Wait loop iteration {i+1}/{int(wait_time)}")
                if self.debug_server: print(f">>> DEBUG: Checking server running. self._server_process is {'set' if self._server_process else 'None'}")
                if self._server_process:
                    if self.debug_server: print(f">>> DEBUG: Polling process {self._server_process.pid}...")
                    poll_result = self._server_process.poll()
                    if self.debug_server: print(f">>> DEBUG: Poll result: {poll_result}")
                else:
                     poll_result = None # Cannot poll if handle is None

                if self.is_server_running(timeout=0.5):
                    responsive = True
                    print("Server process responded to health check successfully.")
                    break
                # Check if the process died unexpectedly (check poll_result we got above)
                # Check if self._server_process is still valid before using poll_result
                if self._server_process and poll_result is not None:
                    if self.debug_server: print(f">>> DEBUG: Process terminated block entered. Code: {poll_result}")
                    print(f"ERROR: Server process terminated unexpectedly with code {poll_result}. Reading output...")
                    # Read remaining output from terminated process
                    try:
                        if self.debug_server: print(">>> DEBUG: Attempting communicate()...")
                        stdout, stderr = self._server_process.communicate(timeout=1)
                        if self.debug_server: print(">>> DEBUG: communicate() finished.")
                        if stdout: print(f"--- Subprocess Stdout (Unexpected Exit) ---\\n{stdout}")
                        if stderr: print(f"--- Subprocess Stderr (Unexpected Exit) ---\n{stderr}")
                    except subprocess.TimeoutExpired:
                        print("Timed out reading output from terminated process.")
                    except Exception as e:
                        print(f"Error reading output from terminated process: {e}")
                    if self.debug_server: print(">>> DEBUG: Clearing server process handle after termination.")
                    self._server_process = None # Clear the handle
                    raise RuntimeError("Server subprocess failed to start properly.")
                if self.debug_server: print(f">>> DEBUG: Sleeping 1 second...")
                time.sleep(1)

            if not responsive:
                print(f"WARNING: Server process did not respond within {wait_time} seconds.")
                # If the process is still running but unresponsive, try to grab its output
                if self._server_process and self._server_process.poll() is None:
                     print("Attempting to read output from non-responsive process...")
                     # Non-blocking read (might not capture everything if process is stuck)
                     try:
                         # Note: communicate() waits, which we don't want if it's just unresponsive
                         # Accessing pipes directly might be complex. Let's just print a message
                         # and rely on kill_server potentially getting more later.
                         # A more robust solution might involve threading to monitor output.
                         print("(Output reading from non-responsive process is limited in this implementation)")
                         pass # Cannot easily get output without blocking or killing here

                         # If we decide to kill it here to get output:
                         # print("Killing unresponsive process to retrieve output...")
                         # self.kill_server() # This might print output during termination
                         
                     except Exception as e:
                         print(f"Error attempting to check/kill unresponsive process: {e}")
                
                # Also try to get output if the process terminated *during* the loop but *after* the poll() check
                # (handles a race condition)
                elif self._server_process and self._server_process.poll() is not None:
                    print(f"Server process terminated between checks (Code: {self._server_process.poll()}). Reading output...")
                    try:
                        stdout, stderr = self._server_process.communicate(timeout=1)
                        if stdout: print(f"--- Subprocess Stdout (Late Exit) ---\\n{stdout}")
                        if stderr: print(f"--- Subprocess Stderr (Late Exit) ---\\n{stderr}")
                    except Exception as e:
                         print(f"Error reading output from late-terminated process: {e}")
                    self._server_process = None # Clear handle
                
                # Optional: Kill the unresponsive process if it's still somehow running
                if self._server_process and self._server_process.poll() is None:
                    print("Killing unresponsive server process...")
                    self.kill_server() # Attempt to kill it

                # Finally, raise an error indicating timeout
                raise TimeoutError(f"Server process failed to respond within {wait_time} seconds.")
        except Exception as e:
            if self.debug_server: print(f">>> DEBUG: Exception caught in start_server_process: {e.__class__.__name__}: {e}")
            print(f"ERROR: Failed to start server process - {e.__class__.__name__}: {e}")
            if self.debug_server: print(f">>> DEBUG: Clearing server process handle in except block.")
            self._server_process = None # Ensure handle is cleared on error
            raise # Re-raise the exception

    def _run_server_blocking(self):
        """
        INTERNAL USE ONLY: Runs the blocking Uvicorn server.
        This method is intended to be called ONLY when the script is executed
        as a subprocess with the '--internal-run-server' argument.
        """
        if not FASTAPI_AVAILABLE or uvicorn is None:
            print("ERROR (_run_server_blocking): FastAPI/Uvicorn not available.")
            sys.exit(1) # Exit subprocess if dependencies missing

        # Create the app instance just before running
        # Ensure model config is loaded correctly (should be via __init__)
        try:
            if not self.app:
                print("(_run_server_blocking): Creating FastAPI app...")
                self._create_fastapi_app()
            if not self.app:
                 raise RuntimeError("Failed to create FastAPI app instance.")

            print(f"(_run_server_blocking): Starting Uvicorn on http://{self.server_host}:{self.server_port}")
            uvicorn.run(
                self.app,
                host=self.server_host,
                port=self.server_port,
                log_level="info"
            )
            print("(_run_server_blocking): Uvicorn server stopped.")
        except Exception as e:
             print(f"ERROR (_run_server_blocking): Failed to start Uvicorn - {e.__class__.__name__}: {e}")
             sys.exit(1) # Exit subprocess with error


# ========================================================================
# --- Example Usage (`if __name__ == "__main__":`) ---

async def run_direct_call_example(image_path):
    print(f"Running direct call example for image: {image_path}")
    # Placeholder - replace with actual example logic if needed
    connector = LlamaVisionConnector(auto_start=False) # Init for direct use
    try:
        response = await connector.get_response(image_path=image_path)
        print("\n--- Direct Call Response ---")
        print(response)
        print("--- End Direct Call Response ---")
    except Exception as e:
        print(f"Error during direct call: {e}")
    # pass

# --- Function to continuously read and print from a pipe ---
def pipe_reader(pipe, prefix, stop_event):
    try:
        # Use iter(pipe.readline, '') for line-by-line reading with text=True
        for line in iter(pipe.readline, ''):
            if stop_event.is_set():
                break
            if line:
                # line is already a string because text=True was used in Popen
                print(f"{prefix}: {line.strip()}")
    except Exception as e:
        # Pipe might close unexpectedly when process terminates
        if not stop_event.is_set(): # Avoid printing error if we stopped it
             print(f"Error reading from {prefix} pipe: {e}")
    finally:
        # print(f"{prefix} reader thread finished.") # Optional debug
        pass

if __name__ == "__main__":
    import argparse

    # --- Argument Parsing --- Reflects User Intent (Direct vs. Managed Server) ---
    parser = argparse.ArgumentParser(
        description="Llama Vision Connector: Use directly or run as a managed server.",
        formatter_class=argparse.RawTextHelpFormatter # Preserve formatting in help
    )

    # User-facing Mode Selection (Optional Positional Argument)
    # Defaults to 'direct' if no mode is specified.
    parser.add_argument(
        'mode',
        nargs='?', # Makes the argument optional
        choices=['direct', 'server'],
        default='direct',
        help=(
            "Operation mode:\n"
            "  direct: Initialize for direct use within a script (default).\n"
            "  server: Start and manage the API server as a background process.\n"
            "          The main script stays running to manage the server."
        )
    )

    # Internal flag used by subprocess - Hidden from user help
    parser.add_argument(
        '--internal-run-server',
        action='store_true',
        help=argparse.SUPPRESS # Hide this implementation detail from help message
    )

    # Arguments primarily for 'direct' mode
    parser.add_argument("-i", "--image", default="test_image.jpg", # <--- CHANGE DEFAULT PATH
                        help="Path to the image file (used in 'direct' mode example)." + 
                             " (Default: %(default)s)")

    # Arguments primarily for 'server' mode (passed to both manager and internal server)
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host address for the server process. (Default: %(default)s)")
    parser.add_argument("--port", type=int, default=8001,
                        help="Port number for the server process. (Default: %(default)s)")
    parser.add_argument("--config", default="config/models.json",
                        help="Path to the configuration file. (Default: %(default)s)")
    parser.add_argument("--model-key", default=None,
                        help="Specific model key from config to use. (Default: first model)")
    parser.add_argument("--no-auto-start", action='store_true',
                        help="(Server mode only) Prevent automatically starting the server process on init.")

    args = parser.parse_args()

    # --- Mode Execution ---
    print(f"--- Llama Vision Connector Runner ---")

    # Check for the internal flag first - this indicates a subprocess run
    if args.internal_run_server:
        # --- Internal Subprocess Mode --- Run the blocking server
        # This mode is NOT intended for direct user invocation.
        print(f"INTERNAL MODE: Starting blocking Uvicorn server..." + 
              f" (Host: {args.host}, Port: {args.port}, Config: {args.config}, Model: {args.model_key})")
        try:
            # Initialize connector *without* auto-starting (it IS the server process)
            internal_connector = LlamaVisionConnector(
                config_path=args.config,
                model_key=args.model_key,
                auto_start=False, # Crucial: The subprocess doesn't start another process
                server_host=args.host,
                server_port=args.port
            )
            # Directly call the blocking server run method
            internal_connector._run_server_blocking()
            # If _run_server_blocking finishes, the server stopped normally.
            print("INTERNAL MODE: Server stopped normally.")
            sys.exit(0) # Exit subprocess cleanly
        except Exception as e:
            # Log error before exiting subprocess
            print(f"FATAL ERROR in --internal-run-server mode: {e.__class__.__name__}: {e}", file=sys.stderr)
            sys.exit(1) # Exit subprocess with error code

    # --- User-facing Modes --- (If not internal run)
    elif args.mode == 'direct':
        # --- Direct Call Mode --- User intends to use the class directly (example runs here)
        print("MODE: Direct (Example Usage)")
        print(f"Test Image Path: {args.image}")
        try:
            # Initialize connector for direct usage (no server process started here)
            direct_connector = LlamaVisionConnector(
                 config_path=args.config,
                 model_key=args.model_key,
                 auto_start=False # Important: Direct mode doesn't manage a server process
            )
            # Run the example async function
            asyncio.run(run_direct_call_example(args.image))
        except KeyboardInterrupt:
             print("\nDirect call example interrupted by user.")
        except Exception as e:
             print(f"ERROR in direct mode execution: {e.__class__.__name__}: {e}", file=sys.stderr)
             sys.exit(1)

    elif args.mode == 'server':
        # --- Server Management Mode --- User wants a managed background server
        print("MODE: Server (Managed Background Process)")
        print(f"Config Path: {args.config}")
        print(f"Model Key: {args.model_key or '(Default First Model)'}")
        print(f"Server Host: {args.host}")
        print(f"Server Port: {args.port}")
        print(f"Auto-Starting Server Process: {not args.no_auto_start}")
        manager_connector = None
        stdout_thread = None
        stderr_thread = None
        stop_reader_event = threading.Event()

        try:
            manager_connector = LlamaVisionConnector(
                config_path=args.config,
                model_key=args.model_key,
                auto_start=(not args.no_auto_start),
                server_host=args.host,
                server_port=args.port
            )
            print("Connector initialized in server mode.")
            if not args.no_auto_start:
                print("Attempted to start background server process (check logs above for success/failure).",
                      "Use Ctrl+C to stop managing the server.")
            else:
                print("Server process NOT started automatically (--no-auto-start used).",
                      "You can start it manually via connector.start_server_process().",
                      "Use Ctrl+C to exit manager.")

            print("Main script is now running in management mode. Monitoring server process...")

            if manager_connector and hasattr(manager_connector, '_server_process') and manager_connector._server_process:
                server_proc = manager_connector._server_process

                # --- Start separate threads for reading stdout and stderr --- #
                print("Starting pipe reader threads...")
                stdout_thread = threading.Thread(
                    target=pipe_reader,
                    args=(server_proc.stdout, "[Server STDOUT]", stop_reader_event),
                    daemon=True # Allows main thread to exit even if these are running
                )
                stderr_thread = threading.Thread(
                    target=pipe_reader,
                    args=(server_proc.stderr, "[Server STDERR]", stop_reader_event),
                    daemon=True
                )
                stdout_thread.start()
                stderr_thread.start()
                # --- End starting threads --- #

                # Main thread now just waits for process termination or KeyboardInterrupt
                while server_proc.poll() is None:
                    try:
                        # Sleep briefly to avoid busy-waiting
                        time.sleep(0.5)
                    except KeyboardInterrupt:
                        print("\nCtrl+C received by main thread. Initiating shutdown...")
                        # Signal reader threads to stop (optional, as they are daemons)
                        stop_reader_event.set()
                        # Let atexit handle killing the server process
                        raise # Re-raise KeyboardInterrupt to exit gracefully

                # Process finished, signal threads and wait briefly for them
                print(f"\n--- Server process (PID: {server_proc.pid}) terminated with code: {server_proc.poll()} ---")
                stop_reader_event.set()
                # Give threads a moment to finish reading any final output
                if stdout_thread: stdout_thread.join(timeout=1.0)
                if stderr_thread: stderr_thread.join(timeout=1.0)
                print("Pipe reader threads signaled to stop.")

            else:
                print("ERROR: Server process handle not found after initialization. Cannot monitor.")
                while True:
                     try:
                          time.sleep(3600)
                     except KeyboardInterrupt:
                          print("\nCtrl+C received. Exiting.")
                          break
        except KeyboardInterrupt:
             # This catches Ctrl+C if it happens before the main monitoring loop starts
             print("\nCtrl+C received during setup. Exiting server management mode...")
        except Exception as e:
            print(f"ERROR in server management mode: {e.__class__.__name__}: {e}", file=sys.stderr)
            if stop_reader_event: stop_reader_event.set() # Signal threads on error too
            sys.exit(1)
        finally:
            # Ensure threads are signaled to stop if main loop exited unexpectedly
            if stop_reader_event: stop_reader_event.set()
            # Rely on atexit for final server process kill
            print("Exiting management mode.")

    print("\n--- Script Finished --- ({args.mode} mode)")