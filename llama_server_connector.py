import subprocess
import time
import requests
import json
import os
import openai
from typing import Dict, List, Any, Optional
import atexit
import argparse
import base64
import mimetypes

class LlamaServerConnector:
    """
    A class to call a llama.cpp model with given parameters, using a singleton llama-server instance.
    """
    _instance = None
    _process = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LlamaServerConnector, cls).__new__(cls)
            cls._instance._process = None # Ensure process is None on new instance
        return cls._instance

    def __init__(self, 
                 config_path: str = "config/models.json",
                 model_key: str = None,
                 param_overrides: Optional[Dict[str, Any]] = None,
                 initial_port: int = 8080,
                 host: str = "127.0.0.1",
                 max_attempts: int = 10,
                 attempt_delay: int = 1,
                 debug_server: bool = False,
                 client_timeout: Optional[float] = None):
        """
        Initialize the LlamaServerConnector with configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration JSON file
            model_key (str, optional): Key of the model to use from config.
            param_overrides (Dict[str, Any], optional): Parameter overrides.
            initial_port (int): Initial port to try for the server.
            host (str): Host address for the server.
            max_attempts (int): Maximum number of attempts to connect.
            attempt_delay (int): Delay between connection attempts.
            debug_server (bool): Enable detailed debug printing.
            client_timeout (float, optional): Manual override for OpenAI client timeout (seconds).
                                              If None, checks model config. 
                                              If missing in config, defaults to None (Infinite).
        """
        # Initialization guard
        if hasattr(self, 'initialized') and self.initialized:
            if self.debug_server: print(">>> DEBUG: LlamaServerConnector already initialized. Skipping re-init.")
            return

        # Proceed with initialization only if instance doesn't exist or is not initialized
        print(f"Initializing LlamaServerConnector...")
        self.debug_server = debug_server
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.initial_port = initial_port
        
        # Network configuration
        self.host = host
        self.max_attempts = max_attempts
        self.attempt_delay = attempt_delay
        
        # Set up model configuration
        models_config = self.config.get("MODELS", {})
        if not models_config:
            raise ValueError("No models found in configuration")
                
        # If no model key is provided, use first model
        if model_key is None and models_config:
            model_key = next(iter(models_config.keys()))
        
        if model_key not in models_config:
            raise ValueError(f"Model '{model_key}' not found in configuration")
        
        self.model_config = models_config[model_key]
        self.model_key = model_key
        self.model_path = self.model_config.get("MODEL_PATH")
        
        # --- TIMEOUT LOGIC ---
        # Priority: 1. Constructor/CLI Arg -> 2. Specific Model Config -> 3. Default (Infinite)
        if client_timeout is not None:
            self.client_timeout = client_timeout
            print(f"Client configuration: Timeout set to {self.client_timeout}s (Manual Override).")
        else:
            # Look for CLIENT_TIMEOUT inside the specific model's config
            self.client_timeout = self.model_config.get("CLIENT_TIMEOUT", None)
            if self.client_timeout:
                print(f"Client configuration: Timeout set to {self.client_timeout}s (From model config).")
            else:
                print("Client configuration: Timeout set to Infinite (None).")
        
        # Apply parameter overrides if provided
        if param_overrides:
            print(f"Applying parameter overrides: {param_overrides}")
            self.model_config.update(param_overrides)
            
        # Find available port and set server address
        self.urlport = self.find_available_port(self.initial_port, self.host)
        self.server_address = f"http://{host}:{self.urlport}/v1"
        
        # Start server
        self.start_server()
        self.initialized = True

        # Register kill_server to be called on script exit
        atexit.register(self.kill_server)


    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file: {config_path}")

    def set_model(self, model_key: str, param_overrides: Optional[Dict[str, Any]] = None) -> None:
        """
        Change the model configuration and restart the server.
        
        Args:
            model_key (str): Key of the model in the configuration
            param_overrides (Dict[str, Any], optional): Parameter overrides from config.json
        """
        if model_key not in self.config.get("MODELS", {}):
            raise ValueError(f"Model '{model_key}' not found in configuration")
            
        # Kill existing server
        self.kill_server()
        
        # Update model configuration
        self.model_config = self.config["MODELS"][model_key]
        self.model_key = model_key
        self.model_path = self.model_config.get("MODEL_PATH")

        # Update timeout from the new model config (unless manually forced in a way we want to persist, 
        # but standard behavior here is to adopt the new model's traits)
        new_timeout = self.model_config.get("CLIENT_TIMEOUT", None)
        self.client_timeout = new_timeout
        if self.client_timeout:
            print(f"Model changed. Client timeout updated to {self.client_timeout}s.")
        else:
            print(f"Model changed. Client timeout set to Infinite.")
        
        # Apply parameter overrides if provided
        if param_overrides:
            print(f"Applying parameter overrides for model change: {param_overrides}")
            self.model_config.update(param_overrides)
        
        # Reinitialize server
        self.urlport = self.find_available_port(self.initial_port, self.host)
        self.server_address = f"http://{self.host}:{self.urlport}/v1"
        self.start_server()

    def find_available_port(self, initial_port: int, host: str) -> int:
        """Find an available port starting from initial_port."""
        port = initial_port
        while True:
            try:
                response = requests.get(f'http://{host}:{port}/v1/models')
                if response.status_code != 200:
                    return port
                else:
                    port += 1
            except requests.ConnectionError:
                return port

    def build_server_command(self) -> List[str]:
        """Build the llama-server command with all necessary parameters from config."""
        # Default command with model path
        cmd = [
            "llama-server",
            "-m", self.model_path,
        ]
        
        # Add GPU layers
        cmd.extend(["-ngl", str(self.model_config.get("NUM_LAYERS_TO_GPU", 99))])
        
        # Add temperature
        cmd.extend(["--temp", str(self.model_config.get("TEMPERATURE", 0.3))])
        
        # Add forced alignment and sampling mode
        cmd.extend(["-fa", "on"])
        
        # Add chat template if specified
        chat_template = self.model_config.get("CHAT_TEMPLATE")
        if chat_template:
            cmd.extend(["--chat-template", chat_template])
        
        # Add context size
        cmd.extend(["-c", str(self.model_config.get("NUM_TOKENS_OF_CONTEXT", 65536))])
        
        # Add cache types
        cmd.extend(["-ctk", self.model_config.get("CACHE_TYPE_K", "q8_0")])
        cmd.extend(["-ctv", self.model_config.get("CACHE_TYPE_V", "q8_0")])
        
        # Add port
        cmd.extend(["--port", str(self.urlport)])
        
        # add the mmproj path if it exists
        if "MMPROJ_PATH" in self.model_config:
            cmd.extend(["--mmproj", self.model_config["MMPROJ_PATH"]])
        
        return cmd

    def is_server_running(self) -> bool:
        """Check if the llama-server is running and responding."""
        try:
            response = requests.get(f"{self.server_address}/models")
            if response.status_code == 200:
                # Get model name from config or fallback to the path basename
                model_name = os.path.basename(self.model_path)
                
                # Test the server with a simple completion request
                test_response = requests.post(
                    f"{self.server_address}/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "temperature": 0.4,
                        "max_tokens": 5
                    }
                )
                return test_response.status_code == 200
            return False
        except Exception as e:
            print(f"Server check failed: {str(e)}")
            return False

    def start_server(self):
        """Start the llama-server if it's not already running."""
        if not self.is_server_running():
            cmd = self.build_server_command()
            print(f"Starting server with command: {' '.join(cmd)}")
            
            if self._process is None or self._process.poll() is not None:
                if self.debug_server: print(">>> DEBUG: Attempting subprocess.Popen...")
                self._process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if self.debug_server: print(f">>> DEBUG: subprocess.Popen succeeded. PID: {self._process.pid}")
                print(f"Server process started with PID: {self._process.pid}")
                
                attempts = 0
                while True:
                    if self.debug_server: print(f">>> DEBUG: Startup attempt {attempts + 1}/{self.max_attempts}...")
                    if self.is_server_running():
                        print(f"Server startup completed after {attempts + 1} attempts with PID {self._process.pid} on port {self.urlport}")
                        time.sleep(0.5)
                        break

                    if self._process.poll() is not None:
                        poll_result = self._process.poll()
                        if self.debug_server: print(f">>> DEBUG: Server process terminated prematurely. Code: {poll_result}")
                        print(f"ERROR: Server process terminated unexpectedly with code {poll_result}. Reading output...")
                        stdout, stderr = "", ""
                        try:
                            if self.debug_server: print(">>> DEBUG: Attempting communicate() on premature exit...")
                            stdout, stderr = self._process.communicate(timeout=1) 
                            if self.debug_server: print(">>> DEBUG: communicate() finished.")
                            if stdout: print(f"--- Subprocess Stdout (Premature Exit) ---\n{stdout}")
                            if stderr: print(f"--- Subprocess Stderr (Premature Exit) ---\n{stderr}")
                        except Exception as comm_e:
                             print(f"Error reading output from prematurely terminated process: {comm_e}")
                        self._process = None
                        raise RuntimeError(f"Server process failed to start properly, terminated with code {poll_result}.")
                    
                    if attempts >= self.max_attempts:
                        if self.debug_server: print(">>> DEBUG: Max attempts reached.")
                        stdout, stderr = "", ""
                        if self._process:
                             poll_result_final = self._process.poll()
                             if poll_result_final is None:
                                  if self.debug_server: print(">>> DEBUG: Process unresponsive. Attempting to kill and get output...")
                                  try:
                                       self._process.kill()
                                       stdout, stderr = self._process.communicate(timeout=2)
                                       if stdout: print(f"--- Subprocess Stdout (Unresponsive Killed) ---\n{stdout}")
                                       if stderr: print(f"--- Subprocess Stderr (Unresponsive Killed) ---\n{stderr}")
                                  except Exception as kill_comm_e:
                                       print(f"Error getting output after killing unresponsive process: {kill_comm_e}")
                             else:
                                  if self.debug_server: print(f">>> DEBUG: Process terminated between checks. Code: {poll_result_final}. Reading output...")
                                  try:
                                       stdout, stderr = self._process.communicate(timeout=1) 
                                       if stdout: print(f"--- Subprocess Stdout (Late Exit) ---\n{stdout}")
                                       if stderr: print(f"--- Subprocess Stderr (Late Exit) ---\n{stderr}")
                                  except Exception as late_comm_e:
                                       print(f"Error reading output from late-terminated process: {late_comm_e}")
                        self._process = None
                        raise RuntimeError(f"Server startup failed after {self.max_attempts} attempts.")
                        
                    if self.debug_server: print(f">>> DEBUG: Sleeping for {self.attempt_delay}s...")
                    time.sleep(self.attempt_delay)
                    attempts += 1
                
            else:
                if self.debug_server: print(">>> DEBUG: start_server called but process exists and is running (PID {self._process.pid if self._process else 'N/A'}). Skipping Popen.")

    def kill_server(self):
        """Kill the llama-server process and clean up."""
        if self.debug_server: print(f">>> DEBUG: kill_server called. self._process is {'set' if self._process else 'None'}")
        try:
            if self._process is not None and self._process.poll() is None:
                pid = self._process.pid
                if self.debug_server: print(f">>> DEBUG: Attempting to kill process {pid}...")
                self._process.kill()
                try:
                    self._process.wait(timeout=1.0)
                    if self.debug_server: print(f">>> DEBUG: Process {pid} confirmed terminated after kill.")
                except subprocess.TimeoutExpired:
                    if self.debug_server: print(f">>> DEBUG: Process {pid} did not terminate immediately after kill (Timeout). Force clearing handle.")
                except Exception as wait_e:
                     if self.debug_server: print(f">>> DEBUG: Error waiting for process {pid} termination: {wait_e}")

                print(f"kill_server: Server process (PID {pid}) killed.")
            elif self._process:
                 if self.debug_server: print(f">>> DEBUG: Server process (PID {self._process.pid}) already terminated (poll result: {self._process.poll()}).")
            else:
                 if self.debug_server: print(">>> DEBUG: No server process handle to kill.")

        except Exception as e:
            print(f"kill_server: Warning during kill: {str(e)}")
        finally:
            if self.debug_server: print(">>> DEBUG: Clearing process handle and singleton instance in finally block.")
            self._process = None
            LlamaServerConnector._instance = None
            print("kill_server: Cleanup complete.")

    def get_server_url(self) -> str:
        """Return the server URL for use with the OpenAI client."""
        return self.server_address
        
    def get_response(self, prompt: str, api_key: str = None, image_path: str = None, image_paths: List[str] = None) -> Optional[str]:
        """
        Send a prompt to the llama-server and get the response.
        
        Args:
            prompt (str): The input prompt to send to the model
            api_key (str, optional): API key if your server requires authentication
            image_path (str, optional): Path to a single image (legacy support)
            image_paths (List[str], optional): List of paths to images to send to the model
            
        Returns:
            str: The model's response text or None if there was an error
        """
        # Get model name from the path basename
        model = os.path.basename(self.model_path)
        
        # Configure the client
        # Uses self.client_timeout resolved in __init__ (or set_model)
        client = openai.OpenAI(
            base_url=self.server_address,
            api_key=api_key or "not-needed",
            timeout=self.client_timeout 
        )
        
        content = [{"type": "text", "text": prompt}]
        
        # Handle single image path (legacy)
        if image_path and not image_paths:
            image_paths = [image_path]
            
        if image_paths:
            for img_path in image_paths:
                try:
                    mime_type, _ = mimetypes.guess_type(img_path)
                    if mime_type is None:
                        mime_type = "image/jpeg"  # Default fallback
                    with open(img_path, "rb") as img_file:
                        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                    image_url = f"data:{mime_type};base64,{base64_image}"
                    content.append({"type": "image_url", "image_url": {"url": image_url}})
                except FileNotFoundError:
                    print(f"Error: Image file not found at {img_path}")
                    return None
                except Exception as e:
                    print(f"Error processing image {img_path}: {str(e)}")
                    return None
        
        try:
            # Send the completion request
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": content}
                ]
            )
            
            # Extract and return the response text
            return response.choices[0].message.content
        
        except openai.APITimeoutError:
            print(f"Error: The request timed out (Limit: {self.client_timeout}s).")
            return None
        except Exception as e:
            print(f"Error communicating with llm server: {str(e)}")
            return None

# Helper function to parse parameter overrides from command line
def parse_param_overrides(overrides_list: Optional[List[str]]) -> Dict[str, Any]:
    """Parses key=value pairs from a list into a dictionary."""
    overrides = {}
    if overrides_list:
        for item in overrides_list:
            try:
                key, value_str = item.split('=', 1)
                key = key.strip()
                value_str = value_str.strip()
                # Attempt to convert value to appropriate type (int, float, bool, str)
                try:
                    value = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        if value_str.lower() == 'true':
                            value = True
                        elif value_str.lower() == 'false':
                            value = False
                        else:
                            value = value_str # Keep as string if no other type matches
                overrides[key] = value
            except ValueError:
                print(f"Warning: Could not parse override '{item}'. Expected format KEY=VALUE. Skipping.")
    return overrides

# Example usage with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LlamaServerConnector with command-line arguments.")
    
    # Arguments for LlamaServerConnector initialization
    parser.add_argument("--config-path", type=str, default="config/models.json", help="Path to the configuration JSON file.")
    parser.add_argument("--model-key", type=str, default=None, help="Key of the model to use from config. If None, uses the first non-vision model.")
    parser.add_argument("--override", action='append', help="Parameter overrides for the model config (e.g., --override TEMPERATURE=0.5 --override NUM_TOKENS_TO_OUTPUT=100).")
    parser.add_argument("--initial-port", type=int, default=8080, help="Initial port to try for the server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for the server.")
    parser.add_argument("--max-attempts", type=int, default=10, help="Maximum connection attempts.")
    parser.add_argument("--attempt-delay", type=int, default=1, help="Delay between connection attempts.")
    parser.add_argument("--debug-server", action='store_true', help="Enable server debug prints.")
    
    # NEW ARGUMENT
    parser.add_argument("--client-timeout", type=float, default=None, help="Timeout in seconds for the OpenAI client. Default is None (infinite).")

    # Argument for the prompt
    parser.add_argument("--prompt", type=str, required=False, default=None, help="Optional prompt to send to the model for a single interaction (Direct Mode). If omitted, the server starts and runs until interrupted (Server Mode).")
    
    args = parser.parse_args()

    # Parse overrides from the collected list
    param_overrides = parse_param_overrides(args.override)
    
    connector = None # Initialize connector to None for error handling
    try:
        # Initialize the server with configuration from command line args
        # This happens regardless of mode, as server startup is common
        connector = LlamaServerConnector(
            config_path=args.config_path,
            model_key=args.model_key,
            param_overrides=param_overrides,
            initial_port=args.initial_port,
            host=args.host,
            max_attempts=args.max_attempts,
            attempt_delay=args.attempt_delay,
            debug_server=args.debug_server,
            client_timeout=args.client_timeout # Pass CLI argument
        )
        
        # Get the server URL (just for information)
        server_url = connector.get_server_url()
        print(f"Server URL: {server_url}")

        # Decide mode based on prompt presence
        if args.prompt:
            # Direct Mode: Send prompt, get response, then exit (atexit handles cleanup)
            print(f"\nRunning in Direct Mode...")
            print(f"Sending prompt: '{args.prompt}'")
            response = connector.get_response(args.prompt)
            print(f"\nResponse:\n{response}")
            print("Direct mode finished. Server will shut down.")
        else:
            # Server Mode: Keep script running until interrupted
            print(f"\nRunning in Server Mode. Server is active.")
            print("Press Ctrl+C to shut down the server.")
            try:
                while True:
                    time.sleep(1) # Keep the main thread alive
            except KeyboardInterrupt:
                print("\nCtrl+C detected. Shutting down server...")
                # atexit handler will call kill_server() automatically
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Attempt cleanup if connector was partially initialized and process started
        # Check if connector and _process exist and if process is still running
        if connector is not None and hasattr(connector, '_process') and connector._process is not None and connector._process.poll() is None:
            print("Attempting to clean up server process due to error...")
            connector.kill_server()
        else:
            print("Exiting due to error.")

    # Note: atexit handles the final cleanup if the script exits normally or via Ctrl+C after successful init