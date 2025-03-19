import subprocess
import time
import requests
import json
import os
import openai
from typing import Dict, List, Any, Optional

class LlamaServerConnector:
    """
    A class to call a llama.cpp model with given parameters, using a singleton llama-server instance.
    """
    _instance = None
    _process = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LlamaServerConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self, 
                 config_path: str = "config/models.json",
                 model_key: str = None,
                 initial_port: int = 8080,
                 host: str = "127.0.0.1",
                 max_attempts: int = 10,
                 attempt_delay: int = 1):
        """
        Initialize the LlamaServerConnector with configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration JSON file
            model_key (str, optional): Key of the model to use from config.
                                     If None, will look for a model with appropriate config.
            initial_port (int): Initial port to try for the server
            host (str): Host address for the server
            max_attempts (int): Maximum number of attempts to connect to the server
            attempt_delay (int): Delay between connection attempts in seconds
        """
        if not hasattr(self, 'initialized'):
            # Load configuration
            self.config = self._load_config(config_path)
            
            # Network configuration
            self.host = host
            self.initial_port = initial_port
            self.max_attempts = max_attempts
            self.attempt_delay = attempt_delay
            
            # Set up model configuration
            models_config = self.config.get("MODELS", {})
            if not models_config:
                raise ValueError("No models found in configuration")
                
            # If no model key is provided, use first model that doesn't require mmproj (non-vision model)
            if model_key is None:
                for key, model in models_config.items():
                    if "MMPROJ_PATH" not in model or not model.get("MMPROJ_PATH"):
                        model_key = key
                        break
                
                # If still no model found, use the first one
                if model_key is None and models_config:
                    model_key = next(iter(models_config.keys()))
            
            if model_key not in models_config:
                raise ValueError(f"Model '{model_key}' not found in configuration")
                
            self.model_config = models_config[model_key]
            self.model_key = model_key
            self.model_path = self.model_config.get("MODEL_PATH")
            
            if not self.model_path:
                raise ValueError(f"Model path not specified for model: {model_key}")
            
            # Find available port and set server address
            self.urlport = self.find_available_port(self.initial_port, self.host)
            self.server_address = f"http://{host}:{self.urlport}/v1"
            
            # Start server
            self.start_server()
            self.initialized = True

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration JSON file
            
        Returns:
            dict: Configuration as a dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file: {config_path}")

    def set_model(self, model_key: str) -> None:
        """
        Change the model configuration and restart the server.
        
        Args:
            model_key (str): Key of the model in the configuration
        """
        if model_key not in self.config.get("MODELS", {}):
            raise ValueError(f"Model '{model_key}' not found in configuration")
            
        # Kill existing server
        self.kill_server()
        
        # Update model configuration
        self.model_config = self.config["MODELS"][model_key]
        self.model_key = model_key
        self.model_path = self.model_config.get("MODEL_PATH")
        
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
            self.model_config.get("CLI_CMD", "llama-server"),
            "-m", self.model_path,
        ]
        
        # Add GPU layers
        cmd.extend(["-ngl", str(self.model_config.get("NUM_LAYERS_TO_GPU", 99))])
        
        # Add temperature
        cmd.extend(["--temp", str(self.model_config.get("TEMPERATURE", 0.3))])
        
        # Add forced alignment and sampling mode
        cmd.extend(["-fa", "-sm", "row"])
        
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
            
            if self._process is None or self._process.poll() is not None:
                self._process = subprocess.Popen(cmd)
                print(f"Server process started with PID: {self._process.pid}")
                
                attempts = 0
                while not self.is_server_running():
                    if attempts >= self.max_attempts:
                        raise RuntimeError(f"Server startup failed after {attempts} attempts.")
                    time.sleep(self.attempt_delay)
                    attempts += 1
                
                print(f"Server startup completed after {attempts} attempts with PID {self._process.pid} on port {self.urlport}")
                time.sleep(0.5)

    def kill_server(self):
        """Kill the llama-server process and clean up."""
        try:
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
                print("kill_server: Server process killed.")
        except Exception as e:
            print(f"kill_server: Warning: {str(e)}")
        finally:
            self._process = None
            LlamaServerConnector._instance = None
            print("kill_server: Cleanup complete.")

    def get_server_url(self) -> str:
        """Return the server URL for use with the OpenAI client."""
        return self.server_address
        
    def get_response(self, prompt: str, api_key: str = None) -> Optional[str]:
        """
        Send a prompt to the llama-server and get the response.
        
        Args:
            prompt (str): The input prompt to send to the model
            api_key (str, optional): API key if your server requires authentication
        
        Returns:
            str: The model's response text or None if there was an error
        """
        # Get model name from the path basename
        model = os.path.basename(self.model_path)
        
        # Configure the client
        client = openai.OpenAI(
            base_url=self.server_address,
            api_key=api_key or "not-needed"  # llama-server typically doesn't require an API key
        )
        
        try:
            # Send the completion request
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and return the response text
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error communicating with llm server: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize the server with configuration from config.json
    connector = LlamaServerConnector(
        config_path="config/models.json",
        model_key="DEEPSEEK-R1-QWEN-14B"  # Specify the model key from config
    )
    
    # Get the server URL (just for information)
    server_url = connector.get_server_url()
    print(f"Server URL: {server_url}")
    
    # Use the integrated method to get a response
    response = connector.get_response("Hello, how are you?")
    print(f"Response: {response}")
    
    # When done, kill the server
    connector.kill_server()