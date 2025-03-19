import subprocess
import asyncio
import os
import json
from typing import Dict, Any, Optional, Tuple, List

class LlamaVisionConnector:
    """
    A class to call a llama.cpp vision model with given parameters, using its CLI interface.
    """
    
    def __init__(self, config_path: str = "config/models.json", model_key: str = None):
        """
        Initialize a LlamaVisionConnector with configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration JSON file
            model_key (str, optional): Key of the model to use from config.
                                      If None, the first model in the config will be used.
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up basic configuration
        self.processing_folder = self.config.get("OUTPUT_FOLDER", "output")
        self.vision_prompt_filename = self.config.get("VISION_PROMPT_FILENAME", "vision-prompt.txt")
        self.valid_image_extensions = self.config.get("VALID_IMAGE_EXTENSIONS", ['.jpg', '.jpeg', '.png', '.gif'])
        
        # Set up model configuration
        models_config = self.config.get("MODELS", {})
        if not models_config:
            raise ValueError("No models found in configuration")
            
        # If no model key is provided, use the first model in the configuration
        if model_key is None:
            model_key = next(iter(models_config.keys()))
            
        if model_key not in models_config:
            raise ValueError(f"Model '{model_key}' not found in configuration")
            
        self.model_config = models_config[model_key]
        self.model_key = model_key
        
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
        Change the model configuration.
        
        Args:
            model_key (str): Key of the model in the configuration
        """
        if model_key not in self.config.get("MODELS", {}):
            raise ValueError(f"Model '{model_key}' not found in configuration")
            
        self.model_config = self.config["MODELS"][model_key]
        self.model_key = model_key
        
    async def get_response(self, image_path: str, prompt: str = None) -> Optional[str]:
        """
        Generate description for an image using a vision-enabled model.
        Process everything in memory, without creating temporary files.
        
        Args:
            image_path (str): Path to the image file
            prompt (str, optional): Custom prompt to use instead of the vision prompt file
        
        Returns:
            str: Generated description if successful, None otherwise
        """
        try:
            if not os.path.exists(image_path):
                print(f"ERROR - Image file not found: {image_path}")
                return None
                
            # Use provided prompt or load from file
            if prompt is None:
                # Find vision prompt file
                vision_prompt_path = self.vision_prompt_filename
                if not os.path.exists(vision_prompt_path):
                    vision_prompt_path = os.path.join(os.path.dirname(self.processing_folder), self.vision_prompt_filename)
                    if not os.path.exists(vision_prompt_path):
                        print(f"ERROR - Vision prompt file not found: {self.vision_prompt_filename}")
                        return None
                    
                # Get vision prompt content from file
                with open(vision_prompt_path, 'r') as f:
                    vision_prompt_content = f.read()
            else:
                # Use the provided prompt
                vision_prompt_content = prompt
            
            # Build command based on model configuration
            cli_cmd = self.model_config.get("CLI_CMD", "llama-gemma3-cli")
            model_path = self.model_config.get("MODEL_PATH")
            mmproj_path = self.model_config.get("MMPROJ_PATH")
            
            if "GEMMA3" in self.model_key or "QWEN2_VL" in self.model_key:
                # For models that support vision
                if not mmproj_path:
                    print(f"ERROR - MMPROJ path not specified for vision model: {self.model_key}")
                    return None
                    
                # Build the command for direct output (no file)
                command = [
                    cli_cmd, 
                    "-fa",
                    "-sm", "row",
                    "--image", image_path, 
                    "-p", vision_prompt_content,  # Use the prompt content directly
                    "-m", model_path, 
                    "--mmproj", mmproj_path,
                    "-ngl", str(self.model_config.get("NUM_LAYERS_TO_GPU", 99)),
                    "-n", str(self.model_config.get("NUM_TOKENS_TO_OUTPUT", 20000)),
                    "-c", str(self.model_config.get("NUM_TOKENS_OF_CONTEXT", 12772))
                ]
                
                # Add cache types if specified
                if "CACHE_TYPE_K" in self.model_config:
                    command.extend(["-ctk", self.model_config["CACHE_TYPE_K"]])
                if "CACHE_TYPE_V" in self.model_config:
                    command.extend(["-ctv", self.model_config["CACHE_TYPE_V"]])
                    
                # Add temperature if specified
                if "TEMPERATURE" in self.model_config:
                    command.extend(["--temp", str(self.model_config["TEMPERATURE"])])
                    
                # Print the command for debugging
                print(f"({image_path}) - Describing image...")
                
                # Run the command and capture output directly
                result = subprocess.run(
                    command, 
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True  # Return strings rather than bytes
                )
                
                # Process the output to extract just the model's response
                output_lines = result.stdout.splitlines()
                
                # Process lines to find where the actual output starts
                decode_line_index = -1
                
                # Gemma3: Find the line that starts with "Image decoded in "
                for i, line in enumerate(output_lines):
                    if line.strip().startswith("Image decoded in "):
                        decode_line_index = i
                        break
                
                # Qwen2-VL: Find the line that starts with "encode_image_with_clip: image encoded in "
                if decode_line_index == -1:
                    for i, line in enumerate(output_lines):
                        if line.strip().startswith("encode_image_with_clip: image encoded in "):
                            decode_line_index = i
                            break
                
                # Extract the actual output
                if decode_line_index != -1:
                    output_text = '\n'.join(output_lines[decode_line_index + 1:])
                else:
                    # Fallback if the marker line is not found
                    output_text = '\n'.join(output_lines)
                
                return output_text
            else:
                print(f"ERROR - Model {self.model_key} does not support image processing.")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"Error running model: {e.stderr}")
            return None
        except Exception as e:
            print(f"Unexpected error describing image: {str(e)}")
            return None

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


async def main():
    try:
        ## Example of using the LlamaVisionConnector class to process all images in the same folder as the script;
        ## the descriptions are saved to text files with the same name as the images.
        
        # Create a LlamaVisionConnector instance with a specific model (or use default)
        runner = LlamaVisionConnector(config_path="config/models.json", model_key="GEMMA3_12B")
        
        # Example of a custom prompt
        custom_prompt = "Describe this image in detail, focusing on the main subjects and any text content."
        
        # Process all images in the same folder as the script
        success_count = 0
        total_count = 0

        root_folder = os.path.dirname(os.path.abspath(__file__))  # Get the root folder of the script

        for filename in os.listdir(root_folder):
            # Use the helper method to check for valid image files
            if not runner.is_valid_image_file(filename):
                continue
                
            total_count += 1
            image_path = os.path.join(root_folder, filename)
            
            # Get image description with default prompt (from file)
            # To use custom prompt, uncomment the line below and comment out the one after it:
            #description = await runner.get_response(image_path, prompt=custom_prompt)
            description = await runner.get_response(image_path)
            
            # Save description to file if successful
            if description is not None:
                caption_filename = f"{os.path.splitext(image_path)[0]}.txt"
                with open(caption_filename, "w") as caption_file:
                    caption_file.write(description)
                success_count += 1
            else:
                print(f"Failed to process {filename}")
        
        print(f"Processing complete: {success_count}/{total_count} images processed successfully")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())