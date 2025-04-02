#!/usr/bin/env python3
import asyncio
import os
import base64
import sys
import requests # Needed for server test
import time     # Needed for server test delay

# Add the root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correct import path
from llama_vision_connector import LlamaVisionConnector

def get_test_image_path():
    """Get the path to the test image, relative to the root directory"""
    # Construct path relative to *this script's* location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "test_images", "sample.jpg")

async def test_direct_vision():
    """Test direct vision model interaction using get_response"""
    print("\n=== Testing Direct Vision Model Interaction (using get_response) ===")

    # Initialize connector without auto-starting server
    connector = LlamaVisionConnector(
        config_path="config/models.json",  # Path relative to root
        model_key=None,  # Will use first vision model found in config
        auto_start=False # We will call the CLI directly
    )

    try:
        image_path = get_test_image_path()
        if not os.path.exists(image_path):
            print(f"Error: Test image not found at {image_path}. Skipping direct test.")
            return

        print(f"\nTesting with image: {image_path}")

        # Test basic vision response using the direct get_response method
        prompt = "Describe this image in detail."
        print(f"Prompt: {prompt}")
        response = await connector.get_response(
            image_path=image_path,
            prompt=prompt
        )
        print(f"\nVision Response:\n{response}")

        # --- Test using prompt_file argument with root vision-prompt.txt ---
        # Construct path relative to the *script's* directory to find the root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, '..')) # Go up one level from examples/
        prompt_file_path = os.path.join(root_dir, connector.vision_prompt_filename)

        print(f"\nTesting with prompt file from root: {prompt_file_path}")
        # Assuming the prompt file exists in the root, remove the check
        if not os.path.exists(prompt_file_path):
             print(f"WARNING: Root prompt file not found at {prompt_file_path}. Skipping file-prompted test.")
        else:
             response_file_prompt = await connector.get_response(
                 image_path=image_path,
                 prompt_file=prompt_file_path # Use prompt_file argument pointing to root file
             )
             print(f"\nVision Response (from root prompt file):\n{response_file_prompt}")

    except Exception as e:
        print(f"Error during direct vision testing: {e}")
    # No cleanup needed here as server wasn't started

async def test_vision_server():
    """Test vision model interaction through the auto-started FastAPI server"""
    print("\n=== Testing Vision Model via Auto-Started FastAPI Server ===")

    # Initialize connector WITH auto-starting server
    # The connector instance will manage the server process lifecycle
    connector = LlamaVisionConnector(
        config_path="config/models.json",  # Path relative to root
        model_key=None,  # Will use first vision model
        auto_start=True, # Start the server process
        server_host="127.0.0.1",
        server_port=8001, # Initial port, connector finds available one
        debug_server=True # Enable debug prints for server part of test
    )

    # Note: connector.kill_server() will be called automatically on exit via atexit

    try:
        # Wait briefly for server to initialize (connector already does basic checks)
        print("Waiting a moment for server to fully initialize...")
        await asyncio.sleep(5) # Increased wait time

        # Check if the server process managed by the connector is running
        if connector.is_server_running(timeout=2.0):
            print("Server process appears to be running.")
            server_url = f"http://{connector.server_host}:{connector.server_port}/v1/chat/completions"
            print(f"Attempting to connect to server endpoint: {server_url}")
        else:
            print("Error: Server process failed to start or is not responding after wait. Check logs.")
            # Attempt to kill any lingering process just in case
            connector.kill_server()
            return

        image_path = get_test_image_path()
        if not os.path.exists(image_path):
            print(f"Error: Test image not found at {image_path}. Skipping server test.")
            return

        # Read and encode the image for the OpenAI format
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            image_url_data = f"data:image/jpeg;base64,{image_data}"

        # Prepare OpenAI-compatible request payload
        request_data = {
            "model": connector.model_key, # Use the model key from the connector
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What animal is in this image? Be concise."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url_data
                            }
                        }
                    ]
                }
            ],
            # Add other params if needed, matching server expectations or OpenAI spec
            # "temperature": 0.7,
            # "max_tokens": 100
        }

        print("\nSending request to server...")
        # Use requests library to send the POST request (sync call in async context)
        response = await asyncio.to_thread(requests.post, server_url, json=request_data, timeout=60)

        print(f"Server Response Status Code: {response.status_code}")

        if response.status_code == 200:
            print("\nServer Response JSON:")
            try:
                print(response.json())
            except requests.exceptions.JSONDecodeError:
                print("Error: Could not decode JSON response from server.")
                print("Raw Response Text:", response.text)
        else:
            print("\nError: Server returned non-200 status code.")
            print("Response Text:", response.text)

    except Exception as e:
        print(f"An error occurred during server testing: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging

    # No explicit connector.kill_server() needed here because of atexit registration
    print("\nServer test finished. Server process should shut down automatically on script exit.")


async def main():
    """Main test function"""
    # Test direct vision interaction (calls CLI directly)
    await test_direct_vision()

    # Reset the singleton instance before the server test
    print("\nResetting LlamaVisionConnector singleton for server test...")
    LlamaVisionConnector._instance = None
    LlamaVisionConnector._server_process = None # Also ensure the process handle is cleared

    # Test server mode (starts FastAPI server, sends HTTP request)
    await test_vision_server()

if __name__ == "__main__":
    asyncio.run(main()) 