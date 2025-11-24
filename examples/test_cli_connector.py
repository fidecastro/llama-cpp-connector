#!/usr/bin/env python3
import asyncio
import os
import base64
import sys
# import requests # No longer needed for server test
import httpx    # <-- Import httpx
import time     # Needed for server test delay

# Add the root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correct import path
from llama_cli_connector import LlamaCLIConnector

def get_test_image_path():
    """Get the path to the test image, relative to the root directory"""
    # Construct path relative to *this script's* location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "test_images", "sample.jpg")

async def test_direct_vision():
    """Test direct vision model interaction using get_response"""
    print("\n=== Testing Direct Vision Model Interaction (using get_response) ===")

    # Initialize connector without auto-starting server
    connector = LlamaCLIConnector(
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

async def test_vision_server_external():
    """Test vision model interaction by connecting to an EXTERNALLY started server"""
    print("\n=== Testing Vision Model via EXTERNALLY Started FastAPI Server ===")

    # Configuration (matching default server settings)
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 8001
    # You might need to load config just to get the model key, or hardcode it for test
    # For simplicity, let's assume a default model or retrieve from config if needed
    # This connector instance is NOT used to manage the server, only potentially for config
    temp_connector_for_config = LlamaCLIConnector(config_path="config/models.json", auto_start=False)
    test_model_key = temp_connector_for_config.model_key # Get the default model key
    del temp_connector_for_config # Don't need this instance anymore

    server_url = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions"
    health_url = f"http://{SERVER_HOST}:{SERVER_PORT}/health"

    print(f"Target server endpoint: {server_url}")
    print("!!! IMPORTANT: Ensure the LlamaCLIConnector server is running externally: !!!")
    print(f"!!! `python llama_vision_connector.py server --port {SERVER_PORT}` !!!")

    try:
        # Optional: Check health endpoint first to see if server is reachable
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                health_response = await client.get(health_url)
                health_response.raise_for_status()
                print(f"Health check OK (Status: {health_response.status_code})")
            except (httpx.RequestError, httpx.HTTPStatusError) as health_err:
                print(f"\nError: Could not connect to or get valid health check from {health_url}")
                print(f"Details: {health_err}")
                print("Please ensure the server is running externally before running this test.")
                return

        image_path = get_test_image_path()
        if not os.path.exists(image_path):
             print(f"Error: Test image not found at {image_path}. Skipping server test.")
             return

        # Read and encode the image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            image_url_data = f"data:image/jpeg;base64,{image_data}"

        # Prepare request payload
        request_data = {
            "model": test_model_key, # Use the retrieved model key
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
            ]
        }

        print(f"\n[{time.strftime('%H:%M:%S')}] Sending request to external server via httpx...")
        start_time = time.monotonic()
        response = None
        try:
            # Use httpx.AsyncClient for the request (timeout can likely be shorter now)
            async with httpx.AsyncClient(timeout=60.0) as client: # Reduced timeout to 60s
                 response = await client.post(server_url, json=request_data)

            # Process response (moved outside the try/except for client call)
            end_time = time.monotonic()
            print(f"[{time.strftime('%H:%M:%S')}] Request finished. Duration: {end_time - start_time:.2f}s")
            print(f"Server Response Status Code: {response.status_code}")

            # Check status code *after* getting the response
            response.raise_for_status() # Raise exception for 4xx/5xx errors

            print("\nServer Response JSON:")
            print(response.json())

        except httpx.ReadTimeout as timeout_err:
             end_time = time.monotonic()
             print(f"\n!!! [{time.strftime('%H:%M:%S')}] Request TIMED OUT via httpx after {end_time - start_time:.2f}s !!!")
             print(f"Error details: {timeout_err}")
        except httpx.HTTPStatusError as status_err:
             # Handle non-2xx responses caught by raise_for_status()
             end_time = time.monotonic()
             print(f"\n!!! [{time.strftime('%H:%M:%S')}] Request FAILED (HTTP Status {status_err.response.status_code}) via httpx after {end_time - start_time:.2f}s !!!")
             print(f"Response body: {status_err.response.text}")
        except httpx.RequestError as req_err:
             # Handle other httpx request errors (connection issues etc.)
             end_time = time.monotonic()
             print(f"\n!!! [{time.strftime('%H:%M:%S')}] Request FAILED (Request Error) via httpx after {end_time - start_time:.2f}s !!!")
             print(f"Error details: {req_err}")
             import traceback
             traceback.print_exc()
        # Keep the generic Exception catch for other potential errors
        except Exception as req_err:
             end_time = time.monotonic()
             print(f"\n!!! [{time.strftime('%H:%M:%S')}] Request FAILED (Generic Error) via httpx after {end_time - start_time:.2f}s !!!")
             print(f"Error details: {req_err}")
             import traceback
             traceback.print_exc()

    except Exception as e:
        print(f"An error occurred during external server testing setup: {e}")
        import traceback
        traceback.print_exc()

    print("\nExternal server test finished.")

async def main():
    """Main test function"""
    # Test direct vision interaction (calls CLI directly)
    print("\n--- Running Direct Call Test --- ")
    await test_direct_vision()
    print("--- Direct Call Test Finished ---")

    # NOTE: No singleton reset needed as we are not managing the server process

    # Test connection to an EXTERNAL server process
    print("\n--- Running External Server Connection Test --- ")
    await test_vision_server_external()
    print("--- External Server Connection Test Finished ---")

if __name__ == "__main__":
    asyncio.run(main()) 