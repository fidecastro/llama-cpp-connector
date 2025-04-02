#!/usr/bin/env python3
import asyncio
import os
import sys
import openai  # Need openai client

# Add the root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correct import path
from llama_server_connector import LlamaServerConnector

async def main():
    # Initialize the connector - this also starts the server
    # The server will be automatically killed on script exit
    connector = LlamaServerConnector(
        config_path="config/models.json",  # Path relative to root
        model_key=None,  # Will use first non-vision model
        initial_port=8080,
        host="127.0.0.1",
        debug_server=True
    )

    # Option 1: Use the connector's simplified get_response method
    print("\n--- Testing with connector.get_response ---")
    try:
        prompt = "What is the capital of France?"
        print(f"Prompt: {prompt}")
        response_text = connector.get_response(prompt=prompt)
        print(f"Response: {response_text}")

        prompt = "Tell me a short joke."
        print(f"Prompt: {prompt}")
        response_text = connector.get_response(prompt=prompt)
        print(f"Response: {response_text}")

    except Exception as e:
        print(f"Error using get_response: {e}")

    # Option 2: Use the OpenAI client directly with the server URL
    print("\n--- Testing with OpenAI client ---")
    try:
        server_url = connector.get_server_url()
        print(f"Server URL: {server_url}")
        client = openai.AsyncOpenAI(base_url=server_url, api_key="dummy_key") # Use Async client

        print("\nTesting chat completion...")
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a short joke."}
            ],
            model=connector.model_key, # Use the model key from the connector
            max_tokens=100,
            temperature=0.7
        )
        print(f"Chat Response: {chat_completion.choices[0].message.content}")

        print("\nTesting streaming completion...")
        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Count from 1 to 5:"}],
            model=connector.model_key,
            stream=True,
        )
        print("Streaming response:")
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")

    except Exception as e:
        print(f"Error using OpenAI client: {e}")

    # No need to explicitly call connector.close() or kill_server()
    # atexit handles cleanup.
    print("\nScript finished. Server should shut down automatically.")


if __name__ == "__main__":
    # Note: get_response is now async, so we need the async main function
    asyncio.run(main()) 