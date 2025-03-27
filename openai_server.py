from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import tempfile
import os
from llama_vision_connector import LlamaVisionConnector
import json
from datetime import datetime

app = FastAPI(title="Llama Vision OpenAI API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LlamaVisionConnector with default model
connector = LlamaVisionConnector()

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

def get_connector_for_model(model_name: str) -> LlamaVisionConnector:
    """
    Get a LlamaVisionConnector instance for the specified model.
    Falls back to default model if specified model is not found.
    """
    try:
        return LlamaVisionConnector(model_key=model_name)
    except ValueError:
        # If model not found, return default connector
        return connector

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion with vision capabilities.
    The last message should contain the image in base64 format.
    """
    try:
        # Get the last message which should contain the image
        last_message = request.messages[-1]
        
        # Extract image from content
        if not last_message.content.startswith("data:image"):
            raise HTTPException(status_code=400, detail="Last message must contain an image in base64 format")
            
        # Extract base64 image data
        image_data = last_message.content.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        
        # Create a temporary file for the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name
            
        try:
            # Get the prompt from previous messages
            prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages[:-1]])
            
            # Get connector for the specified model
            model_connector = get_connector_for_model(request.model)
            
            # Get response from the vision model
            response = await model_connector.get_response(temp_file_path, prompt)
            
            if not response:
                raise HTTPException(status_code=500, detail="Failed to get response from vision model")
                
            # Create OpenAI-compatible response
            completion_response = {
                "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # We don't have token counting yet
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            return completion_response
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """
    List available models.
    """
    # Load the configuration to get available models
    with open("config/models.json", "r") as f:
        config = json.load(f)
    
    # Get all vision models from the configuration
    vision_models = [
        {
            "id": model_key,
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": "llama-cpp-connector"
        }
        for model_key in config.get("MODELS", {}).keys()
    ]
    
    return {
        "data": vision_models,
        "object": "list"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 