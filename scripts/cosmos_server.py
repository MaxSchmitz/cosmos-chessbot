#!/usr/bin/env python3
"""Remote Cosmos inference server.

Run this on your GPU server (H100/GB200) to serve Cosmos-Reason2 inference.
The local orchestrator will connect to this server via HTTP.
"""

import argparse
import base64
import io
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cosmos_chessbot.vision import CosmosPerception


class InferenceRequest(BaseModel):
    """Request for Cosmos inference."""
    image_base64: str
    max_new_tokens: int = 2048
    temperature: float = 0.1


class InferenceResponse(BaseModel):
    """Response from Cosmos inference."""
    fen: str
    confidence: float
    anomalies: list[str]
    raw_response: str


app = FastAPI(title="Cosmos Inference Server")
perception = None


@app.on_event("startup")
async def load_model():
    """Load Cosmos model on startup."""
    global perception
    print("Loading Cosmos-Reason2 model...")
    perception = CosmosPerception()
    print("Model loaded and ready!")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": perception is not None}


@app.post("/perceive", response_model=InferenceResponse)
async def perceive(request: InferenceRequest):
    """Run perception on an image."""
    if perception is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))

        # Run inference
        board_state = perception.perceive(
            image,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )

        return InferenceResponse(
            fen=board_state.fen,
            confidence=board_state.confidence,
            anomalies=board_state.anomalies,
            raw_response=board_state.raw_response,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Cosmos inference server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    args = parser.parse_args()

    print(f"Starting Cosmos inference server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
