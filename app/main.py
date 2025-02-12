from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from pathlib import Path
from .tour_agent import TourDateAgent
import json

load_dotenv()

app = FastAPI(title="Tour Date Parser")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory of the current file
BASE_DIR = Path(__file__).resolve().parent

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

class TourRequest(BaseModel):
    image_url: Optional[str] = None
    text_input: Optional[str] = None
    output_format: str = "lambgoat"  # default format
    model: str = "auto"  # default to auto-router

@app.get("/")
async def read_root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))

async def stream_response(agent, request):
    try:
        print(f"Starting stream response for model: {request.model}")  # Debug log
        async for chunk in agent.process(
            image_url=request.image_url,
            text_input=request.text_input,
            output_format=request.output_format,
            model=request.model
        ):
            if chunk:  # Only yield non-empty chunks
                try:
                    # Try to parse as JSON to validate it's proper content
                    json_chunk = {"choices": [{"delta": {"content": chunk}}]}
                    json_str = json.dumps(json_chunk)
                    print(f"Yielding chunk: {json_str}")  # Debug log
                    yield f"data: {json_str}\n\n"
                except Exception as e:
                    print(f"Error formatting chunk: {e}")  # Debug log
                    continue
    except Exception as e:
        error_msg = str(e)
        print(f"Stream error: {error_msg}")  # Debug log
        
        if "No endpoints found" in error_msg:
            error = "Selected model is not available. Please try a different model."
        elif "timeout" in error_msg.lower():
            error = "Request timed out. Please try again."
        elif "API key" in error_msg:
            error = "Invalid API key. Please check your configuration."
        else:
            error = f"Error: {error_msg}"
        
        error_json = json.dumps({"error": {"message": error}})
        yield f"data: {error_json}\n\n"

@app.post("/parse_tour_dates")
async def parse_tour_dates(request: TourRequest):
    if not request.image_url and not request.text_input:
        raise HTTPException(status_code=400, detail="Either image_url or text_input must be provided")
    
    try:
        agent = TourDateAgent()
        return StreamingResponse(
            stream_response(agent, request),
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"Endpoint error: {e}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e)) 