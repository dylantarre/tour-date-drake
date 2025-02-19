import os
import base64
import logging
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import imghdr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Better Lover")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_openai_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://github.com/dylantarre/better-lover",
            "X-Title": "Better Lover"
        },
        timeout=120.0  # Increased to 120 seconds
    )

async def process_image(client: OpenAI, image_data: str, is_url: bool = False) -> str:
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Processing image with Claude-3 (attempt {attempt + 1}/{max_retries})")
            
            # For URLs, send the URL directly. For uploaded files, use the data URL as is
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": image_data if is_url else image_data  # Use data URL directly
                }
            }
            
            response = client.chat.completions.create(
                model="anthropic/claude-3-opus",
                messages=[
                    {
                        "role": "system",
                        "content": "You are Better Lover, an expert at formatting tour dates from images. Your task is to extract tour dates from images and format them as MM/DD followed by City, ST @ Venue Name, with special characters preserved and dates separated by line breaks."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please extract and format the tour dates from this image."
                            },
                            image_content
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            logger.info(f"Raw OpenRouter response: {response}")
            
            if hasattr(response, 'error'):
                error_msg = response.error.get('message', 'Unknown error')
                error_metadata = response.error.get('metadata', {})
                raw_error = error_metadata.get('raw', '{}')
                logger.error(f"OpenRouter error: {error_msg}, Raw error: {raw_error}")
                raise HTTPException(status_code=500, detail=f"OpenRouter error: {error_msg}")
                
            if not response:
                logger.error("Empty response from OpenRouter API")
                raise HTTPException(status_code=500, detail="No response from AI service")
                
            if not hasattr(response, 'choices') or not response.choices:
                logger.error(f"No choices in response. Response type: {type(response)}")
                raise HTTPException(status_code=500, detail="Invalid response format from AI service")
                
            if not response.choices[0] or not hasattr(response.choices[0], 'message'):
                logger.error(f"Invalid choice format. First choice: {response.choices[0]}")
                raise HTTPException(status_code=500, detail="Invalid response format from AI service")
                
            content = response.choices[0].message.content
            if not content:
                logger.error("Empty content in message")
                raise HTTPException(status_code=500, detail="No content in AI service response")
                
            logger.info(f"Successfully extracted content: {content}")
            return content + "\n\nPlease double-check all info as Better Lover can make mistakes."
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds: {str(e)}")
                await asyncio.sleep(retry_delay)
                continue
            logger.error(f"All retries failed: {str(e)}", exc_info=True)
            raise

async def process_text(client: OpenAI, text: str) -> str:
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Processing text with GPT-4 (attempt {attempt + 1}/{max_retries})")
            response = client.chat.completions.create(
                model="openai/gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are Better Lover, an expert at formatting tour dates from text. Format dates as MM/DD followed by City, ST @ Venue Name, with special characters preserved and dates separated by line breaks."
                    },
                    {
                        "role": "user",
                        "content": f"Format these tour dates: {text}"
                    }
                ],
                max_tokens=1000
            )
            logger.info(f"Received response from GPT-4: {response.choices[0].message.content}")
            return response.choices[0].message.content + "\n\nPlease double-check all info as Better Lover can make mistakes."
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds: {str(e)}")
                await asyncio.sleep(retry_delay)
                continue
            logger.error(f"All retries failed: {str(e)}", exc_info=True)
            raise

class TextRequest(BaseModel):
    text: str

@app.post("/format/text")
async def format_text(request: TextRequest):
    try:
        logger.info(f"Received text request: {request.text[:100]}...")
        client = init_openai_client()
        result = await process_text(client, request.text)
        logger.info(f"Returning formatted result: {result}")
        return {"formatted_dates": result}
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/format/image")
async def format_image(file: UploadFile):
    try:
        logger.info(f"Received file: {file.filename} ({file.content_type})")
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        # Initialize OpenAI client
        client = init_openai_client()
        
        # Check if this is a URL
        is_url = file.filename.startswith(('http://', 'https://'))
        if is_url:
            logger.info("Processing as URL")
            result = await process_image(client, file.filename, is_url=True)
        else:
            # Process as uploaded file
            # Detect actual content type from file header
            actual_type = imghdr.what(None, h=contents)
            if actual_type is None:
                raise HTTPException(status_code=400, detail="Invalid image format")
            
            # Map imghdr types to MIME types
            mime_types = {
                'jpeg': 'image/jpeg',
                'jpg': 'image/jpeg',
                'png': 'image/png',
                'gif': 'image/gif',
                'webp': 'image/webp'
            }
            content_type = mime_types.get(actual_type, f'image/{actual_type}')
            logger.info(f"Detected content type: {content_type}")
            
            image_base64 = base64.b64encode(contents).decode('utf-8')
            logger.info(f"Base64 size: {len(image_base64)} chars")
            data_url = f"data:{content_type};base64,{image_base64}"
            result = await process_image(client, data_url, is_url=False)
            
        logger.info("Successfully processed image")
        return {"formatted_dates": result}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing image: {error_msg}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=error_msg) 