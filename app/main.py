import os
import base64
import logging
import asyncio
import re
from fastapi import FastAPI, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import imghdr
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Tour Date Drake")

# Add rate limit exceeded handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configure request size limits (20 GB)
MAX_REQUEST_SIZE = 20 * 1024 * 1024 * 1024  # 20 GB in bytes
LARGE_FILE_THRESHOLD = 1 * 1024 * 1024 * 1024  # 1 GB in bytes

def init_openai_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://github.com/dylantarre/tour-date-drake",
            "X-Title": "Tour Date Drake"
        },
        timeout=120.0  # Increased to 120 seconds
    )

async def process_image(client: OpenAI, image_data: str, is_url: bool = False) -> str:
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Processing image with model 'google/gemini-2.0-flash-001' (attempt {attempt + 1}/{max_retries})")
            
            # For URLs, send the URL directly. For uploaded files, use the data URL as is
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": image_data if is_url else image_data  # Use data URL directly
                }
            }
            
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-001", # Stable model
                messages=[
                    {
                        "role": "system",
                        "content": "You are Tour Date Drake, a helpful assistant that formats tour dates."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Please extract and format all tour dates from this image using these rules:

FORMATTING RULES:
- **ALWAYS use American date format MM/DD for ALL dates (even for European/international tours)**
- **IMPORTANT: Convert any European format dates (DD/MM) to American format (MM/DD) in the output**
- **For example: "07/03" in European format should be converted to "03/07" in American format (March 7th)**
- **Date format must be MM/DD without ANY dashes or hyphens (e.g., "06/15 City, ST" not "06/15 - City, ST")**
- **ABSOLUTELY NO DASHES in the final output**
- **Always format as: MM/DD City, ST @ Venue (if venue is known)**
- City and Venue names should have normal capitalization (e.g., "Hamburg" not "HAMBURG")
- If venues are listed separately, match them to the correct dates. If a venue cannot be matched, omit it.
- For US states, ST is the state code (e.g., NY, CA)
- For countries, ST is the country code (e.g., DE, UK)
- All state/country codes must be two letters and in caps
- **City comes first, then ST (e.g., "Leeuwarden, NL" not "NL, Leeuwarden")**
- Separate each date with a line break
- Preserve any special characters in city names
- Remove any dashes, commas, or extra formatting from the original text
- For long venue names, keep them concise if possible
- **IMPORTANT: Preserve all informational notes, supporting act info, and venue details like "(NOTE)" or "* Supporting Band"**
- **If a date, city, or venue has a special character, symbol, or note (such as *, %, †, etc.), make sure to keep that character at the end of that specific line, after the venue, not just at the end of the list. For example: "06/15 City, ST @ Venue *"**
- If there are notes at the bottom of the list (like "* Supporting Band"), include them at the end of the output"""
                            },
                            image_content
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            logger.info(f"Received response from model 'google/gemini-2.0-flash-001': {response}")
            
            # Enhanced error logging
            try:
                # Log detailed response structure for debugging
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response attributes: {dir(response)}")
                
                # Check if response is valid
                if not hasattr(response, 'choices') or not response.choices:
                    logger.error(f"No choices in response. Response type: {type(response)}")
                    raise ValueError("Invalid response format from AI service")
                    
                if not response.choices[0] or not hasattr(response.choices[0], 'message'):
                    logger.error(f"Invalid choice format. First choice: {response.choices[0]}")
                    raise ValueError("Invalid response format from AI service")
                    
                content = response.choices[0].message.content
                if not content:
                    logger.error("Empty content in message")
                    raise ValueError("No content in AI service response")
                
                logger.info(f"Received response from model 'google/gemini-2.0-flash-001': {content}")
                
                return content
            except AttributeError as ae:
                logger.error(f"AttributeError parsing response: {ae}")
                # Try alternative response format parsing if available
                if hasattr(response, 'model_dump'):
                    response_dict = response.model_dump()
                    logger.debug(f"Response as dict: {response_dict}")
                    if 'choices' in response_dict and response_dict['choices']:
                        if 'message' in response_dict['choices'][0]:
                            content = response_dict['choices'][0]['message'].get('content')
                            if content:
                                return content
                raise ValueError(f"Failed to parse response format: {ae}")
            
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
            logger.info(f"Processing text with google/gemini-2.0-flash-001 (attempt {attempt + 1}/{max_retries})")
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-001", # Stable model
                messages=[
                    {
                        "role": "system",
                        "content": "You are Tour Date Drake, a helpful assistant that formats tour dates."
                    },
                    {
                        "role": "user",
                        "content": f"""Please extract and format all tour dates from this text using these rules:

FORMATTING RULES:
- **ALWAYS use American date format MM/DD for ALL dates (even for European/international tours)**
- **IMPORTANT: Convert any European format dates (DD/MM) to American format (MM/DD) in the output**
- **For example: "07/03" in European format should be converted to "03/07" in American format (March 7th)**
- **Date format must be MM/DD without ANY dashes or hyphens (e.g., "06/15 City, ST" not "06/15 - City, ST")**
- **ABSOLUTELY NO DASHES in the final output**
- **Always format as: MM/DD City, ST @ Venue (if venue is known)**
- City and Venue names should have normal capitalization (e.g., "Hamburg" not "HAMBURG")
- If venues are listed separately, match them to the correct dates. If a venue cannot be matched, omit it.
- For US states, ST is the state code (e.g., NY, CA)
- For countries, ST is the country code (e.g., DE, UK)
- All state/country codes must be two letters and in caps
- **City comes first, then ST (e.g., "Leeuwarden, NL" not "NL, Leeuwarden")**
- Separate each date with a line break
- Preserve any special characters in city names
- Remove any dashes, commas, or extra formatting from the original text
- For long venue names, keep them concise if possible
- **IMPORTANT: Preserve all informational notes, supporting act info, and venue details like "(NOTE)" or "* Supporting Band"**
- **If a date, city, or venue has a special character, symbol, or note (such as *, %, †, etc.), make sure to keep that character at the end of that specific line, after the venue, not just at the end of the list. For example: "06/15 City, ST @ Venue *"**
- If there are notes at the bottom of the list (like "* Supporting Band"), include them at the end of the output

Here are the dates to format: {text}"""
                    }
                ],
                max_tokens=2000
            )
            
            logger.info(f"Raw response from model: {response}")
            
            # Enhanced error logging
            try:
                # Log detailed response structure for debugging
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response attributes: {dir(response)}")
                
                # Check if response is valid
                if not hasattr(response, 'choices') or not response.choices:
                    logger.error(f"No choices in response. Response type: {type(response)}")
                    raise ValueError("Invalid response format from AI service")
                    
                if not response.choices[0] or not hasattr(response.choices[0], 'message'):
                    logger.error(f"Invalid choice format. First choice: {response.choices[0]}")
                    raise ValueError("Invalid response format from AI service")
                    
                content = response.choices[0].message.content
                if not content:
                    logger.error("Empty content in message")
                    raise ValueError("No content in AI service response")
                
                logger.info(f"Received response from model 'google/gemini-2.0-flash-001': {content}")
                
                return content
            except AttributeError as ae:
                logger.error(f"AttributeError parsing response: {ae}")
                # Try alternative response format parsing if available
                if hasattr(response, 'model_dump'):
                    response_dict = response.model_dump()
                    logger.debug(f"Response as dict: {response_dict}")
                    if 'choices' in response_dict and response_dict['choices']:
                        if 'message' in response_dict['choices'][0]:
                            content = response_dict['choices'][0]['message'].get('content')
                            if content:
                                return content
                raise ValueError(f"Failed to parse response format: {ae}")
            
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
@limiter.limit("20/minute")  # Rate limit: 20 requests per minute
async def format_text(request: Request, text_request: TextRequest):
    try:
        logger.info(f"Received text request: {text_request.text[:100]}...")
        client = init_openai_client()
        result = await process_text(client, text_request.text)
        logger.info(f"Returning formatted result: {result}")
        return {"formatted_dates": result}
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/format/image")
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute
async def format_image(request: Request, file: UploadFile):
    try:
        logger.info(f"Received file: {file.filename} ({file.content_type})")
        
        # Check file size before processing
        contents = await file.read(MAX_REQUEST_SIZE + 1)  # Read one extra byte to check if file is too large
        file_size = len(contents)
        
        if file_size > MAX_REQUEST_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum limit of {MAX_REQUEST_SIZE / (1024 * 1024 * 1024):.1f}GB"
            )
            
        if file_size > LARGE_FILE_THRESHOLD:
            logger.warning(f"Processing large file ({file_size / (1024 * 1024 * 1024):.1f}GB). This may impact server performance.")
            
        logger.info(f"File size: {file_size / (1024 * 1024):.1f}MB")
        
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
            
            try:
                image_base64 = base64.b64encode(contents).decode('utf-8')
                logger.info(f"Base64 size: {len(image_base64) / (1024 * 1024):.1f}MB")
                data_url = f"data:{content_type};base64,{image_base64}"
                result = await process_image(client, data_url, is_url=False)
            finally:
                # Explicitly delete large objects to help garbage collection
                del contents
                del image_base64
                del data_url
            
        logger.info("Successfully processed image")
        return {"formatted_dates": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing image: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

# Add a startup event to log the rate limits
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Tour Date Drake API with the following limits:")
    logger.info("- /format/text: 20 requests per minute")
    logger.info("- /format/image: 10 requests per minute")
    logger.info(f"- Maximum request size: {MAX_REQUEST_SIZE / (1024 * 1024 * 1024):.1f}GB")
    logger.info(f"- Large file warning threshold: {LARGE_FILE_THRESHOLD / (1024 * 1024 * 1024):.1f}GB") 