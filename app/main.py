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

app = FastAPI(title="Tour Date Drake")

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
            logger.info(f"Processing image with model 'google/gemini-2.0-pro-exp-02-05:free' (attempt {attempt + 1}/{max_retries})")
            
            # For URLs, send the URL directly. For uploaded files, use the data URL as is
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": image_data if is_url else image_data  # Use data URL directly
                }
            }
            
            response = client.chat.completions.create(
                model="google/gemini-2.0-pro-exp-02-05:free",
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
- If there are notes at the bottom of the list (like "* Supporting Band"), include them at the end of the output

EXAMPLE OUTPUTS:

US Shows:
06/15 Brooklyn, NY @ Saint Vitus
06/16 Philadelphia, PA @ First Unitarian Church
06/17 Boston, MA @ The Middle East

European Shows (still using MM/DD format):
06/20 Hamburg, DE @ Viper Room
06/21 Berlin, DE @ SO36
06/22 Wrocław, PL @ Klub Pogłos

Mixed Tour with Notes:
03/14 Austin, TX @ Central Presbyterian Church (SXSW)
05/31 Raleigh, NC @ The Ritz *
06/02 Cleveland, OH @ Globe Iron *
06/03 Toronto, ON @ HISTORY *
06/15 Brooklyn, NY @ Saint Vitus
06/20 Hamburg, DE @ Viper Room

* Supporting Panchiko

Festival Appearances:
06/13 London, UK @ Lido Festival
06/14 Manchester, UK @ Outbreak Festival
06/19 Lisbon, PT @ Kalorama Festival
06/20 Madrid, ES @ Kalorama Festival

Common Mistakes to Avoid:
- Using dashes in dates: Use "06/15" not "06/15 -"
- Using European date format: Use "06/15" (June 15) not "15/06"
- Incorrect city/ST order: Use "Leeuwarden, NL" not "NL, Leeuwarden"
- Removing important context: Keep venue details like "(SXSW)" and supporting act info

Note: Always verify all dates and venue information as accuracy is crucial."""
                            },
                            image_content
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            logger.info(f"Received response from model 'google/gemini-2.0-pro-exp-02-05:free': {response}")
            
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
            return content
            
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
            logger.info(f"Processing text with google/gemini-2.0-pro-exp-02-05:free(attempt {attempt + 1}/{max_retries})")
            response = client.chat.completions.create(
                model="google/gemini-2.0-pro-exp-02-05:free",
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
- **IMPORTANT: Preserve all informational notes, supporting act info, and venue details like "(SXSW)" or "* Supporting Band"**
- If there are notes at the bottom of the list (like "* Supporting Band"), include them at the end of the output

EXAMPLE OUTPUTS:

US Shows:
06/15 Brooklyn, NY @ Saint Vitus
06/16 Philadelphia, PA @ First Unitarian Church
06/17 Boston, MA @ The Middle East

European Shows (still using MM/DD format):
06/20 Hamburg, DE @ Viper Room
06/21 Berlin, DE @ SO36
06/22 Wrocław, PL @ Klub Pogłos

Mixed Tour with Notes:
03/14 Austin, TX @ Central Presbyterian Church (SXSW)
05/31 Raleigh, NC @ The Ritz *
06/02 Cleveland, OH @ Globe Iron *
06/03 Toronto, ON @ HISTORY *
06/15 Brooklyn, NY @ Saint Vitus
06/20 Hamburg, DE @ Viper Room

* Supporting Panchiko

Festival Appearances:
06/13 London, UK @ Lido Festival
06/14 Manchester, UK @ Outbreak Festival
06/19 Lisbon, PT @ Kalorama Festival
06/20 Madrid, ES @ Kalorama Festival

Common Mistakes to Avoid:
- Using dashes in dates: Use "06/15" not "06/15 -"
- Using European date format: Use "06/15" (June 15) not "15/06"
- Incorrect city/ST order: Use "Leeuwarden, NL" not "NL, Leeuwarden"
- Removing important context: Keep venue details like "(SXSW)" and supporting act info

Note: Always verify all dates and venue information as accuracy is crucial.

Here are the dates to format: {text}"""
                    }
                ],
                max_tokens=2000
            )
            
            logger.info(f"Raw response from model: {response}")
            
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
                
            logger.info(f"Received response from model 'google/gemini-2.0-pro-exp-02-05:free': {content}")
            return content
            
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