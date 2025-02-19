import os
import httpx
import requests
from PIL import Image
from io import BytesIO
import base64
from typing import Optional, Dict
import json

class TourDateAgent:
    def __init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Model mappings for OpenRouter
        self.model_mappings = {
            # Vision Models
            "gemini-2-flash-lite": "google/gemini-2.0-flash-lite-preview-02-05:free",
            "qwen-vl-plus": "qwen/qwen-vl-plus:free",
            "llama-3-vision": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "gemini-2-flash": "google/gemini-2.0-flash-001",
            "gemini-pro-vision": "google/gemini-pro-vision",
            "gpt-4-vision": "openai/gpt-4-vision-preview",
            
            # Free Text Models
            "gemini-2-pro": "google/gemini-2.0-pro-exp-02-05:free",
            "llama-3-11b": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "qwen-72b": "qwen/qwen2.5-vl-72b-instruct:free",
            
            # Budget Text Models
            "nova-lite": "amazon/nova-lite-v1",
            "gemini-flash": "google/gemini-flash-1.5-8b",
            "mixtral": "mistralai/mixtral-8x7b-instruct",
            
            # Standard Text Models
            "claude-3-haiku": "anthropic/claude-3-haiku",
            "nova-pro": "amazon/nova-pro-v1",
            "gemini-pro": "google/gemini-pro-1.5",
            
            # Premium Text Models
            "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
            "claude-3-opus": "anthropic/claude-3-opus-20240229",
            
            # OpenAI Models (ordered by price)
            "gpt-4o-mini": "openai/gpt-4o-mini",  # $0.15/1M input
            "gpt-4o-mini-dated": "openai/gpt-4o-mini-2024-07-18",  # $0.15/1M input
            "gpt-4o": "openai/gpt-4o",  # $2.5/1M input
            "gpt-4o-dated": "openai/gpt-4o-2024-11-20",  # $2.5/1M input
            "gpt-4o-old": "openai/gpt-4o-2024-05-13",  # $5/1M input
            "chatgpt-4o": "openai/chatgpt-4o-latest",  # $5/1M input
            "gpt-4o-extended": "openai/gpt-4o:extended",  # $6/1M input
            "gpt-4-turbo": "openai/gpt-4-turbo",  # $10/1M input
            "o1": "openai/o1",  # $15/1M input
            
            # Auto-router
            "auto": "openrouter/auto"
        }
        
        # Fallback models for each primary model
        self.fallback_models = {
            # Vision model fallbacks (prioritizing free/low-cost options)
            "google/gemini-2.0-flash-lite-preview-02-05:free": ["qwen/qwen-vl-plus:free", "meta-llama/llama-3.2-11b-vision-instruct:free"],
            "qwen/qwen-vl-plus:free": ["meta-llama/llama-3.2-11b-vision-instruct:free", "google/gemini-2.0-flash-001"],
            "meta-llama/llama-3.2-11b-vision-instruct:free": ["google/gemini-2.0-flash-lite-preview-02-05:free", "qwen/qwen-vl-plus:free"],
            "google/gemini-2.0-flash-001": ["google/gemini-pro-vision", "openai/gpt-4-vision-preview"],
            "google/gemini-pro-vision": ["google/gemini-2.0-flash-001", "openai/gpt-4-vision-preview"],
            "openai/gpt-4-vision-preview": ["google/gemini-pro-vision", "google/gemini-2.0-flash-001"],
            
            # Free text model fallbacks
            "google/gemini-2.0-pro-exp-02-05:free": ["meta-llama/llama-3.2-11b-vision-instruct:free", "qwen/qwen2.5-vl-72b-instruct:free"],
            "meta-llama/llama-3.2-11b-vision-instruct:free": ["qwen/qwen2.5-vl-72b-instruct:free", "amazon/nova-lite-v1"],
            "qwen/qwen2.5-vl-72b-instruct:free": ["google/gemini-2.0-pro-exp-02-05:free", "amazon/nova-lite-v1"],
            
            # Budget text model fallbacks
            "amazon/nova-lite-v1": ["google/gemini-flash-1.5-8b", "mistralai/mixtral-8x7b-instruct"],
            "google/gemini-flash-1.5-8b": ["mistralai/mixtral-8x7b-instruct", "amazon/nova-lite-v1"],
            "mistralai/mixtral-8x7b-instruct": ["amazon/nova-lite-v1", "google/gemini-flash-1.5-8b"],
            
            # Standard text model fallbacks
            "anthropic/claude-3-haiku": ["amazon/nova-pro-v1", "google/gemini-pro-1.5"],
            "amazon/nova-pro-v1": ["google/gemini-pro-1.5", "anthropic/claude-3-haiku"],
            "google/gemini-pro-1.5": ["anthropic/claude-3-haiku", "amazon/nova-pro-v1"],
            
            # Premium text model fallbacks
            "anthropic/claude-3-sonnet-20240229": ["openai/gpt-4o", "anthropic/claude-3-opus-20240229"],
            "anthropic/claude-3-opus-20240229": ["openai/gpt-4o", "anthropic/claude-3-sonnet-20240229"],
            
            # OpenAI model fallbacks (ordered by price)
            "openai/gpt-4o-mini": ["openai/gpt-4o-mini-2024-07-18", "openai/gpt-4o"],
            "openai/gpt-4o-mini-2024-07-18": ["openai/gpt-4o-mini", "openai/gpt-4o"],
            "openai/gpt-4o": ["openai/gpt-4o-2024-11-20", "openai/chatgpt-4o-latest"],
            "openai/gpt-4o-2024-11-20": ["openai/gpt-4o", "openai/gpt-4o-2024-05-13"],
            "openai/gpt-4o-2024-05-13": ["openai/chatgpt-4o-latest", "openai/gpt-4o"],
            "openai/chatgpt-4o-latest": ["openai/gpt-4o", "openai/gpt-4o-extended"],
            "openai/gpt-4o:extended": ["openai/gpt-4-turbo", "openai/chatgpt-4o-latest"],
            "openai/gpt-4-turbo": ["openai/gpt-4o:extended", "openai/o1"],
            "openai/o1": ["openai/gpt-4-turbo", "openai/gpt-4o:extended"]
        }
        
        # Vision prompt for image analysis
        self.vision_prompt = """I'm Better Lover, and I'll help format these tour dates clearly and accurately.

Let me organize this information:

## Tour/Festival
{Main event name if present}

## Artists
{List of performing artists}

## Tour Dates
{Formatted as follows}
- MM/DD City, ST @ Venue Name [special_char]

## Notes
{Explanations for any special marks or symbols}

Guidelines:
- Only include dates that are clearly legible
- Mark unclear information with [?]
- Maintain proper markdown formatting
- Explain any special symbols in the Notes section
- Omit venue if not specified

Example format:
## Tour/Festival
Summer Tour 2024

## Artists
- Main Artist
- Supporting Act
- Special Guest

## Tour Dates
- 01/23 Los Angeles, CA @ The Wiltern
- 01/24 San Francisco, CA @ The Warfield *

## Notes
* with special guest performance

If any information is unclear, I'll either mark it with [?] or omit it entirely."""

        # Text prompt for text analysis
        self.text_prompt = """I'm Better Lover, and I'll help format these tour dates consistently.

Let me organize this as follows:

## Tour Dates
{Formatted with bullet points}
- MM/DD City, ST @ Venue Name [special_char]

## Notes
{Explanations for special marks}

Guidelines:
- Use bullet points for each date
- Maintain proper markdown formatting
- Include venue names after @ symbol
- Preserve special marks (*, %, #)
- Mark unclear info with [?]
- Omit venue when not specified

Example format:
## Tour Dates
- 01/23 Los Angeles, CA @ The Wiltern
- 01/24 San Francisco, CA @ The Warfield *

## Notes
* with special guest performance"""

    def _encode_image(self, image_url: str) -> str:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    
    async def process(
        self,
        image_url: Optional[str],
        text_input: Optional[str],
        output_format: str,
        model: str
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/dylan-tarre",  # For OpenRouter rankings
            "X-Title": "Better Lover",  # For OpenRouter rankings
            "Content-Type": "application/json",
            "Accept": "text/event-stream"  # Required for streaming
        }

        # Map the model ID to OpenRouter format and get fallbacks
        openrouter_model = self.model_mappings.get(model, model)
        fallback_models = self.fallback_models.get(openrouter_model, [])

        # Select appropriate prompt based on input type
        base_prompt = self.vision_prompt if image_url else self.text_prompt
        
        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "system", "content": "Remember: Never guess or hallucinate information. If something is unclear, mark it with [?] or omit it."}
        ]
        
        if image_url:
            # For image URLs, we need to handle both direct URLs and base64 images
            if image_url.startswith('data:image'):
                image_data = image_url  # Already base64 encoded
            else:
                image_data = self._encode_image(image_url)
            
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please extract and format the tour dates from this image. If any information is unclear, mark it with [?] or omit it entirely."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data,
                            "detail": "auto"  # Let the model decide the detail level
                        }
                    }
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Please format these tour dates according to the specified format:\n\n{text_input}"
            })

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    self.api_url,
                    headers=headers,
                    json={
                        "model": openrouter_model,
                        "messages": messages,
                        "max_tokens": 1500,
                        "temperature": 0.1,
                        "models": fallback_models if fallback_models else None,
                        "route": "fallback" if not fallback_models else None,
                        "stream": True  # Enable streaming
                    }
                ) as response:
                    print(f"Response status: {response.status_code}")  # Debug log
                    print(f"Response headers: {response.headers}")  # Debug log
                    
                    if response.status_code != 200:
                        try:
                            error_text = ''
                            async for chunk in response.aiter_bytes():
                                error_text += chunk.decode()
                            print(f"Error response: {error_text}")  # Debug log
                            error_data = json.loads(error_text)
                            error_message = error_data.get('error', {}).get('message', 'Unknown error')
                            if error_data.get('error', {}).get('metadata'):
                                error_message += f" ({error_data['error']['metadata']})"
                        except Exception as e:
                            print(f"Error parsing error response: {e}")  # Debug log
                            error_message = f"HTTP {response.status_code}: {response.reason_phrase}"
                        raise Exception(f"OpenRouter API error: {error_message}")
                    
                    # Stream the response
                    markdown_started = False
                    async for line in response.aiter_lines():
                        print(f"Received line: {line}")  # Debug log
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])  # Remove "data: " prefix
                                print(f"Parsed data: {data}")  # Debug log
                                if data.get("choices") and len(data["choices"]) > 0:
                                    chunk = data["choices"][0]
                                    if chunk.get("finish_reason") == "stop":
                                        yield "\n\n---\n*Note: Please verify all information as Better Lover may make mistakes.*"
                                        break
                                    if chunk.get("delta", {}).get("content"):
                                        content = chunk["delta"]["content"]
                                        print(f"Yielding content: {content}")  # Debug log
                                        if not markdown_started and content.strip():
                                            markdown_started = True
                                        yield content
                            except json.JSONDecodeError as e:
                                print(f"JSON decode error: {e}")  # Debug log
                                continue
                            except Exception as e:
                                print(f"Unexpected error processing line: {e}")  # Debug log
                                continue
                
        except httpx.TimeoutException:
            raise Exception("Request timed out. Please try again.")
        except Exception as e:
            raise Exception(f"Error processing request: {str(e)}") 