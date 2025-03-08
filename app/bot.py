import os
import discord
from discord import app_commands
import aiohttp
import logging
from dotenv import load_dotenv
import asyncio
from io import BytesIO
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("API_URL", "http://api:4343")
MAX_DISCORD_LENGTH = 2000
CODE_BLOCK_OVERHEAD = 7  # Length of "```\n" + "\n```"
CONTINUED_OVERHEAD = 13  # Length of "(continued...)\n"
DISCLAIMER_LENGTH = 65  # Length of the disclaimer message
HEADER_LENGTH = 20  # Conservative estimate for headers and formatting
EFFECTIVE_LENGTH = MAX_DISCORD_LENGTH - CODE_BLOCK_OVERHEAD - CONTINUED_OVERHEAD - HEADER_LENGTH  # Maximum safe length

def split_message(message: str) -> list[str]:
    """Split a message into chunks that fit within Discord's character limit."""
    if not message:
        return []
        
    # If the message is short enough, return it as is
    if len(message) <= EFFECTIVE_LENGTH:
        return [message]
        
    chunks = []
    current_chunk = ""
    
    # Split into lines first
    lines = message.split('\n')
    
    for line in lines:
        # If a single line is too long, split it by words
        if len(line) > EFFECTIVE_LENGTH:
            words = line.split(' ')
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= EFFECTIVE_LENGTH:
                    current_line = current_line + ' ' + word if current_line else word
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word
                    current_line = word
            
            if current_line:
                if len(current_chunk) + len(current_line) + 1 > EFFECTIVE_LENGTH:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = current_line
                else:
                    current_chunk = current_chunk + '\n' + current_line if current_chunk else current_line
        else:
            # Handle normal lines
            if len(current_chunk) + len(line) + 1 > EFFECTIVE_LENGTH:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk = current_chunk + '\n' + line if current_chunk else line
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

class TourDateDrake(discord.Client):
    def __init__(self):
        # We need message content intent to read messages
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()

client = TourDateDrake()

@client.tree.command(name="dates", description="Format tour dates from text")
@app_commands.describe(
    text="Tour dates text to format"
)
async def dates(interaction: discord.Interaction, text: str):
    try:
        # Input validation
        if len(text) > 4000:  # Conservative limit
            await interaction.response.send_message("Error: Input text is too long. Please keep it under 4000 characters.")
            return
            
        if not text.strip():
            await interaction.response.send_message("Error: Please provide some text containing tour dates.")
            return

        # First respond that we're working on it
        try:
            await interaction.response.defer(thinking=True, ephemeral=False)
        except discord.errors.NotFound:
            logger.error("Interaction not found - it may have timed out")
            return
        except discord.errors.HTTPException as e:
            if e.code == 40060:  # Interaction already acknowledged
                logger.warning("Interaction was already acknowledged")
            else:
                raise
        
        async with aiohttp.ClientSession() as session:
            # Process as regular text
            logger.info(f"Processing text: {text[:100]}...")
            async with session.post(
                f"{API_URL}/format/text",
                json={"text": text},
                timeout=aiohttp.ClientTimeout(total=180)  # 3 minute timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    try:
                        error_json = await response.json()
                        error_detail = error_json.get('detail', 'Unknown error')
                    except:
                        error_detail = error_text
                    logger.error(f"API error response: {error_text}")
                    try:
                        await interaction.followup.send(f"Error: {error_detail}")
                    except discord.NotFound:
                        logger.error("Interaction expired during error handling")
                    return
                    
                result = await response.json()

            formatted_dates = result.get("formatted_dates", "Error: No dates found")
            band_names = result.get("band_names", "")
            
            if not formatted_dates or formatted_dates == "Error: No dates found":
                await interaction.followup.send("I couldn't find any valid tour dates in your text. Please make sure you've provided text containing tour dates.")
                return
                
            logger.info(f"Sending formatted response to Discord: {formatted_dates[:100]}...")
            
            try:
                # Split and send original text in chunks
                original_chunks = split_message(text)
                for i, chunk in enumerate(original_chunks):
                    # Calculate total message length including formatting
                    if i == 0:
                        # First chunk includes the header
                        message = f"**Original Text:**\n```\n{chunk}\n```"
                    else:
                        message = f"```\n(continued...)\n{chunk}\n```"
                    
                    # Double check length before sending
                    if len(message) > MAX_DISCORD_LENGTH:
                        logger.warning(f"Message too long ({len(message)} chars), splitting further")
                        sub_chunks = split_message(chunk)
                        for j, sub_chunk in enumerate(sub_chunks):
                            if j == 0 and i == 0:
                                # First sub-chunk of first chunk includes header
                                sub_message = f"**Original Text:**\n```\n{sub_chunk}\n```"
                            else:
                                sub_message = f"```\n(continued...)\n{sub_chunk}\n```"
                            await interaction.followup.send(sub_message)
                    else:
                        await interaction.followup.send(message)
                
                # If band names were extracted, display them
                if band_names:
                    band_chunks = split_message(band_names)
                    for i, chunk in enumerate(band_chunks):
                        if i == 0:
                            message = f"**Band Names:**\n```\n{chunk}\n```"
                        else:
                            message = f"```\n(continued...)\n{chunk}\n```"
                        
                        # Double check length before sending
                        if len(message) > MAX_DISCORD_LENGTH:
                            logger.warning(f"Band names message too long ({len(message)} chars), splitting further")
                            sub_chunks = split_message(chunk)
                            for j, sub_chunk in enumerate(sub_chunks):
                                if j == 0 and i == 0:
                                    sub_message = f"**Band Names:**\n```\n{sub_chunk}\n```"
                                else:
                                    sub_message = f"```\n(continued...)\n{sub_chunk}\n```"
                                await interaction.followup.send(sub_message)
                        else:
                            await interaction.followup.send(message)
                
                # Split and send formatted dates in chunks
                formatted_chunks = split_message(formatted_dates)
                for i, chunk in enumerate(formatted_chunks):
                    # Calculate total message length including formatting
                    if i == 0:
                        # First chunk includes the header
                        message = f"**Formatted Dates:**\n```\n{chunk}\n```"
                    else:
                        message = f"```\n(continued...)\n{chunk}\n```"
                    
                    # Add disclaimer to the last chunk
                    if i == len(formatted_chunks) - 1:
                        disclaimer = "\n\n*Note: Tour Date Drake is AI-powered and may make mistakes. Please verify important information.*"
                        if len(message) + len(disclaimer) <= MAX_DISCORD_LENGTH:
                            message += disclaimer
                        else:
                            # If adding disclaimer would exceed limit, send it separately
                            await interaction.followup.send(message)
                            await interaction.followup.send(disclaimer)
                            continue
                        
                    # Double check length before sending
                    if len(message) > MAX_DISCORD_LENGTH:
                        logger.warning(f"Message too long ({len(message)} chars), splitting further")
                        sub_chunks = split_message(chunk)
                        for j, sub_chunk in enumerate(sub_chunks):
                            if j == 0 and i == 0:
                                # First sub-chunk of first chunk includes header
                                sub_message = f"**Formatted Dates:**\n```\n{sub_chunk}\n```"
                            else:
                                sub_message = f"```\n(continued...)\n{sub_chunk}\n```"
                            await interaction.followup.send(sub_message)
                    else:
                        await interaction.followup.send(message)
                    
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                await interaction.followup.send(f"Error: {str(e)}")
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        try:
            await interaction.followup.send("Error: Request timed out. Please try again.")
        except discord.NotFound:
            logger.error("Interaction expired during timeout")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        try:
            await interaction.followup.send(f"Error: {str(e)}")
        except discord.NotFound:
            logger.error("Interaction expired during error handling")

@client.tree.command(name="image", description="Extract tour dates from an image")
async def image(interaction: discord.Interaction, image: discord.Attachment):
    try:
        # Input validation
        if not image:
            await interaction.response.send_message("Error: Please provide an image containing tour dates.")
            return
            
        if not image.content_type or not image.content_type.startswith('image/'):
            await interaction.response.send_message("Error: The file you uploaded is not a valid image. Please upload an image file (JPEG, PNG, etc).")
            return
            
        if image.size > 10 * 1024 * 1024:  # 10 MB limit
            await interaction.response.send_message("Error: Image is too large. Please upload an image smaller than 10 MB.")
            return

        # First respond that we're working on it
        try:
            await interaction.response.defer(thinking=True, ephemeral=False)
        except discord.errors.NotFound:
            logger.error("Interaction not found - it may have timed out")
            return
        except discord.errors.HTTPException as e:
            if e.code == 40060:  # Interaction already acknowledged
                logger.warning("Interaction was already acknowledged")
            else:
                raise
        
        # Download the image
        image_bytes = await image.read()
        
        async with aiohttp.ClientSession() as session:
            # Create a FormData object
            data = aiohttp.FormData()
            data.add_field('file', 
                          image_bytes,
                          filename=image.filename,
                          content_type=image.content_type)
            
            # Send the image to our API
            logger.info(f"Sending image to API: {image.filename} ({image.size} bytes)")
            async with session.post(
                f"{API_URL}/format/image",
                data=data,
                timeout=aiohttp.ClientTimeout(total=180)  # 3 minute timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    try:
                        error_json = await response.json()
                        error_detail = error_json.get('detail', 'Unknown error')
                    except:
                        error_detail = error_text
                    logger.error(f"API error response: {error_text}")
                    try:
                        await interaction.followup.send(f"Error: {error_detail}")
                    except discord.NotFound:
                        logger.error("Interaction expired during error handling")
                    return
                    
                result = await response.json()
            
            formatted_dates = result.get("formatted_dates", "Error: No dates found")
            band_names = result.get("band_names", "")
            
            if not formatted_dates or formatted_dates == "Error: No dates found":
                await interaction.followup.send("I couldn't find any valid tour dates in the image. Please make sure the image contains clearly visible tour dates.")
                return
                
            logger.info(f"Sending formatted response to Discord: {formatted_dates[:100]}...")
            
            try:
                # Create a combined message with both band names and formatted dates
                combined_message = ""
                
                # Add band names if available
                if band_names:
                    # Format band names as a list
                    band_list = ""
                    # Remove any headers like "Bands:" or "Band Names:"
                    cleaned_band_names = re.sub(r'^(?i)(bands?:?|band\s*names?:?)\s*', '', band_names.strip(), flags=re.MULTILINE)
                    
                    for i, band in enumerate(cleaned_band_names.split('\n')):
                        if band.strip():
                            # Remove any existing bullet points or dashes at the beginning of the line
                            clean_band = band.strip()
                            if clean_band.startswith('•') or clean_band.startswith('-') or clean_band.startswith('*'):
                                clean_band = clean_band[1:].strip()
                            band_list += f"• {clean_band}\n"
                    
                    combined_message += f"**Band Names:**\n```\n{band_list}```"
                
                # Add formatted dates - no extra line break
                # Remove any headers like "Tour Dates:" or "Formatted Dates:"
                cleaned_dates = re.sub(r'^(?i)(tour\s*dates?:?|formatted\s*dates?:?|dates?:?)\s*', '', formatted_dates.strip(), flags=re.MULTILINE)
                combined_message += f"\n**Formatted Dates:**\n```\n{cleaned_dates}\n```"
                
                # Add disclaimer
                combined_message += "\n\n*Note: Tour Date Drakemay make mistakes. Please verify important information.*"
                
                # Split the combined message if it's too long
                combined_chunks = split_message(combined_message)
                
                # First send the original image back to the user
                file = discord.File(BytesIO(image_bytes), filename=image.filename)
                await interaction.followup.send(file=file)
                
                # Then send the combined message
                for i, chunk in enumerate(combined_chunks):
                    if i == 0:
                        await interaction.followup.send(chunk)
                    else:
                        await interaction.followup.send(f"```\n(continued...)\n{chunk}\n```")
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                await interaction.followup.send(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        if not interaction.response.is_done():
            await interaction.response.send_message(f"Error: {str(e)}")
        else:
            await interaction.followup.send(f"Error: {str(e)}")

@client.tree.command(name="imageurl", description="Extract tour dates from an image URL")
@app_commands.describe(
    url="URL of an image containing tour dates"
)
async def imageurl(interaction: discord.Interaction, url: str):
    try:
        # Input validation
        if not url or not url.strip():
            await interaction.response.send_message("Error: Please provide a valid image URL.")
            return
            
        # Simple URL validation
        if not url.startswith(('http://', 'https://')):
            await interaction.response.send_message("Error: Invalid URL. Please provide a valid HTTP or HTTPS URL.")
            return

        # First respond that we're working on it
        try:
            await interaction.response.defer(thinking=True, ephemeral=False)
        except discord.errors.NotFound:
            logger.error("Interaction not found - it may have timed out")
            return
        except discord.errors.HTTPException as e:
            if e.code == 40060:  # Interaction already acknowledged
                logger.warning("Interaction was already acknowledged")
            else:
                raise
        
        async with aiohttp.ClientSession() as session:
            # Send the URL to our API
            logger.info(f"Sending image URL to API: {url}")
            async with session.post(
                f"{API_URL}/format/image",
                json={"url": url},
                timeout=aiohttp.ClientTimeout(total=180)  # 3 minute timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    try:
                        error_json = await response.json()
                        error_detail = error_json.get('detail', 'Unknown error')
                    except:
                        error_detail = error_text
                    logger.error(f"API error response: {error_text}")
                    try:
                        await interaction.followup.send(f"Error: {error_detail}")
                    except discord.NotFound:
                        logger.error("Interaction expired during error handling")
                    return
                    
                result = await response.json()
            
            formatted_dates = result.get("formatted_dates", "Error: No dates found")
            band_names = result.get("band_names", "")
            
            if not formatted_dates or formatted_dates == "Error: No dates found":
                await interaction.followup.send("I couldn't find any valid tour dates in the image. Please make sure the image contains clearly visible tour dates.")
                return
                
            logger.info(f"Sending formatted response to Discord: {formatted_dates[:100]}...")
            
            try:
                # Create a combined message with both band names and formatted dates
                combined_message = ""
                
                # Add band names if available
                if band_names:
                    # Format band names as a list
                    band_list = ""
                    # Remove any headers like "Bands:" or "Band Names:"
                    cleaned_band_names = re.sub(r'^(?i)(bands?:?|band\s*names?:?)\s*', '', band_names.strip(), flags=re.MULTILINE)
                    
                    for i, band in enumerate(cleaned_band_names.split('\n')):
                        if band.strip():
                            # Remove any existing bullet points or dashes at the beginning of the line
                            clean_band = band.strip()
                            if clean_band.startswith('•') or clean_band.startswith('-') or clean_band.startswith('*'):
                                clean_band = clean_band[1:].strip()
                            band_list += f"• {clean_band}\n"
                    
                    combined_message += f"**Band Names:**\n```\n{band_list}```"
                
                # Add formatted dates - no extra line break
                # Remove any headers like "Tour Dates:" or "Formatted Dates:"
                cleaned_dates = re.sub(r'^(?i)(tour\s*dates?:?|formatted\s*dates?:?|dates?:?)\s*', '', formatted_dates.strip(), flags=re.MULTILINE)
                combined_message += f"\n**Formatted Dates:**\n```\n{cleaned_dates}\n```"
                
                # Add disclaimer
                combined_message += "\n\n*Note: Tour Date Drake is AI-powered and may make mistakes. Please verify important information.*"
                
                # Split the combined message if it's too long
                combined_chunks = split_message(combined_message)
                
                # First send the image URL
                await interaction.followup.send(f"**Original Image:** {url}")
                
                # Then send the combined message
                for i, chunk in enumerate(combined_chunks):
                    if i == 0:
                        await interaction.followup.send(chunk)
                    else:
                        await interaction.followup.send(f"```\n(continued...)\n{chunk}\n```")
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                await interaction.followup.send(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing image URL: {str(e)}")
        if not interaction.response.is_done():
            await interaction.response.send_message(f"Error: {str(e)}")
        else:
            await interaction.followup.send(f"Error: {str(e)}")

@client.event
async def on_ready():
    try:
        # Force sync commands
        commands = await client.tree.sync()
        logger.info(f"Synced {len(commands)} commands: {[cmd.name for cmd in commands]}")
        
        invite_link = discord.utils.oauth_url(
            client.user.id,
            permissions=discord.Permissions(
                send_messages=True,
                read_messages=True,
                attach_files=True,
                read_message_history=True
            )
        )
        logger.info(f"Bot is ready! Logged in as {client.user}")
        logger.info(f"Invite the bot using this link: {invite_link}")
    except Exception as e:
        logger.error(f"Error syncing commands: {str(e)}", exc_info=True)

def run_bot():
    if not DISCORD_TOKEN:
        raise ValueError("DISCORD_TOKEN environment variable is not set")
    
    client.run(DISCORD_TOKEN) 
