import os
import discord
from discord import app_commands
import aiohttp
import logging
from dotenv import load_dotenv
import asyncio
from io import BytesIO

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
                        disclaimer = "\nPlease double-check all info as Tour Date Drake can make mistakes."
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
                            # Add disclaimer to the last sub-chunk of the last chunk
                            if j == len(sub_chunks) - 1 and i == len(formatted_chunks) - 1:
                                if len(sub_message) + len(disclaimer) <= MAX_DISCORD_LENGTH:
                                    sub_message += disclaimer
                                else:
                                    await interaction.followup.send(sub_message)
                                    await interaction.followup.send(disclaimer)
                                    continue
                            await interaction.followup.send(sub_message)
                    else:
                        await interaction.followup.send(message)
                    
            except discord.NotFound:
                logger.error("Initial interaction expired")
                return

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
        # First respond that we're working on it
        try:
            await interaction.response.defer(thinking=True, ephemeral=False)
        except discord.errors.HTTPException as e:
            if e.code == 40060:  # Interaction already acknowledged
                logger.warning("Interaction already acknowledged, skipping")
                return
            raise
        
        async with aiohttp.ClientSession() as session:
            # Process image
            logger.info(f"Processing image: {image.filename}")
            
            # Download the image
            image_data = await image.read()
            
            # Send to our API using proper multipart form
            form = aiohttp.FormData()
            form.add_field('file', 
                          image_data,
                          filename=image.filename,
                          content_type=image.content_type)
            
            async with session.post(
                f"{API_URL}/format/image",
                data=form,
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
                    await interaction.followup.send(f"Error: {error_detail}")
                    return
                result = await response.json()
                logger.info(f"Parsed API response: {result}")

            formatted_dates = result.get("formatted_dates", "Error: No dates found")
            if not formatted_dates or formatted_dates == "Error: No dates found":
                logger.warning("No dates found in the image")
                await interaction.followup.send(file=discord.File(fp=BytesIO(image_data), filename=image.filename))
                await interaction.followup.send("I couldn't find any tour dates in this image. Please make sure the image contains clear tour date information.")
                return
                
            logger.info(f"Sending formatted response to Discord: {formatted_dates}")
            
            # Split long messages
            chunks = split_message(formatted_dates)
            
            try:
                # Send first chunk as initial response with the image
                await interaction.followup.send(file=discord.File(fp=BytesIO(image_data), filename=image.filename))
                for i, chunk in enumerate(chunks):
                    message = f"```\n{chunk}\n```"
                    if i > 0:
                        message = f"```\n(continued...)\n{chunk}\n```"
                    if i == len(chunks) - 1:
                        message += "\nPlease double-check all info as Tour Date Drake can make mistakes."
                    await interaction.followup.send(message)
                    
            except discord.NotFound:
                logger.error("Interaction expired")
                return
                
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        try:
            await interaction.followup.send("Sorry, the request took too long to process. Please try again with a smaller image or check your internet connection.")
        except discord.NotFound:
            logger.error("Interaction expired during timeout")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        try:
            await interaction.followup.send("Sorry, something went wrong while processing your request. Please try again later.")
        except discord.NotFound:
            logger.error("Interaction expired during error handling")

@client.tree.command(name="imageurl", description="Extract tour dates from an image URL")
@app_commands.describe(
    url="URL of the image to process (jpg, png, gif, webp)"
)
async def imageurl(interaction: discord.Interaction, url: str):
    try:
        # First respond that we're working on it
        try:
            await interaction.response.defer(thinking=True, ephemeral=False)
        except discord.errors.HTTPException as e:
            if e.code == 40060:  # Interaction already acknowledged
                logger.warning("Interaction already acknowledged, skipping")
                return
            raise

        async with aiohttp.ClientSession() as session:
            # Download the image from URL
            logger.info(f"Downloading image from URL: {url}")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download image: HTTP {response.status}")
                
                # Get content type and filename
                content_type = response.headers.get('content-type', 'image/jpeg')
                filename = url.split('/')[-1]
                
                # Download the image data
                image_data = await response.read()
                
                # Send to API using the same format as successful image upload
                form = aiohttp.FormData()
                form.add_field('file',
                             image_data,
                             filename=filename,
                             content_type=content_type)
                
                async with session.post(
                    f"{API_URL}/format/image",
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=180)  # 3 minute timeout
                ) as api_response:
                    if api_response.status != 200:
                        error_text = await api_response.text()
                        try:
                            error_json = await api_response.json()
                            error_detail = error_json.get('detail', 'Unknown error')
                        except:
                            error_detail = error_text
                        logger.error(f"API error response: {error_text}")
                        await interaction.followup.send(f"Error: {error_detail}")
                        return
                    result = await api_response.json()
                    logger.info(f"Parsed API response: {result}")

            formatted_dates = result.get("formatted_dates", "Error: No dates found")
            if not formatted_dates or formatted_dates == "Error: No dates found":
                logger.warning("No dates found in the image")
                await interaction.followup.send(file=discord.File(fp=BytesIO(image_data), filename=filename))
                await interaction.followup.send("I couldn't find any tour dates in this image. Please make sure the image contains clear tour date information.")
                return
                
            logger.info(f"Sending formatted response to Discord: {formatted_dates}")
            
            # Split long messages
            chunks = split_message(formatted_dates)
            
            try:
                # Send first chunk as initial response with the image
                await interaction.followup.send(file=discord.File(fp=BytesIO(image_data), filename=filename))
                for i, chunk in enumerate(chunks):
                    message = f"```\n{chunk}\n```"
                    if i > 0:
                        message = f"```\n(continued...)\n{chunk}\n```"
                    if i == len(chunks) - 1:
                        message += "\nPlease double-check all info as Tour Date Drake can make mistakes."
                    await interaction.followup.send(message)
                    
            except discord.NotFound:
                logger.error("Interaction expired")
                return
                
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        try:
            await interaction.followup.send("Sorry, the request took too long to process. Please try again with a smaller image or check your internet connection.")
        except discord.NotFound:
            logger.error("Interaction expired during timeout")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        try:
            await interaction.followup.send("Sorry, something went wrong while processing your request. Please try again later.")
        except discord.NotFound:
            logger.error("Interaction expired during error handling")

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