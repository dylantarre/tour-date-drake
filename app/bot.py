import os
import discord
from discord import app_commands
import aiohttp
import logging
from dotenv import load_dotenv
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("API_URL", "http://api:4343")
MAX_DISCORD_LENGTH = 1990  # Leave some room for the code block markers

def split_message(message: str) -> list[str]:
    """Split a message into chunks that fit within Discord's character limit."""
    chunks = []
    current_chunk = ""
    
    for line in message.split('\n'):
        # If adding this line would exceed the limit, start a new chunk
        if len(current_chunk) + len(line) + 1 > MAX_DISCORD_LENGTH:
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
        # First respond that we're working on it
        await interaction.response.defer(thinking=True, ephemeral=False)
        
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
                    await interaction.followup.send(f"Error: {error_detail}")
                    return
                    
                result = await response.json()

            formatted_dates = result.get("formatted_dates", "Error: No dates found") + "\n\nPlease double-check all info as Better Lover can make mistakes."
            logger.info(f"Sending formatted response to Discord: {formatted_dates}")
            
            # Split long messages
            chunks = split_message(formatted_dates)
            
            # Send first chunk as initial response
            try:
                await interaction.followup.send(f"```\n{chunks[0]}\n```")
            except discord.NotFound:
                logger.error("Initial interaction expired, creating new message")
                return
                
            # Send remaining chunks as follow-ups
            if len(chunks) > 1:
                try:
                    for chunk in chunks[1:]:
                        await interaction.followup.send(f"```\n(continued...)\n{chunk}\n```")
                except discord.NotFound:
                    logger.error("Follow-up interaction expired")
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

@client.tree.command(name="read", description="Extract tour dates from an image")
async def read(interaction: discord.Interaction, image: discord.Attachment):
    try:
        # First respond that we're working on it
        await interaction.response.defer(thinking=True, ephemeral=False)
        
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

            formatted_dates = result.get("formatted_dates", "Error: No dates found") + "\n\nPlease double-check all info as Better Lover can make mistakes."
            logger.info(f"Sending formatted response to Discord: {formatted_dates}")
            
            # Split long messages
            chunks = split_message(formatted_dates)
            
            # Send first chunk as initial response
            try:
                await interaction.followup.send(f"```\n{chunks[0]}\n```")
            except discord.NotFound:
                logger.error("Initial interaction expired, creating new message")
                return
                
            # Send remaining chunks as follow-ups
            if len(chunks) > 1:
                try:
                    for chunk in chunks[1:]:
                        await interaction.followup.send(f"```\n(continued...)\n{chunk}\n```")
                except discord.NotFound:
                    logger.error("Follow-up interaction expired")
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

@client.tree.command(name="imageurl", description="Extract tour dates from an image URL")
@app_commands.describe(
    url="URL of the image to process (jpg, png, gif, webp)"
)
async def imageurl(interaction: discord.Interaction, url: str):
    try:
        # First respond that we're working on it
        await interaction.response.defer(thinking=True, ephemeral=False)
        
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

            formatted_dates = result.get("formatted_dates", "Error: No dates found") + "\n\nPlease double-check all info as Better Lover can make mistakes."
            logger.info(f"Sending formatted response to Discord: {formatted_dates}")
            
            # Split long messages
            chunks = split_message(formatted_dates)
            
            # Send first chunk as initial response
            try:
                await interaction.followup.send(f"```\n{chunks[0]}\n```")
            except discord.NotFound:
                logger.error("Initial interaction expired, creating new message")
                return
                
            # Send remaining chunks as follow-ups
            if len(chunks) > 1:
                try:
                    for chunk in chunks[1:]:
                        await interaction.followup.send(f"```\n(continued...)\n{chunk}\n```")
                except discord.NotFound:
                    logger.error("Follow-up interaction expired")
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