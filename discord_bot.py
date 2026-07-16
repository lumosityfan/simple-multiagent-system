import discord
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://localhost:8000/chat"

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

AGENT_EMOJIS = {
    "math": "🧮",
    "weather": "🌤️",
    "joke": "😂",
    "news": "📰",
    "recipe": "🍳",
    "translation": "🌐",
    "dictionary": "📖",
    "currency": "💱",
    "fallback": "🤖",
}

def format_response(response: str) -> str:
    formatted_blocks = []
    for block in response.split("\n\n"):
        for agent, emoji in AGENT_EMOJIS.items():
            if block.startswith(f"{agent} - "):
                content = block[len(f"{agent} - "):].strip()
                block = f"{emoji}\n{content}"  # emoji on its own line
                break
        formatted_blocks.append(block)
    return "\n".join(formatted_blocks)

@client.event
async def on_ready():
    print(f"Bot logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if isinstance(message.channel, discord.DMChannel) or client.user.mentioned_in(message):
        content = message.content.replace(f"<@{client.user.id}>", "").strip()
        if not content:
            return
        async with message.channel.typing():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(API_URL, json={
                        "message": content,
                        "thread_id": str(message.author.id)
                    }) as resp:
                        data = await resp.json()
                        result = format_response(data["response"])
            except Exception as e:
                result = f"Error: {e}"
        
        # Handle Discord's 2000 character limit
        if len(result) > 2000:
            chunks = [result[i:i+1990] for i in range(0, len(result), 1990)]
            for chunk in chunks:
                await message.channel.send(chunk)
        else:
            await message.channel.send(result)

client.run(os.getenv("DISCORD_BOT_TOKEN"))