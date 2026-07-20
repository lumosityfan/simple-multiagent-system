import discord
from discord import app_commands
import aiohttp
import os
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# Per-user rate limit config
DISCORD_RATE_LIMIT = 5
DISCORD_RATE_WINDOW = 60

user_request_log: dict[str, list[datetime]] = defaultdict(list)

def check_discord_rate_limit(user_id: str) -> tuple[bool, int]:
    """Returns (allowed, seconds_until_reset)."""
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(seconds=DISCORD_RATE_WINDOW)

    user_request_log[user_id] = [
        t for t in user_request_log[user_id] if t > window_start
    ]

    if len(user_request_log[user_id]) >= DISCORD_RATE_LIMIT:
        oldest = user_request_log[user_id][0]
        reset_in = int((oldest + timedelta(seconds=DISCORD_RATE_WINDOW) - now).total_seconds())
        return False, reset_in

    user_request_log[user_id].append(now)
    return True, 0

async def handle_request(interaction: discord.Interaction, prompt: str):
    """Shared handler for all slash commands with rate limiting."""
    allowed, reset_in = check_discord_rate_limit(str(interaction.user.id))
    if not allowed:
        await interaction.response.send_message(
            f"⏳ You're sending too many requests. Please wait {reset_in} seconds.",
            ephemeral=True  # only visible to the user who hit the limit
        )
        return
    
    await interaction.response.defer()
    result = await call_api(prompt, str(interaction.user.id))
    await send_response(interaction, result)

load_dotenv()

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000") + "/chat"

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

async def call_api(message: str, thread_id: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, json={
                "message": message,
                "thread_id": thread_id
            }) as resp:
                data = await resp.json()
                return format_response(data["response"])
    except Exception as e:
        return f"Error connecting to server: {e}"

async def send_response(interaction: discord.Interaction, result: str):
    """Handle Discord's 2000 character limit."""
    if len(result) > 2000:
        chunks = [result[i:i+1990] for i in range(0, len(result), 1990)]
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
    else:
        await interaction.followup.send(result)

# Bot setup
class MultiAgentBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()
        print("Slash commands synced")

client = MultiAgentBot()

# Slash commands

@client.tree.command(name="ask", description="Ask the assistant anything")
@app_commands.describe(prompt="Your question or request")
async def ask(interaction: discord.Interaction, prompt: str):
    await handle_request(interaction, prompt)

@client.tree.command(name="weather", description="Get current weather for a city")
@app_commands.describe(city="The city to get weather for")
async def weather(interaction: discord.Interaction, city: str):
    await handle_request(interaction, f"What is the weather in {city}?")

@client.tree.command(name="joke", description="Get a joke on any topic")
@app_commands.describe(topic="The topic for the joke (optional)")
async def joke(interaction: discord.Interaction, topic: str = "anything"):
    await handle_request(interaction, f"Tell me a joke about {topic}")

@client.tree.command(name="news", description="Get latest news on a topic")
@app_commands.describe(topic="The news topic (e.g. technology, sports, politics)")
async def news(interaction: discord.Interaction, topic: str = "top headlines"):
    await handle_request(interaction, f"Latest news on {topic}")

@client.tree.command(name="recipe", description="Get a recipe by dish name or ingredients")
@app_commands.describe(query="Dish name or list of ingredients")
async def recipe(interaction: discord.Interaction, query: str):
    await handle_request(interaction, f"Give me a recipe for {query}")

@client.tree.command(name="math", description="Solve a math problem")
@app_commands.describe(problem="The math problem to solve")
async def math(interaction: discord.Interaction, problem: str):
    await handle_request(interaction, problem)

@client.tree.command(name="translate", description="Translate text to another language")
@app_commands.describe(text="Text to translate", language="Target language")
async def translate(interaction: discord.Interaction, text: str, language: str):
    await handle_request(interaction, f"Translate '{text}' to {language}")

@client.tree.command(name="define", description="Get the definition of a word")
@app_commands.describe(word="The word to define")
async def define(interaction: discord.Interaction, word: str):
    await handle_request(interaction, f"What does {word} mean?")

@client.tree.command(name="currency", description="Convert between currencies")
@app_commands.describe(amount="Amount to convert", from_currency="Source currency (e.g. USD)", to_currency="Target currency (e.g. EUR)")
async def currency(interaction: discord.Interaction, amount: float, from_currency: str, to_currency: str):
    await handle_request(interaction, f"Convert {amount} {from_currency} to {to_currency}")

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
        
        allowed, reset_in = check_discord_rate_limit(str(message.author.id))
        if not allowed:
            await message.channel.send(
                f"⏳ {message.author.mention} You're sending too many requests. "
                f"Please wait {reset_in} seconds."
            )
            return
        
        async with message.channel.typing():
            result = await call_api(content, str(message.author.id))
        
        if len(result) > 2000:
            chunks = [result[i:i+1990] for i in range(0, len(result), 1990)]
            for chunk in chunks:
                await message.channel.send(chunk)
        else:
            await message.channel.send(result)

client.run(os.getenv("DISCORD_BOT_TOKEN"))