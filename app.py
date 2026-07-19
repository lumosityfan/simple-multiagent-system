from dotenv import load_dotenv
from openai import OpenAI
import uuid
import json
import os
import requests
from typing import Annotated
from langgraph.types import Send
from datetime import datetime, timezone, timedelta
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import operator

load_dotenv(override=True)

def merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}

def replace_or_add(existing: list, new: list) -> list:
    """If new is empty, reset. Otherwise append."""
    if not new:
        return []
    return existing + new

agent = None # set by server.py lifespan or run() below
THREAD_ID = "main-session"

class AgentState(TypedDict):
    user_prompt: str
    sub_prompts: list[str]
    current_prompt: str
    last_agent: str
    last_result: str
    next_agent: str
    results: list[str]
    parallel_results: Annotated[list[dict], replace_or_add]
    joke_history: list[str]
    weather_cache: Annotated[dict, merge_dicts]
    currency_cache: Annotated[dict, merge_dicts]
    news_cache: Annotated[dict, merge_dicts]
    conversation_history: list[dict]

def split_node(state: AgentState) -> dict:
    resolved = resolve_context(state["user_prompt"], state.get("conversation_history", []))
    sub_prompts = split_prompt(resolved)
    return {"sub_prompts": sub_prompts}

def parallel_dispatch(state: AgentState):
    """
    Replaces route_node for multi-sub-prompt requests.
    Fires a Send for each sub-prompt simultaneously.
    """
    sends = []
    for sub_prompt in state["sub_prompts"]:
        # route each sub-prompt to find its agent
        decision = route_prompt(sub_prompt)
        agent_name = decision.get("agent", "fallback")
        node = agent_name if agent_name in NODES else "fallback"
        sends.append(Send(node, {
            **state,
            "current_prompt": sub_prompt,
            "last_agent": state.get("last_agent", ""),
            "last_result": state.get("last_result", ""),
            "parallel_results": [], 
            "results": [],
        }))
    return sends

def merge_node(state: AgentState) -> dict:
    """
    Collects all parallel results after agents finish.
    Combines them into last_result and results for history_node.
    """
    combined_results = state.get("parallel_results", [])
    # sort by original sub_prompt order to keep output consistent
    combined_results.sort(key=lambda x: state["sub_prompts"].index(x["sub_prompt"])
                          if x["sub_prompt"] in state["sub_prompts"] else 0)
    
    last = combined_results[-1] if combined_results else {}
    return {
        "last_agent": last.get("agent", ""),
        "last_result": last.get("result", ""),
        "results": [f"{r['agent']} - {r['result']}" for r in combined_results],
    }

def math_node(state: AgentState) -> dict:
    prompt = state["current_prompt"]
    if state.get("last_agent") == "weather" and state.get("last_result"):
        prompt += f"\n\nFor context, use this weather data: {state['last_result']}"
    result = math_agent(prompt)
    return {
        "parallel_results": [{"agent": "math", "result": result, "sub_prompt": state["current_prompt"]}],
        }

def weather_node(state: AgentState) -> dict:
    weather_cache = state.get("weather_cache", {})
    result = weather_agent(state["current_prompt"], weather_cache)
    return {
        "weather_cache": weather_cache,
        "parallel_results": [{"agent": "weather", "result": result, "sub_prompt": state["current_prompt"]}],
        }

def joke_node(state: AgentState) -> dict:
    joke_history = state.get("joke_history", [])
    history_note = ""
    if joke_history:
        history_note = f" Do NOT tell any of these jokes: {'; '.join(joke_history)}"
    prompt = state["current_prompt"]
    if state.get("last_agent") == "math" and state.get("last_result"):
        math_answer = extract_math_answer(state["last_result"])
        prompt += f"\n\nFor context, the previous math result was: {math_answer}"
    if state.get("last_agent") == "weather" and state.get("last_result"):
        prompt += f"\n\nFor context, the current weather is: {state['last_result']}"
    if state.get("last_agent") == "news" and state.get("last_result"):
        prompt += f"\n\nFor context, the requested news is: {state['last_result']}"
    result = joke_agent(prompt, history_note)
    return {
            "parallel_results": [{"agent": "joke", "result": result, "sub_prompt": state["current_prompt"]}],
            "joke_history": joke_history + [result]}

def translation_node(state: AgentState) -> dict:
    result = translation_agent(state["current_prompt"])
    return {
            "parallel_results": [{"agent": "translation", "result": result, "sub_prompt": state["current_prompt"]}]}

def dictionary_node(state: AgentState) -> dict:
    result = dictionary_agent(state["current_prompt"])
    return { 
            "parallel_results": [{"agent": "dictionary", "result": result, "sub_prompt": state["current_prompt"]}]}

def recipe_node(state: AgentState) -> dict:
    result = recipe_agent(state["current_prompt"])
    return {
            "parallel_results": [{"agent": "recipe", "result": result, "sub_prompt": state["current_prompt"]}]}

def news_node(state: AgentState) -> dict:
    news_cache = state.get("news_cache", {})
    result = news_agent(state["current_prompt"], news_cache)
    return { "parallel_results": [{"agent": "news", "result": result, "sub_prompt": state["current_prompt"]}], "news_cache": news_cache}

def currency_node(state: AgentState) -> dict:
    currency_cache = state.get("currency_cache", {})
    result = currency_agent(state["current_prompt"], currency_cache)
    return {"parallel_results": [{"agent": "currency", "result": result, "sub_prompt": state["current_prompt"]}], "currency_cache": currency_cache}

def fallback_node(state: AgentState) -> dict:
    result = fallback_agent(state["current_prompt"])
    return {"parallel_results": [{"agent": "fallback", "result": result, "sub_prompt": state["current_prompt"]}]}

def history_node(state: AgentState) -> dict:
    history = list(state.get("conversation_history", []))
    if not history or history[-1].get("content") != state["user_prompt"] or history[-1].get("role") != "user":
        history.append({
            "role": "user",
            "content": state["user_prompt"]
        })
    history.append({
        "role": "assistant",
        "agent": state["last_agent"],
        "content": state["last_result"]
    })
    return {"conversation_history": history[-20:]}

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def resolve_context(user_prompt: str, conversation_history: list[dict]) -> str:
    if not conversation_history:
        return user_prompt
    
    history_text = "\n".join([
        f"{entry['role'].upper()} ({entry.get('agent', '')}): {entry['content'][:500]}"
        for entry in conversation_history[-10:]  # last 10 turns
    ])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a context resolver. Given a conversation history and a new user prompt, "
                "rewrite the prompt to be fully self-contained if it references previous context "
                "(e.g. 'that', 'it', 'the last one', 'make it vegetarian'). "
                "If the prompt is already self-contained, return it unchanged. "
                "Respond with ONLY the rewritten prompt, nothing else."
            )},
            {"role": "user", "content": f"Conversation history:\n{history_text}\n\nNew prompt: {user_prompt}"}
        ]
    )
    return response.choices[0].message.content.strip()

def extract_math_answer(math_result):
    system_prompt = "Extract only the final numerical answer or solution from this math result. Be extremely concise, one line only."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": math_result}
        ]
    )
    return response.choices[0].message.content.strip()

def math_agent(user_prompt):
    system_prompt = """You are a math expert who is here to help solve mathematical problems.
                       Please be detailed in your steps and solve the problem clearly, being as concise as possible.
                       Please provide it in syntax and formatting that is visible in a terminal screen.
                       Do NOT use LaTeX notation (no \\( \\), no \\[ \\], no \\boxed{}, no \\frac{}).
                       Use plain text and ASCII formatting only. For example:
                       - Use 'x^2' instead of '\\( x^2 \\)'
                       - Use 'sqrt(x)' instead of '\\( \\sqrt{x} \\)'
                       - Use fractions as 'a/b' instead of '\\( \\frac{a}{b} \\)'"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def weather_agent(user_prompt, weather_cache):
    system_prompt = """You are given the user's weather prompt. Extract the city name and respond with ONLY the city name, nothing else.
                       If no city is mentioned, respond with 'unknown'."""
    # Step 1: use the LLM to extract the location from the prompt
    extraction = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    city = extraction.choices[0].message.content.strip()

    if city.lower() == "unknown":
        # Get current location for weather
        try:
            response = requests.get('https://ipinfo.io')
            data = response.json()
            city = data.get('city', 'unknown')
        except Exception as e:
            return f"Error retrieving location: {e}"
        
    # Check for cached location
    if city not in weather_cache or datetime.now(timezone.utc) - datetime.fromisoformat(weather_cache[city]["timestamp"]) >= CACHE_TTL:
        # if not, Step 2: hit the OpenWeatherMap API
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        url = "https://api.openweathermap.org/data/2.5/weather"
        response = requests.get(url, params={
            "q": city,
            "appid": api_key,
            "units": "imperial",
        })

        if response.status_code == 404:
            return f"Couldn't find weather data for '{city}'."
        if response.status_code != 200:
            return f"Weather API error: {response.status_code}"
        
        data = response.json()
        weather_cache[city] = {
            "temp": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    cached_weather = weather_cache[city]

    return (
        f"Weather in {city}: {cached_weather["description"]}, {cached_weather["temp"]} degrees F "
        f"(feels like {cached_weather["feels_like"]} degrees F), humidity {cached_weather["humidity"]}%."
    )

def joke_agent(user_prompt, history_note):
    system_prompt = f"""You are a comedian. Tell a funny, clever joke relevant to the user's prompt. Keep it short.\
        Please specifically reference any numbers of values that may be provided in the math context.{history_note}"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def translation_agent(user_prompt):
    system_prompt = """You are a translation expert. Translate the text as requested.
                       Identify the source and target language, then provide the translation.
                       Format: 'Translation (<source> → <target>): <translated text>'"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def dictionary_agent(user_prompt):
    system_prompt = """You are a dictionary and language expert. Provide definitions, 
                       synonyms, antonyms, and etymology for words as requested.
                       Be concise but thorough. Format clearly for a terminal screen."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def recipe_agent(user_prompt):
    system_prompt = """You are a professional chef and recipe expert. When given ingredients, 
                       a cuisine type, or a dish name, provide a clear and concise recipe.
                       Include: dish name, ingredients with quantities, and numbered steps.
                       Keep it practical and formatted for a terminal screen."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def news_agent(user_prompt, news_cache):
    extraction_prompt = extraction_prompt = """Extract the specific topic or keyword the user wants news about.
Respond with ONLY the topic keyword(s), nothing else.
If the user is asking about a broad category like technology, sports, politics, business, etc., return that category.
Only respond with 'top headlines' if the user gives NO topic at all (e.g. 'what's in the news today?').

Examples:
'What are the latest news in technology?' → 'technology'
'Latest news on AI' → 'artificial intelligence'
'What's happening in politics?' → 'politics'
'Tell me about climate change news' → 'climate change'
'What's in the news?' → 'top headlines'
'Give me the headlines' → 'top headlines'"""
    extraction_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    topic = extraction_response.choices[0].message.content.strip()

    if topic not in news_cache or datetime.now(timezone.utc) - datetime.fromisoformat(news_cache[topic]["timestamp"]) >= CACHE_TTL:    
        api_key = os.getenv("NEWS_API_KEY")
        url = "https://newsapi.org/v2/everything" if topic != "top headlines" else "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": api_key,
            "language": "en",
            "pageSize": 5,
        }
        if topic != "top headlines":
            params["q"] = topic
        else:
            params["country"] = "us"

        response = requests.get(url, params=params)
        if response.status_code != 200:
            return f"News API error: {response.status_code}"
    
        articles = response.json().get("articles", [])
        if not articles:
            category_map = {
                "technology": "technology",
                "tech": "technology", 
                "ai": "technology",
                "artificial intelligence": "technology",
                "sports": "sports",
                "business": "business",
                "health": "health",
                "science": "science",
                "entertainment": "entertainment",
            }
            category = category_map.get(topic.lower())
            params = {
                "apiKey": api_key,
                "country": "us",
                "language": "en",
                "pageSize": 5,
            }
            if category:
                params["category"] = category
            else:
                params["q"] = topic  # retry with top-headlines endpoint
            
            response = requests.get("https://newsapi.org/v2/top-headlines", params=params)
            articles = response.json().get("articles", [])

        # Stage 3: if still empty, say so clearly
        if not articles:
            return f"Couldn't find recent news for '{topic}'. Try a broader topic."
    
        lines = [f"Top news for '{topic}':\n"]
        for i, article in enumerate(articles, 1):
            lines.append(f"{i}. {article['title']}")
            lines.append(f"   Source: {article['source']['name']}")
            lines.append(f"   {article['description'] or ''}\n")

        news_cache[topic] = {
            "articles": lines,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    lines = news_cache[topic]["articles"]

    return "\n".join(lines)

def currency_agent(user_prompt, currency_cache):
    extraction_prompt = """Extract the conversion details from the user's currency prompt.
    Respond ONLY with a JSON object like: 
    {"amount": 100, "from": "USD", "to": "EUR"}
    Use standard 3-letter currency codes. If the amount is not specified, use 1."""
    extraction_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    try:
        details = json.loads(extraction_response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        return "Couldn't parse the currency conversion request."
    
    amount = details.get("amount", 1)
    from_currency = details.get("from", "").upper()
    to_currency = details.get("to", "").upper()

    if not from_currency or not to_currency:
        return "Please specify both currencies, e.g. 'Convert 100 USD to EUR'."
    
    currency_key = f"{from_currency}->{to_currency}"
    if currency_key not in currency_cache or datetime.now(timezone.utc) - datetime.fromisoformat(currency_cache[currency_key]["timestamp"]) >= CACHE_TTL:
        api_key = os.getenv("EXCHANGE_RATE_API_KEY")
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency}/{to_currency}/{amount}"
        response = requests.get(url)

        if response.status_code != 200:
            return f"Currency API error: {response.status_code}"
    
        data = response.json()
        if data.get("result") != "success":
            return f"Conversion failed: {data.get('error-type', 'unknown error')}"

        currency_cache[currency_key] = {
            "rate": data["conversion_rate"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    rate = currency_cache[currency_key]["rate"]
    converted = amount * rate

    return (
        f"{amount} {from_currency} = {converted:.2f} {to_currency}\n"
        f"Exchange rate: 1 {from_currency} = {rate:.4f} {to_currency}"
    )

def fallback_agent(user_prompt):
    system_prompt = "You are a helpful general assistant. Answer the question as best as you can. Be concise."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def split_prompt(user_prompt: str) -> list[str]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SPLITTER_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    return json.loads(response.choices[0].message.content)

def route_prompt(user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    return json.loads(response.choices[0].message.content)

# Define constants
SPLITTER_PROMPT = """Given a user prompt, split it into independent sub-prompts if it contains multiple distinct requests.
Respond ONLY with a JSON array of strings.
If it's a single request, return an array with just that one prompt.
When there are a group of words after 'about' for jokes, make sure to contain everything after the about until the next independent sub-prompt.
For instance, "Tell me a joke about weather and rain" -> ["Tell me a joke about weather and rain"]
For weather when there are two separate cities, make sure to separate into 2 separate sub-prompts.
Examples:
"What is 2+2?" → ["What is 2+2?"]
"Tell me a joke and what is the weather?" → ["Tell me a joke", "What is the weather?"]
"What is the weather in Portland and Seattle?" → ["What is the weather in Portland?", "What is the weather in Seattle?"]
"Solve x² - 5x + 6 = 0 and tell me a joke about math" → ["Solve x² - 5x + 6 = 0", "Tell me a joke about math"]
"What's the weather in Seattle and convert the temperature to Celsius?" → ["What's the weather in Seattle?", "Convert the temperature to Celsius"]
"""

ROUTER_PROMPT = """You are a routing assistant. Given a user prompt, respond ONLY with a JSON object like:
{"agent": "<agent_name>"} where agent_name is one of: math, weather, joke, translation, dictionary, recipe, news, currency, fallback

Rules:
- math: calculations, equations, unit conversions, temperature conversion
- weather: current weather conditions in a city
- joke: humor, jokes, anything meant to be funny
- translation: converting text from one language to another
- dictionary: word definitions, synonyms, antonyms, etymology, spelling
- recipe: cooking instructions, ingredient lists, meal ideas, how to make a dish
- news: recent headlines, current events, what's happening in the world
- currency: exchange rates, converting between currencies

Examples:
"Convert the temperature to Celsius" → {"agent": "math"}
"What's the weather in Tokyo?" → {"agent": "weather"}
"Translate 'hello' to Spanish" → {"agent": "translation"}
"What does ephemeral mean?" → {"agent": "dictionary"}
"Tell me a joke" → {"agent": "joke"}
"How do I make pasta carbonara?" → {"agent": "recipe"}
"What can I make with chicken and rice?" → {"agent": "recipe"}
"What's in the news today?" → {"agent": "news"}
"Convert 100 USD to EUR" → {"agent": "currency"}
No other text."""

AGENTS = {
    "math": math_agent,
    "weather": weather_agent,
    "joke": joke_agent,
    "translation": translation_agent,
    "dictionary": dictionary_agent,
    "recipe": recipe_agent,
    "news": news_agent,
    "currency": currency_agent,
    "fallback": fallback_agent
}

NODES = {
    "math": math_node,
    "weather": weather_node,
    "joke": joke_node,
    "translation": translation_node,
    "dictionary": dictionary_node,
    "recipe": recipe_node,
    "news": news_node,
    "currency": currency_node,
    "fallback": fallback_node,
}

CACHE_TTL = timedelta(minutes=15)

def run():
    print("Multi-agent assistant ready. Type 'quit', 'exit', or 'q' to exit.")
    print(f"Agents available: {', '.join(AGENTS.keys())}\n")

    config = {"configurable": {"thread_id": THREAD_ID}}

    existing = agent.get_state(config)
    first_invoke = not existing.values
    conversation_history = existing.values.get("conversation_history", [])
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        import asyncio
        result = asyncio.run(chat(user_input))
        print(result)

agent_builder = StateGraph(AgentState)

agent_builder.add_node("split", split_node)
agent_builder.add_node("merge", merge_node)

agent_builder.add_edge(START, "split")

agent_builder.add_conditional_edges("split", parallel_dispatch)

agent_builder.add_node("history", history_node)

for node in NODES:
    agent_builder.add_node(node, NODES[node])
    agent_builder.add_edge(node, "merge")

agent_builder.add_edge("merge", "history")
agent_builder.add_edge("history", END)

async def chat(user_input: str, thread_id: str = "main-session") -> str:
    """
    Main entry point for all interfaces.
    Takes a user message, runs it through the graph, returns the result string.
    """
    config = {"configurable": {"thread_id": thread_id}}
    existing = agent.get_state(config)
    first_invoke = not existing.values

    invoke_state = {
        "user_prompt": user_input,
        "current_prompt": user_input,
        "sub_prompts": [],
        "last_agent": "",
        "last_result": "",
        "next_agent": "",
        "results": [],
        "parallel_results": [],
        "conversation_history": existing.values.get("conversation_history", []),
    }

    if first_invoke:
        invoke_state["joke_history"] = []
        invoke_state["weather_cache"] = {}
        invoke_state["currency_cache"] = {}
        invoke_state["news_cache"] = {}
        invoke_state["conversation_history"] = []
        first_invoke = False

    final_state = agent.invoke(invoke_state, config=config)
    return "\n\n".join(final_state.get("results", []))

if __name__ == "__main__":
    db_url = os.getenv("DATABASE_URL", "")
    if db_url:
        from langgraph.checkpoint.postgres import PostgresSaver
        checkpointer_cm = PostgresSaver.from_conn_string(db_url)
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer_cm = SqliteSaver.from_conn_string("checkpoints.db")

    with checkpointer_cm as checkpointer:
        checkpointer.setup()
        agent = agent_builder.compile(checkpointer=checkpointer)
        run()