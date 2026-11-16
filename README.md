# Simple Multi-Agent System

A supervisor-style multi-agent assistant built with **LangGraph**. A single request such as *"What's the weather in Tokyo, and convert 100 USD to yen?"* is split into sub-tasks, routed to specialist agents that run in parallel, then merged into one answer. You can talk to it through a **Discord bot** or a **Streamlit** chat app, both backed by one **FastAPI** service.

## Demo

> 🚧 **Screenshots and a short demo GIF of the Streamlit app and Discord bot are coming soon.**

## How it works

```
user message
     │
     ▼
 split_node ── resolves context from conversation history, splits into sub-prompts
     │
     ▼
 parallel_dispatch ── routes each sub-prompt to an agent and fires them simultaneously (LangGraph `Send`)
     │
     ├── math ── weather ── news ── recipe
     ├── joke ── translation ── dictionary ── currency      (+ fallback)
     ▼
 merge_node ── collects the parallel results in the original order
     │
     ▼
 supervisor_node ── reviews each result and sends failed ones back with a retry instruction
     │
     ▼
 history_node ── updates conversation history ──► END
```

**Highlights**

- **Parallel fan-out and merge:** multi-part requests run concurrently instead of one after another.
- **Supervisor with retries:** results are checked, and an agent that missed the mark is re-run with a corrective note.
- **Persistent memory:** conversation state is checkpointed per `thread_id` in Postgres (`DATABASE_URL`) or a local SQLite file when no database is configured.
- **Context-aware agents:** agents can reuse earlier results, for example the math agent using weather data, and the joke agent avoids repeating jokes.
- **Rate limiting:** per-thread limits on the API (10 requests/min) and per-user limits on the Discord bot (5 requests/min).

## Agents

| Agent | What it does | Data source |
|---|---|---|
| math | Solves math problems | OpenAI |
| weather | Current weather for a city | OpenWeatherMap |
| news | Latest headlines on a topic | NewsAPI |
| currency | Currency conversion | Exchange-rate API |
| recipe | Recipes by dish or ingredients | OpenAI |
| joke | Jokes on any topic, no repeats | OpenAI |
| translation | Translates text | OpenAI |
| dictionary | Word definitions | OpenAI |
| fallback | Handles anything else | OpenAI |

## Project layout

| File | Purpose |
|---|---|
| `app.py` | LangGraph graph, supervisor logic and all agents |
| `server.py` | FastAPI service exposing `POST /chat` and `GET /health` |
| `streamlit_app.py` | Streamlit chat UI |
| `discord_bot.py` | Discord bot with slash commands: `/ask`, `/weather`, `/joke`, `/news`, `/recipe`, `/math`, `/translate`, `/define`, `/currency` |
| `Procfile`, `railway.toml` | Deployment config (Railway) |

## Getting started

**Prerequisites:** Python 3.11+, an OpenAI API key, and optionally keys for the weather, news and exchange-rate APIs and a Discord bot token.

```bash
git clone https://github.com/lumosityfan/simple-multiagent-system.git
cd simple-multiagent-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then fill in your keys
```

| Variable | Needed for |
|---|---|
| `OPENAI_API_KEY` | all agents |
| `OPENWEATHERMAP_API_KEY` | weather agent |
| `NEWS_API_KEY` | news agent |
| `EXCHANGE_RATE_API_KEY` | currency agent |
| `DISCORD_BOT_TOKEN` | Discord bot |
| `DATABASE_URL` | Postgres checkpointing (optional; SQLite is used if unset) |

Run the API, then whichever front end you want, each in its own terminal:

```bash
uvicorn server:app --port 8000       # API
streamlit run streamlit_app.py       # web chat UI
python discord_bot.py                # Discord bot
```

The front ends call the API at `FASTAPI_URL` (default `http://localhost:8000`).

### API example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a joke about cats and define the word ephemeral", "thread_id": "demo"}'
```

## Deployment

The repo includes a `Procfile` and `railway.toml` that define three services (FastAPI, Streamlit and the Discord bot) for deployment on [Railway](https://railway.app).

## Tech stack

Python · LangGraph · FastAPI · Streamlit · discord.py · PostgreSQL / SQLite · OpenAI API (`gpt-4o-mini`)

## Roadmap

- [ ] Add tests for the router, the supervisor retry logic and the API
- [ ] Restrict CORS on the API (currently open to all origins)
- [ ] Add a `LICENSE`
