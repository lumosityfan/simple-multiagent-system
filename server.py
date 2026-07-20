from contextlib import asynccontextmanager
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
import psycopg
import os
import app as agent_app

# Rate limit config
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60

# Track request timestamps per client IP
request_log: dict[str, list[datetime]] = defaultdict(list)
log_lock = asyncio.Lock()

async def check_rate_limit(client_ip: str) -> bool:
    """Returns True if request is allowed, False if rate limited."""
    async with log_lock:
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)

        # remove timestamps outside the window
        request_log[client_ip] = [
            t for t in request_log[client_ip] if t > window_start
        ]

        if len(request_log[client_ip]) >= RATE_LIMIT_REQUESTS:
            return False
        
        request_log[client_ip].append(now)
        return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = os.getenv("DATABASE_URL", "")
    if db_url:
        checkpointer_cm = PostgresSaver.from_conn_string(db_url)
    else:
        checkpointer_cm = SqliteSaver.from_conn_string("checkpoints.db")
    
    # Startup: open the SQLite/Postgres connection and compile the graph
    with checkpointer_cm as checkpointer:
        checkpointer.setup()
        agent_app.agent = agent_app.agent_builder.compile(checkpointer=checkpointer)
        yield
    # Shutdown: the 'with' block exits here, closing the connection clearly

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "main-session"

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    rate_limit_key = request.thread_id if request.thread_id else req.client.host

    if not await check_rate_limit(rate_limit_key):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please wait before sending another request."}
        )
    response = await agent_app.chat(request.message, request.thread_id)
    return ChatResponse(response=response)

@app.get("/health")
async def health():
    return {"status": "ok"}