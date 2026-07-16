from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
import os
import app as agent_app

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: open the SQLite connection and compile the graph
    with PostgresSaver.from_conn_string(os.getenv("DATABASE_URL")) as checkpointer:
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
async def chat_endpoint(request: ChatRequest):
    response = await agent_app.chat(request.message, request.thread_id)
    return ChatResponse(response=response)

@app.get("/health")
async def health():
    return {"status": "ok"}