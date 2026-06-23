import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
from app.services.embedding_service import EmbeddingGenerator
from app.core.config import REDIS_URL
from app.routes.chat_routes import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = EmbeddingGenerator()
    app.state.embedder = embedder

    r = aioredis.from_url(REDIS_URL)
    await r.ping()
    await r.aclose()

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(chat_router)

@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def getHealth():
    return {"message": "Learning Assistant Service is active..."}
