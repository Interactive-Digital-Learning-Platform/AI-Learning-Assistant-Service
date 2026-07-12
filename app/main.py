# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.embedding_service import EmbeddingGenerator
from app.services.session_service import SessionService
from app.services.retrieval_service import RetrievalService
from app.services.chat_service import ChatService
from app.routes.chat_routes import router as chat_router
from app.core.redis import redis_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embedder = EmbeddingGenerator()
    app.state.session_service = SessionService()
    app.state.retrieval_service = RetrievalService(embedder=app.state.embedder)
    app.state.chat_service = ChatService()
    redis_instance.ping()

    yield
    await redis_instance.aclose()


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
