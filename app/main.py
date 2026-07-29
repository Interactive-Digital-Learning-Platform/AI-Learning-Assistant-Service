# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.groq import llm
from app.core.redis import redis_instance
from app.graph.nodes import GraphNodes
from app.graph.workflow import create_assistant_graph
from app.routes.chat_routes import router as chat_router
from app.services.chat_service import ChatService
from app.services.embedding_service import EmbeddingGenerator
from app.services.intent_service import IntentService
from app.services.retrieval_service import RetrievalService
from app.services.session_service import SessionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = EmbeddingGenerator()
    session_service = SessionService()
    retrieval_service = RetrievalService(embedder=embedder)
    chat_service = ChatService(llm)
    intent_service = IntentService(llm)

    nodes = GraphNodes(
        chat_service=chat_service,
        session_service=session_service,
        retrieval_service=retrieval_service
    )

    assistant_graph = create_assistant_graph(
        nodes,
        intent_service
    )

    app.state.embedder = embedder
    app.state.session_service = session_service
    app.state.retrieval_service = retrieval_service
    app.state.chat_service = chat_service
    app.state.intent_service = intent_service
    app.state.assistant_graph = assistant_graph
    
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
