# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv

from app.core.config import settings
from app.services.attachment_ingestion_service import AttachmentIngestionService
from app.services.inline_attachment_service import InlineAttachmentService

load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.exceptions import RedisError

from app.core.arq import create_arq_pool
from app.core.groq import llm
from app.core.redis import redis_instance
from app.graph.nodes import GraphNodes
from app.graph.workflow import create_assistant_graph
from app.routes.attachment_routes import router as attachment_router
from app.routes.chat_routes import router as chat_router
from app.routes.webhook_routes import router as webhook_router
from app.services.chat_service import ChatService
from app.services.embedding_service import EmbeddingGenerator
from app.services.intent_service import IntentService
from app.services.rerank_service import RerankService
from app.services.retrieval_service import RetrievalService
from app.services.session_service import SessionService
from app.services.storage_service import StorageService


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = EmbeddingGenerator()
    session_service = SessionService()
    rerank_service = RerankService()
    attachment_ingestion_service = AttachmentIngestionService()
    retrieval_service = RetrievalService(embedder=embedder, reranker=rerank_service)
    attachment_retrieval_service = RetrievalService(
        embedder=embedder,
        reranker=None,
        collection=settings.ATTACHMENT_QDRANT_COLLECTION,
        top_k=settings.ATTACHMENT_TOP_K_CHUNKS,
        threshold=settings.ATTACHMENT_SCORE_THRESHOLD
    )
    chat_service = ChatService(llm)
    intent_service = IntentService(llm)
    storage_service = StorageService()
    inline_attachment_service = InlineAttachmentService(storage_service, attachment_ingestion_service)
    arq_pool = await create_arq_pool()

    nodes = GraphNodes(
        chat_service=chat_service,
        session_service=session_service,
        retrieval_service=retrieval_service,
        attachment_retrieval_service=attachment_retrieval_service,
        inline_attachment_service=inline_attachment_service
    )

    assistant_graph = create_assistant_graph(
        nodes,
        intent_service
    )

    app.state.embedder = embedder
    app.state.session_service = session_service
    app.state.retrieval_service = retrieval_service
    app.state.attachment_retrieval_service = attachment_retrieval_service
    app.state.chat_service = chat_service
    app.state.intent_service = intent_service
    app.state.storage_service = storage_service
    app.state.assistant_graph = assistant_graph
    app.state.arq_pool = arq_pool
    
    try:
        await redis_instance.ping()
    except RedisError as e:
        raise RuntimeError("Failed to connect to Redis") from e

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
app.include_router(attachment_router)
app.include_router(webhook_router)

@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def getHealth():
    return {"message": "Learning Assistant Service is active..."}
