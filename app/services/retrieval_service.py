import logging
from dataclasses import dataclass
from typing import Optional
from asyncio import to_thread
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, MatchValue, Filter
from app.core.config import settings
from app.services.embedding_service import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict


class RetrievalService:

    def __init__(self, embedder: EmbeddingGenerator):
        self.client = AsyncQdrantClient(url=settings.QDRANT_URL)
        self.collection = settings.QDRANT_COLLECTION
        self.embedder = embedder
        self.top_k = settings.TOP_K_CHUNKS
        self.threshold = settings.SCORE_THRESHOLD

    async def search(
        self,
        query: str,
        filename: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:

        query_vector = await to_thread(self.embedder.embed_single, query)

        must_conditions = []
        if filename:
            must_conditions.append(
                FieldCondition(key="filename", match=MatchValue(value=filename))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k or self.top_k,
            query_filter=query_filter,
            score_threshold=self.threshold,
            with_payload=True,
        )

        results = []
        for hit in response.points:
            
            payload = hit.payload or {}
            score = hit.score

            results.append(
                SearchResult(
                    text=payload.get("text", ""),
                    score=round(score or 0.0, 4),
                    metadata={k: v for k, v in payload.items() if k != "text"},
                )
            )

        logger.info(
            f"Retrieval: query='{query[:50]}' "
            f"→ {len(results)} results "
            f"(threshold={self.threshold})"
        )

        if results:
            logger.debug(
                f"  Top result: score={results[0].score} "
                f"page={results[0].metadata.get('page_start')} "
                f"file={results[0].metadata.get('filename')}"
            )

        return results
