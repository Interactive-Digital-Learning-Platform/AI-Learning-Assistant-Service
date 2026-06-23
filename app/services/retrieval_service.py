import logging
from dataclasses import dataclass
from typing import Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, MatchValue, Filter
from app.core.config import QDRANT_URL, QDRANT_COLLECTION, TOP_K_CHUNKS, SCORE_THRESHOLD
from app.services.embedding_service import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict


class RetrievalService:

    def __init__(self, embedder: EmbeddingGenerator):
        self.client = AsyncQdrantClient(url=QDRANT_URL)
        self.collection = QDRANT_COLLECTION
        self.embedder = embedder
        self.top_k = TOP_K_CHUNKS
        self.threshold = SCORE_THRESHOLD

    async def search(
        self,
        query: str,
        filename: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:

        import asyncio
        query_vector = await asyncio.to_thread(self.embedder.embed_single, query)

        must_conditions = []
        if filename:
            must_conditions.append(
                FieldCondition(key="filename", match=MatchValue(value=filename))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        hits = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k or self.top_k,
            query_filter=query_filter,
            score_threshold=self.threshold,
            with_payload=True,
        )

        results = []
        for hit in hits:
            record = hit[0] if isinstance(hit, tuple) and hit else hit
            payload = getattr(record, "payload", None) or {}
            score = getattr(record, "score", None)

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
