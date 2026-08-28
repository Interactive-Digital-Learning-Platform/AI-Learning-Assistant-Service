import logging
from asyncio import to_thread
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

from app.core.config import settings
from app.schemas.retrieval import SearchResult
from app.services.embedding_service import EmbeddingGenerator
from app.services.rerank_service import RerankService

logger = logging.getLogger(__name__)



class RetrievalService:

    def __init__(
        self, embedder: EmbeddingGenerator, 
        reranker: RerankService | None = None,
        collection: str = settings.QDRANT_COLLECTION,
        top_k: int = settings.TOP_K_CHUNKS,
        threshold: float = settings.SCORE_THRESHOLD
    ):
        self.client = AsyncQdrantClient(url=settings.QDRANT_URL)
        self.collection = collection
        self.embedder = embedder
        self.top_k = top_k
        self.threshold = threshold
        self.reranker = reranker

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: dict[str, str] | None = None
    ) -> list[SearchResult]:

        query_vector = await to_thread(self.embedder.embed_single, query)
        k = top_k or self.top_k

        query_filter = None

        if filters:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )

        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=k * settings.RERANK_OVERFETCH,
            with_payload=True,
        )

        candidates = []
        for hit in response.points:
            
            payload = hit.payload or {}
            score = hit.score

            candidates.append(
                SearchResult(
                    text=payload.get("text", ""),
                    score=round(score or 0.0, 4),
                    metadata={k: v for k, v in payload.items() if k != "text"},
                )
            )
        reranked = await self.reranker.rerank(query, candidates, top_k=k) if self.reranker else candidates
        
        logger.info(
            f"Retrieval: query='{query[:50]}' "
            f"→ {len(candidates)} results "
            f"(threshold={self.threshold})"
        )

        if candidates:
            logger.debug(
                f"  Top result: score={candidates[0].score} "
                f"page={candidates[0].metadata.get('page_start')} "
                f"file={candidates[0].metadata.get('filename')}"
            )

        return [r for r in reranked if r.score >= self.threshold]


    async def delete_by_filter(self, filters: dict[str, str]) -> None:

        await self.client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
                        for key, value in filters.items()
                    ]
                )
            )
        )