import logging
from asyncio import to_thread

import numpy as np
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.schemas.retrieval import SearchResult

logger = logging.getLogger(__name__)


class RerankService:

    def __init__(self, model_name: str = settings.RERANK_MODEL):
        self.model = CrossEncoder(model_name)
        self._warm_up()

    def _warm_up(self) -> None:
        try:
            self.model.predict([("warm up query", "warm up passage")])
        except Exception:
            logger.exception("Reranker warm-up failed; model will load on first use")

    async def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:

        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        raw_scores = await to_thread(self.model.predict, pairs)

        scores = 1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype=float)))

        ranked = sorted(
            zip(candidates, scores), key=lambda pair: pair[1], reverse=True
        )

        return [
            SearchResult(text=c.text, score=round(float(s), 4), metadata=c.metadata)
            for c, s in ranked[:top_k]
        ]
