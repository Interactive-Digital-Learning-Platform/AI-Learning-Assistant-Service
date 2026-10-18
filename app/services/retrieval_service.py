import logging
import re
from asyncio import gather, to_thread, wait_for
from typing import Optional

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.config import settings
from app.schemas.retrieval import RetrievalOutcome, SearchResult
from app.services.embedding_service import EmbeddingGenerator
from app.services.rerank_service import RerankService
from app.services.sparse_embedding_service import SparseEmbeddingGenerator

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\S+")
_MIN_OVERLAP_WORDS = 3
_MAX_OVERLAP_WORDS = 80
_NEIGHBOUR_FETCH_TIMEOUT_SECONDS = 3.0
_STATUS_RANK = {"none": 0, "weak": 1, "confident": 2}

FilterValue = str | int | qmodels.Range
QueryVectors = dict[str, tuple[list[float], qmodels.SparseVector | None]]


def is_transient_retrieval_error(exc: BaseException) -> bool:
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is not None and (
            exc.status_code == 429 or exc.status_code >= 500
        )

    return isinstance(
        exc,
        (TimeoutError, ConnectionError, ResponseHandlingException, httpx.TransportError),
    )


def _shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())

    if len(words) < size:
        return {tuple(words)} if words else set()

    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def stitch_chunks(first: str, second: str) -> str:
    first_words = first.split()
    matches = list(_WORD.finditer(second))
    limit = min(len(first_words), len(matches), _MAX_OVERLAP_WORDS)

    for size in range(limit, _MIN_OVERLAP_WORDS - 1, -1):
        if first_words[-size:] == [m.group(0) for m in matches[:size]]:
            rest = second[matches[size - 1].end():]

            if not rest.strip():
                return first

            separator = "\n\n" if rest.startswith("\n") else " "
            return f"{first}{separator}{rest.strip()}"

    return f"{first}\n\n{second}"


def _condition(key: str, value: FilterValue) -> qmodels.FieldCondition:
    if isinstance(value, qmodels.Range):
        return qmodels.FieldCondition(key=key, range=value)

    return qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))


def scope_levels(grade: int | None, subject: str | None) -> list[dict[str, int | str]]:
    levels: list[dict[str, int | str]] = []

    if grade is not None and subject:
        levels.append({"grade": grade, "subject": subject})
    if grade is not None:
        levels.append({"grade": grade})
    elif subject:
        levels.append({"subject": subject})

    return [*levels, {}]


def _chunk_position(result: SearchResult) -> tuple[str, int] | None:
    metadata = result.metadata
    job_id = metadata.get("job_id")
    index = metadata.get("chunk_index")

    if (
        not isinstance(job_id, str)
        or not isinstance(index, int)
        or not isinstance(metadata.get("section_path"), list)
    ):
        return None

    return job_id, index


def remove_near_duplicates(
    results: list[SearchResult], threshold: float
) -> list[SearchResult]:
    kept: list[SearchResult] = []
    kept_shingles: list[set[tuple[str, ...]]] = []

    for result in results:
        shingles = _shingles(result.text)

        is_duplicate = bool(shingles) and any(
            len(shingles & other) / min(len(shingles), len(other)) >= threshold
            for other in kept_shingles
            if other
        )

        if is_duplicate:
            continue

        kept.append(result)
        kept_shingles.append(shingles)

    return kept


class RetrievalService:

    def __init__(
        self, embedder: EmbeddingGenerator,
        reranker: RerankService | None = None,
        collection: str = settings.QDRANT_COLLECTION,
        top_k: int = settings.TOP_K_CHUNKS,
        threshold: float = settings.SCORE_THRESHOLD,
        rerank_threshold: float = settings.RERANK_SCORE_THRESHOLD,
        rerank_weak_threshold: float = settings.RERANK_WEAK_SCORE_THRESHOLD,
        weak_top_k: int = settings.RETRIEVAL_WEAK_TOP_K,
        dedup_threshold: float = settings.RETRIEVAL_DEDUP_THRESHOLD,
        sparse_embedder: SparseEmbeddingGenerator | None = None,
        prefetch_limit: int = settings.HYBRID_PREFETCH_LIMIT,
        neighbour_window: int = settings.RETRIEVAL_NEIGHBOUR_WINDOW,
        neighbour_top_n: int = settings.RETRIEVAL_NEIGHBOUR_TOP_N,
        neighbour_max_words: int = settings.RETRIEVAL_NEIGHBOUR_MAX_WORDS,
    ):
        self.client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
        self.collection = collection
        self.embedder = embedder
        self.top_k = top_k
        self.threshold = threshold
        self.rerank_threshold = rerank_threshold
        self.rerank_weak_threshold = rerank_weak_threshold
        self.weak_top_k = weak_top_k
        self.dedup_threshold = dedup_threshold
        self.reranker = reranker
        self.sparse_embedder = sparse_embedder
        self.prefetch_limit = prefetch_limit
        self.neighbour_window = neighbour_window
        self.neighbour_top_n = neighbour_top_n
        self.neighbour_max_words = neighbour_max_words
        self.dense_vector_name: str | None = None
        self.hybrid = False
        self._layout_resolved = False

    async def startup(self) -> None:
        try:
            await self.initialize()
        except Exception as exc:
            if not is_transient_retrieval_error(exc):
                raise

            logger.warning(
                f"Could not inspect Qdrant collection '{self.collection}' at startup "
                f"({type(exc).__name__}: {exc}); retrying on first search"
            )

    async def initialize(self) -> None:
        if not await self.client.collection_exists(self.collection):
            logger.warning(
                f"Qdrant collection '{self.collection}' does not exist; "
                f"its vector layout will be resolved on first search"
            )
            return

        params = (await self.client.get_collection(self.collection)).config.params
        vectors = params.vectors
        sparse_vectors = params.sparse_vectors or {}

        if isinstance(vectors, qmodels.VectorParams):
            dense = vectors
            dense_vector_name = None
        elif isinstance(vectors, dict) and settings.DENSE_VECTOR_NAME in vectors:
            dense = vectors[settings.DENSE_VECTOR_NAME]
            dense_vector_name = settings.DENSE_VECTOR_NAME
        else:
            raise ValueError(
                f"Qdrant collection '{self.collection}' has neither a single unnamed vector "
                f"nor a '{settings.DENSE_VECTOR_NAME}' vector"
            )

        if dense.size != settings.EMBEDDING_DIM:
            raise ValueError(
                f"Qdrant collection '{self.collection}' stores {dense.size}-dim vectors "
                f"but EMBEDDING_DIM={settings.EMBEDDING_DIM}"
            )

        has_sparse = settings.SPARSE_VECTOR_NAME in sparse_vectors

        self.dense_vector_name = dense_vector_name
        self.hybrid = (
            has_sparse and self.sparse_embedder is not None and self.reranker is not None
        )
        self._layout_resolved = True

        logger.info(
            f"Retrieval on '{self.collection}': mode={'hybrid' if self.hybrid else 'dense'} "
            f"(sparse_vector={has_sparse}, sparse_encoder={self.sparse_embedder is not None}, "
            f"reranker={self.reranker is not None})"
        )

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: dict[str, FilterValue] | None = None,
        sub_queries: list[str] | None = None,
    ) -> list[SearchResult]:

        rankings, cutoff = await self._rank_queries(query, sub_queries, top_k, filters)
        k = top_k or self.top_k

        results = self._merge(
            [[r for r in ranked if r.score >= cutoff] for ranked in rankings], k
        )

        logger.info(
            f"Retrieval: query='{query[:50]}' queries={len(rankings)} "
            f"→ {sum(len(r) for r in rankings)} ranked, {len(results)} above cutoff "
            f"(cutoff={cutoff}, reranked={self.reranker is not None}, hybrid={self.hybrid})"
        )

        if results:
            top = results[0]
            logger.debug(
                f"  Top result: score={top.score} "
                f"page={top.metadata.get('page_start')} "
                f"file={top.metadata.get('filename')}"
            )

        return results

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, FilterValue] | None = None,
        sub_queries: list[str] | None = None,
    ) -> RetrievalOutcome:

        outcome = await self._outcome(query, top_k, filters, sub_queries)
        outcome.results = await self.expand_neighbours(outcome.results)

        return outcome

    async def retrieve_scoped(
        self,
        query: str,
        grade: int | None = None,
        subject: str | None = None,
        top_k: int | None = None,
        sub_queries: list[str] | None = None,
    ) -> RetrievalOutcome:

        outcomes: list[RetrievalOutcome] = []
        vectors: QueryVectors = {}

        for scope in scope_levels(grade, subject):
            outcome = await self._outcome(
                query, top_k, self._scope_filters(scope), sub_queries, vectors
            )
            outcome.scope = scope
            outcomes.append(outcome)

            if outcome.status == "confident":
                break

        best = max(outcomes, key=lambda outcome: _STATUS_RANK[outcome.status])

        if grade is not None or subject:
            logger.info(
                f"Scoped retrieval: grade={grade} subject={subject} "
                f"→ scope={best.scope or 'all'} status={best.status}"
            )

        best.results = await self.expand_neighbours(best.results)
        return best

    def _scope_filters(self, scope: dict[str, int | str]) -> dict[str, FilterValue]:
        filters: dict[str, FilterValue] = {}

        grade = scope.get("grade")
        if isinstance(grade, int):
            filters["grade"] = qmodels.Range(
                gte=grade - max(settings.RETRIEVAL_GRADE_LOOKBACK, 0), lte=grade
            )

        subject = scope.get("subject")
        if isinstance(subject, str):
            filters["subject"] = subject

        return filters

    async def _outcome(
        self,
        query: str,
        top_k: int | None,
        filters: dict[str, FilterValue] | None,
        sub_queries: list[str] | None,
        vectors: QueryVectors | None = None,
    ) -> RetrievalOutcome:

        rankings, cutoff = await self._rank_queries(
            query, sub_queries, top_k, filters, vectors
        )
        k = top_k or self.top_k
        best_score = max((r.score for ranked in rankings for r in ranked), default=None)

        confident = self._merge(
            [[r for r in ranked if r.score >= cutoff] for ranked in rankings], k
        )

        if confident:
            outcome = RetrievalOutcome(
                status="confident", results=confident, best_score=best_score
            )
        elif (
            self.reranker
            and best_score is not None
            and best_score >= self.rerank_weak_threshold
        ):
            weak = self._merge(
                [[r for r in ranked if r.score >= self.rerank_weak_threshold] for ranked in rankings],
                self.weak_top_k,
            )
            outcome = RetrievalOutcome(status="weak", results=weak, best_score=best_score)
        else:
            outcome = RetrievalOutcome(status="none", best_score=best_score)

        logger.info(
            f"Retrieval: query='{query[:50]}' queries={len(rankings)} "
            f"→ status={outcome.status} results={len(outcome.results)} "
            f"best_score={best_score} filters={sorted(filters or {})} (cutoff={cutoff}, "
            f"weak_cutoff={self.rerank_weak_threshold if self.reranker else None}, "
            f"hybrid={self.hybrid})"
        )

        return outcome

    async def expand_neighbours(self, results: list[SearchResult]) -> list[SearchResult]:
        if self.neighbour_window <= 0 or self.neighbour_top_n <= 0:
            return results

        wanted: dict[str, set[int]] = {}

        for result in results[: self.neighbour_top_n]:
            position = _chunk_position(result)

            if position is None:
                continue

            job_id, index = position
            for offset in range(1, self.neighbour_window + 1):
                wanted.setdefault(job_id, set()).update(
                    i for i in (index - offset, index + offset) if i >= 0
                )

        if not wanted:
            return results

        try:
            neighbours = await wait_for(
                self._fetch_chunks(wanted), timeout=_NEIGHBOUR_FETCH_TIMEOUT_SECONDS
            )
        except (UnexpectedResponse, ResponseHandlingException, httpx.HTTPError, OSError) as exc:
            logger.warning(
                f"Neighbour chunk lookup failed ({type(exc).__name__}: {exc}); "
                f"using the matched chunks alone"
            )
            return results

        covered: set[tuple[str, int]] = set()
        expanded: list[SearchResult] = []
        added = 0

        for rank, result in enumerate(results):
            position = _chunk_position(result)

            if position is None:
                expanded.append(result)
                continue

            if position in covered:
                continue

            if rank >= self.neighbour_top_n:
                covered.add(position)
                expanded.append(result)
                continue

            passage = self._widen(result, position, neighbours, covered)
            added += len(passage) - 1
            covered.update(_chunk_position(chunk) or position for chunk in passage)
            expanded.append(self._join(result, passage))

        if added or len(expanded) < len(results):
            logger.info(
                f"Neighbour expansion: added {added} adjacent chunks, "
                f"{len(results)} → {len(expanded)} excerpts"
            )

        return expanded

    def _widen(
        self,
        hit: SearchResult,
        position: tuple[str, int],
        neighbours: dict[tuple[str, int], SearchResult],
        covered: set[tuple[str, int]],
    ) -> list[SearchResult]:
        job_id, index = position
        section_path = hit.metadata.get("section_path")
        passage = [hit]
        words = len(hit.text.split())
        low = high = index

        for _ in range(self.neighbour_window):
            for target in (high + 1, low - 1):
                key = (job_id, target)
                chunk = neighbours.get(key)

                if (
                    chunk is None
                    or key in covered
                    or chunk.metadata.get("section_path") != section_path
                    or words + len(chunk.text.split()) > self.neighbour_max_words
                ):
                    continue

                words += len(chunk.text.split())

                if target > high:
                    high = target
                    passage.append(chunk)
                else:
                    low = target
                    passage.insert(0, chunk)

        return passage

    @staticmethod
    def _join(hit: SearchResult, passage: list[SearchResult]) -> SearchResult:
        if len(passage) == 1:
            return hit

        text = passage[0].text
        for chunk in passage[1:]:
            text = stitch_chunks(text, chunk.text)

        page_starts = [
            c.metadata["page_start"] for c in passage
            if isinstance(c.metadata.get("page_start"), int)
        ]
        page_ends = [
            c.metadata["page_end"] for c in passage
            if isinstance(c.metadata.get("page_end"), int)
        ]
        metadata = {
            **hit.metadata,
            "chunk_indices": [c.metadata.get("chunk_index") for c in passage],
        }

        if page_starts:
            metadata["page_start"] = min(page_starts)
        if page_ends:
            metadata["page_end"] = max(page_ends)

        return SearchResult(text=text, score=hit.score, metadata=metadata)

    async def _fetch_chunks(
        self, wanted: dict[str, set[int]]
    ) -> dict[tuple[str, int], SearchResult]:
        scroll_filter = qmodels.Filter(
            should=[
                qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="job_id", match=qmodels.MatchValue(value=job_id)
                        ),
                        qmodels.FieldCondition(
                            key="chunk_index", match=qmodels.MatchAny(any=sorted(indices))
                        ),
                    ]
                )
                for job_id, indices in wanted.items()
            ]
        )

        points, _ = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=scroll_filter,
            limit=sum(len(indices) for indices in wanted.values()),
            with_payload=True,
            with_vectors=False,
        )

        chunks: dict[tuple[str, int], SearchResult] = {}

        for point in points:
            payload = point.payload or {}
            chunk = SearchResult(
                text=payload.get("text", ""),
                score=0.0,
                metadata={k: v for k, v in payload.items() if k != "text"},
            )
            position = _chunk_position(chunk)

            if position is not None and chunk.text:
                chunks.setdefault(position, chunk)

        return chunks

    async def _rank_queries(
        self,
        query: str,
        sub_queries: list[str] | None,
        top_k: int | None,
        filters: dict[str, FilterValue] | None,
        vectors: QueryVectors | None = None,
    ) -> tuple[list[list[SearchResult]], float]:
        queries = [q for q in dict.fromkeys(sub_queries or []) if q.strip()]

        if len(queries) < 2:
            queries = [query]

        ranked = await gather(*(self._rank(q, top_k, filters, vectors) for q in queries))

        return [results for results, _ in ranked], ranked[0][1]

    def _merge(self, rankings: list[list[SearchResult]], limit: int) -> list[SearchResult]:
        interleaved: list[SearchResult] = []
        seen_texts: set[str] = set()

        for position in range(max((len(r) for r in rankings), default=0)):
            for ranked in rankings:
                if position < len(ranked) and ranked[position].text not in seen_texts:
                    seen_texts.add(ranked[position].text)
                    interleaved.append(ranked[position])

        if len(rankings) > 1:
            interleaved = remove_near_duplicates(interleaved, self.dedup_threshold)

        return interleaved[:limit]

    async def _query_vectors(
        self, query: str, vectors: QueryVectors | None
    ) -> tuple[list[float], qmodels.SparseVector | None]:
        if vectors is not None and query in vectors:
            return vectors[query]

        sparse_embedder = self.sparse_embedder if self.hybrid else None

        if sparse_embedder is not None:
            dense_vector, sparse_vector = await gather(
                to_thread(self.embedder.embed_single, query),
                to_thread(sparse_embedder.embed_query, query),
            )
        else:
            dense_vector, sparse_vector = await to_thread(self.embedder.embed_single, query), None

        if vectors is not None:
            vectors[query] = (dense_vector, sparse_vector)

        return dense_vector, sparse_vector

    async def _rank(
        self,
        query: str,
        top_k: int | None,
        filters: dict[str, FilterValue] | None,
        vectors: QueryVectors | None = None,
    ) -> tuple[list[SearchResult], float]:

        if not self._layout_resolved:
            await self.initialize()

        k = top_k or self.top_k
        limit = k * settings.RERANK_OVERFETCH

        query_filter = None

        if filters:
            query_filter = qmodels.Filter(
                must=[_condition(key, value) for key, value in filters.items()]
            )

        dense_vector, sparse_vector = await self._query_vectors(query, vectors)

        if sparse_vector is not None:
            prefetch_limit = max(self.prefetch_limit, limit)

            response = await self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    qmodels.Prefetch(
                        query=dense_vector,
                        using=self.dense_vector_name,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                    qmodels.Prefetch(
                        query=sparse_vector,
                        using=settings.SPARSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                ],
                query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            response = await self.client.query_points(
                collection_name=self.collection,
                query=dense_vector,
                using=self.dense_vector_name,
                query_filter=query_filter,
                limit=limit,
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
        if self.reranker:
            ranked = await self.reranker.rerank(query, candidates, top_k=len(candidates))
            cutoff = self.rerank_threshold
        else:
            ranked = candidates
            cutoff = self.threshold

        deduped = remove_near_duplicates(ranked, self.dedup_threshold)

        if len(deduped) < len(ranked):
            logger.info(
                f"Removed {len(ranked) - len(deduped)} near-duplicate chunks "
                f"(threshold={self.dedup_threshold})"
            )

        return deduped[:k], cutoff


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