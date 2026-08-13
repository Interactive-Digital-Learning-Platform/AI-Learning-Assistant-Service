# Retrieval Service Evaluation

Evaluation of `app/services/retrieval_service.py` and the parts of the pipeline it depends on / feeds into (`app/services/embedding_service.py`, `app/graph/nodes.py`, `app/graph/workflow.py`, `app/controllers/chat_controllers.py`, `app/services/chat_service.py`, `app/schemas/conversation.py`, `app/models/conversation_model.py`). Goal: find why the service can return inaccurate or inefficient context for a user query.

> **Revision note:** the original version of this document proposed scoping retrieval by having the client supply a `filename`, and filtering retrieval by `user_id`. Both were based on incorrect assumptions about the system. Corrected architecture:
> - The knowledge base is a single, shared, **admin-curated** corpus. End users never see or choose document filenames — they just ask questions. Any fix must be entirely server-side and invisible to the client.
> - There is no per-user document ownership — documents aren't scoped to the requesting user, so a `user_id` retrieval filter doesn't apply to this system's data model.
>
> Issue 1 has been rewritten around a client-agnostic approach. The former "user isolation" issue has been withdrawn (see the note at the end of this document). All other issues are unchanged.

Issues are ordered by severity/impact.

---

## 1. (Critical) Nothing keeps retrieval focused on the relevant document(s) — cross-document contamination in a shared knowledge base

`RetrievalService.search()` always queries the entire `QDRANT_COLLECTION` with no way to bias results toward the document(s) actually relevant to the conversation:

```python
# app/services/retrieval_service.py
response = await self.client.query_points(
    collection_name=self.collection,
    query=query_vector,
    limit=top_k or self.top_k,
    query_filter=query_filter,   # always None in practice — filename is never passed by any caller
    score_threshold=self.threshold,
    with_payload=True,
)
```

**Impact:** because every query searches the full corpus, a chunk from an unrelated document can still outscore the genuinely relevant chunk purely on embedding-similarity noise (e.g. shared vocabulary, similar phrasing, generic terminology that appears in multiple source documents). This gets worse as the knowledge base grows and covers more distinct subjects/documents. Bi-encoder cosine similarity (what `query_points` ranks on) is known to be a fairly blunt relevance signal — it's fast, but it isn't precise enough on its own to reliably keep unrelated documents out of the top-k once the collection holds more than a handful of source files.

Given the constraint that the client/end user must stay unaware of the underlying document structure, the fix has to work purely from the query itself and from evidence gathered server-side — not from anything the client supplies.

### Fix — two complementary, fully server-side mechanisms

**A. Cross-encoder re-ranking over an over-fetched candidate set (primary fix).**
Over-fetch a wider candidate set from Qdrant with the cheap bi-encoder search, then re-score each candidate against the query with a cross-encoder, which jointly attends over (query, chunk) and is much better at distinguishing "genuinely relevant" from "superficially similar." This directly suppresses cross-document noise without needing to know which document is "correct" ahead of time.

```python
# app/core/config.py
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_OVERFETCH: int = 4  # candidates pulled from Qdrant = top_k * RERANK_OVERFETCH
```

```python
# app/services/rerank_service.py  (new)
import logging
from asyncio import to_thread

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.services.retrieval_service import SearchResult

logger = logging.getLogger(__name__)


class RerankService:
    def __init__(self, model_name: str = settings.RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    async def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = await to_thread(self.model.predict, pairs)

        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

        return [
            SearchResult(text=c.text, score=round(float(s), 4), metadata=c.metadata)
            for c, s in ranked[:top_k]
        ]
```

```python
# app/services/retrieval_service.py
class RetrievalService:
    def __init__(self, embedder: EmbeddingGenerator, reranker: RerankService):
        self.client = AsyncQdrantClient(url=settings.QDRANT_URL)
        self.collection = settings.QDRANT_COLLECTION
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = settings.TOP_K_CHUNKS
        self.threshold = settings.SCORE_THRESHOLD

    async def search(self, query: str, top_k: Optional[int] = None) -> list[SearchResult]:
        query_vector = await to_thread(self.embedder.embed_single, query)
        k = top_k or self.top_k

        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=k * settings.RERANK_OVERFETCH,
            with_payload=True,
        )

        candidates = [
            SearchResult(
                text=(hit.payload or {}).get("text", ""),
                score=round(hit.score or 0.0, 4),
                metadata={k: v for k, v in (hit.payload or {}).items() if k != "text"},
            )
            for hit in response.points
        ]

        reranked = await self.reranker.rerank(query, candidates, top_k=k)

        return [r for r in reranked if r.score >= self.threshold]
```

`self.threshold` (`SCORE_THRESHOLD`) now needs to be re-tuned against cross-encoder scores rather than cosine similarity — they're on a different scale, so pull a fresh baseline from real query logs before picking a cutoff (see also Issue 5 for the threshold-fallback behavior, which still applies here).

Wire `RerankService` into `app/main.py`'s `lifespan` the same way the other services are constructed, and inject it into `RetrievalService`.

**B. (Optional enhancement) Implicit, server-inferred per-conversation document affinity.**
For follow-up turns in the same conversation, it's reasonable to lean toward whatever document(s) already proved relevant earlier in that conversation — but this has to be *inferred from retrieval evidence*, never supplied by the client, and applied as a *soft bias*, not a hard filter, so the conversation can still follow the user into a different document if they change topic.

```python
# app/models/conversation_model.py
class Conversation(Base):
    __tablename__ = "conversations"
    ...
    # Server-inferred only — never set from client input. Updated after each
    # RAG turn based on which filename dominates the reranked top results.
    primary_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
```

```python
# app/graph/nodes.py — retrieve_docs_node, after reranking
if chunks:
    from collections import Counter
    top_filenames = Counter(c.metadata.get("filename") for c in chunks[:3])
    inferred_filename, _ = top_filenames.most_common(1)[0]
    # persisted by the caller/controller after this node runs, not inside the node
```

The affinity can then be fed back into `RerankService.rerank` as a small additive score bonus for candidates whose `metadata["filename"]` matches the conversation's current `primary_filename`, keeping multi-turn answers coherent without ever hard-filtering out a legitimately different document if the user's question drifts.

This is an enhancement on top of (A), not a replacement for it — (A) is what actually keeps unrelated documents out of the context; (B) just makes follow-up turns more consistent once a topic is established.

---

## 2. (Critical) The non-streaming send path is broken — retrieval work is thrown away and the endpoint always fails

`ChatService.get_response` requires a `mode` argument with no default:

```python
# app/services/chat_service.py
async def get_response(
    self, message: str, history: list, context: list[SearchResult], mode: str
):
```

But `send_message` calls it with only 3 positional arguments:

```python
# app/controllers/chat_controllers.py — send_message
chunks = await retrieval.search(query=request.message)
history = await session_service.get_langchain_history(conversation_id)
response = await chat_service.get_response(request.message, history, chunks)
```

**Impact:** `POST /conversations/{id}/messages` (the non-streaming endpoint) raises `TypeError: get_response() missing 1 required positional argument: 'mode'` on every single call. It's caught by the generic `except Exception` and turned into a 500 ("Failed to generate the response"). This means the embedding + Qdrant search that just ran (`retrieval.search(...)`) is always wasted — real work is done, then discarded because the response can never be generated. This endpoint currently cannot deliver any context to a user at all.

### Fix

At minimum, supply `mode`. Since this path has no intent classification (see Issue 6), the simplest correct fix is to reuse `IntentService` the same way the graph does, rather than assuming RAG mode unconditionally:

```python
# app/controllers/chat_controllers.py — send_message
intent_service = api_request.app.state.intent_service
history = await session_service.get_langchain_history(conversation_id)

classification = await intent_service.llm.with_structured_output(
    QueryClassification, method="json_mode"
).ainvoke({"user_message": request.message, "history": history})

chunks = (
    await retrieval.search(query=request.message)
    if classification.intent == "rag"
    else []
)

response = await chat_service.get_response(
    request.message, history, chunks, mode=classification.intent
)
```

(Longer term, this endpoint should just reuse the LangGraph workflow instead of re-implementing a parallel, drifting pipeline — see Issue 6.)

---

## 3. (High) The `retrieve_docs` node's `RetryPolicy` never actually retries anything

The graph wires a retry policy on the retrieval node:

```python
# app/graph/workflow.py
workflow.add_node(
    "retrieve_docs",
    nodes.retrieve_docs_node,
    retry_policy=RetryPolicy(max_attempts=2),
)
```

LangGraph's `RetryPolicy` only retries a node if the node function *raises*. But `retrieve_docs_node` catches everything internally and swallows it:

```python
# app/graph/nodes.py — retrieve_docs_node
try:
    chunks = await wait_for(
        self.retrieval_service.search(query=query),
        timeout=15.0
    )
except Exception:
    logger.exception(
        "Document retrieval failed; continuing without context"
    )
    chunks = []
```

**Impact:** a transient Qdrant blip (connection reset, momentary timeout, brief network hiccup) never gets retried — it's immediately converted into "no context found," and the RAG answer silently degrades to an ungrounded, general-knowledge response even though the intent was correctly classified as `rag`. The retry policy gives a false sense of resilience; it's dead configuration.

### Fix

Let retryable errors propagate so `RetryPolicy` can do its job, and only degrade to empty context once retries are exhausted. `chat_stream_handler` already has a top-level `except Exception` that turns a hard graph failure into a graceful SSE `error` event, so it's safe to let this node fail after retries are exhausted rather than pre-emptively swallowing the very first error:

```python
# app/graph/nodes.py — retrieve_docs_node
async def retrieve_docs_node(self, state: AgentState) -> dict[str, Any]:
    query = state["rewritten_query"] or state["user_message"]

    chunks = await wait_for(
        self.retrieval_service.search(query=query),
        timeout=15.0,
    )

    sources = [...]
    return {
        "sources": sources,
        "context": chunks,
        "retrieved_chunks": chunks,
        "rag_used": True,
    }
```

If graceful degradation to "answer without context" is still desired (rather than surfacing an error to the user) after the 2 retry attempts are exhausted, wrap the *call to the compiled graph* (not the node itself) with a fallback, or give `RetryPolicy` a `retry_on` predicate that only retries genuinely transient errors (timeouts/connection errors) and re-raises the rest — but the node itself must stop unconditionally catching everything, or the policy is pure decoration.

---

## 4. (Medium) Query embedding prefix is hardcoded for one embedding model, but the model is configurable

```python
# app/services/embedding_service.py
def embed_single(self, text: str) -> List[float]:
    prefixed = f"search_query: {text}"
    prefixed = self._truncate(prefixed, label="query")

    embedding = self.model.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.tolist()
```

`"search_query: "` is the instruction prefix required specifically by `nomic-ai/nomic-embed-text-v1.5` (the default in `app/core/config.py`). But `EMBEDDING_MODEL` is a config value, and `.env.example` itself sets a *different* model:

```
# .env.example
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

BGE models use a different (and differently-worded) query instruction convention, and most sentence-transformers models use no instruction prefix at all. Feeding `"search_query: "` into a model that wasn't trained with that literal string as an instruction token doesn't error — it silently embeds as noise, degrading the quality/accuracy of every query vector produced (while document vectors, embedded elsewhere at ingestion time, aren't guaranteed to use a matching convention either).

Separately, the defaults are inconsistent in a way that will hard-fail rather than silently degrade: `config.py` defaults `EMBEDDING_DIM=768` (matches `nomic-embed-text-v1.5`), but `.env.example`'s `BAAI/bge-small-en-v1.5` outputs 384-dim vectors, which trips the dimension guard in `EmbeddingGenerator.__init__` and prevents the service from starting if someone follows the example env file as-is.

### Fix

Make the prefix a config value tied to the embedding model instead of a hardcoded literal, and fix the example env file to match a self-consistent model/dimension pair:

```python
# app/core/config.py
EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_QUERY_PREFIX: str = "search_query: "
EMBEDDING_DIM: int = 768
```

```python
# app/services/embedding_service.py
def embed_single(self, text: str) -> List[float]:
    prefixed = f"{settings.EMBEDDING_QUERY_PREFIX}{text}"
    prefixed = self._truncate(prefixed, label="query")
    ...
```

```
# .env.example — use a model/dim pair that's actually consistent, e.g.:
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
EMBEDDING_DIM=768
```

(If `BAAI/bge-small-en-v1.5` is genuinely the intended default, flip it the other way: change `config.py`'s defaults to match it, set `EMBEDDING_QUERY_PREFIX` appropriately for BGE, and drop the nomic-specific string.)

---

## 5. (Medium) A single hard `score_threshold` silently zeroes out context with no fallback

```python
# app/services/retrieval_service.py
response = await self.client.query_points(
    collection_name=self.collection,
    query=query_vector,
    limit=top_k or self.top_k,
    query_filter=query_filter,
    score_threshold=self.threshold,
    with_payload=True,
)
```

`self.threshold` (`SCORE_THRESHOLD`) is a single static value applied to every query verbatim as a hard cutoff in Qdrant itself — anything below it is never returned, not even as a low-confidence candidate. There's also a config/example drift here too: `config.py` defaults `SCORE_THRESHOLD=0.6`, `.env.example` ships `0.4` — a meaningful difference in how aggressively results get filtered out.

**Impact:** for borderline queries (paraphrased terminology, cross-lingual phrasing, queries about content near the edge of what's indexed), it's easy for every candidate chunk to fall just under the threshold. The result is an empty `chunks` list even though the intent was classified as `rag` — the user gets an answer that's silently ungrounded in their documents (falls back to the LLM's general knowledge per the RAG system prompt's own instructions), with no signal that retrieval effectively failed.

### Fix

Don't apply the threshold as a hard Qdrant-side cutoff with no fallback. Retrieve unfiltered (or post-rerank, per Issue 1's fix), then apply the threshold in application code with a fallback to the best available result(s) so a `rag`-classified query is never silently answered with zero context:

```python
# app/services/retrieval_service.py
above_threshold = [r for r in results if r.score >= self.threshold]

if not above_threshold and results:
    logger.warning(
        f"No chunks cleared threshold={self.threshold} for query='{query[:50]}'; "
        f"falling back to top result (score={results[0].score})"
    )
    return results[:1]

return above_threshold
```

This keeps the threshold as the accuracy bar for normal cases, but avoids the current all-or-nothing behavior where a query 0.01 short of the bar gets exactly the same (empty) context as a completely unrelated query.

---

## 6. (Low/Medium) The non-streaming path always retrieves, with no intent gating — wasted embedding + Qdrant calls

```python
# app/controllers/chat_controllers.py — send_message
chunks = await retrieval.search(query=request.message)
```

Unlike the LangGraph path (`classify_intent` → only `rewrite_query`/`retrieve_docs` on the `rag` branch), `send_message` runs an embedding + Qdrant query for *every* message, including greetings, thanks, and general questions that have nothing to do with the knowledge base.

**Impact:** unnecessary latency and Qdrant load on every non-streaming request, and (combined with Issue 2's missing `mode`) irrelevant `chunks` would otherwise get passed into the RAG prompt path even for messages that were never meant to be document-grounded.

### Fix

Covered by the Issue 2 fix above — gate retrieval behind the same intent classification the graph uses, so retrieval only runs when `intent == "rag"`. Longer term, consider having `send_message` invoke the compiled `assistant_graph` directly (non-streamed, via `.ainvoke`) instead of re-implementing a second, divergent version of the same pipeline — this class of bug (missing `mode`, missing intent gate, missing query rewrite) exists precisely because the two paths are maintained separately and have already drifted apart.

---

## Withdrawn finding: per-user document isolation

The original evaluation flagged the absence of a `user_id` filter in the Qdrant query as a data-isolation gap. That assumed documents were owned by individual end users. In this system, the knowledge base is a single shared corpus curated by admins — end users query against it but don't own or scope documents themselves — so a `user_id` retrieval filter isn't applicable to the actual data model, and no fix is proposed for it.
