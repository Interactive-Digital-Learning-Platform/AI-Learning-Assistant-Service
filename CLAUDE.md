# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this service is

A FastAPI microservice that provides a RAG-based AI study assistant for an interactive digital learning platform. It classifies incoming user messages, optionally retrieves relevant document chunks from Qdrant, and streams an LLM (Groq) response back over SSE, while persisting conversation history to Postgres and caching recent turns in Redis.

## Commands

This project uses `uv` for dependency and environment management (Python >=3.13).

```bash
# Install dependencies
uv sync

# Run the dev server
uv run uvicorn app.main:app --reload --port 8005

# Lint (ruff is the only configured dev tool)
uv run ruff check .
uv run ruff format .

# Type checking (pyright, standard mode — see pyrightconfig.json)
uv run pyright

# Database migrations (Alembic)
uv run alembic revision --autogenerate -m "description"
uv run alembic upgrade head
uv run alembic downgrade -1
```

There is no test suite in this repository currently.

Required external services (see `.env.example` for the full variable list): Postgres (`DB_URL`), Qdrant (`QDRANT_URL`), Redis (`REDIS_URL`), and a Groq API key (`GROQ_API_KEY`). All settings are loaded via `app/core/config.py` (`pydantic-settings`, reads `.env`) and required fields will raise on startup if missing.

## Architecture

### Request flow

Two parallel entry points exist for sending a message in a conversation (`app/routes/chat_routes.py` → `app/controllers/chat_controllers.py`):

- `POST /conversations/{id}/messages/stream` — SSE streaming path, drives the **LangGraph** workflow (`app/graph/`). This is the primary/current path.
- `POST /conversations/{id}/messages` — non-streaming path, calls `ChatService` directly without going through the graph (retrieval → response in a straight line, no intent classification). Kept for simple/synchronous use cases. Applies the same `TranslatorService` pre/post the `ChatService` call as the graph does at its boundaries.

### LangGraph workflow (`app/graph/workflow.py`, `app/graph/nodes.py`)

The streaming path runs a compiled `StateGraph` over `AgentState` (`app/schemas/agent_state.py`):

```
START → translate_input → load_memory → check_attachments → classify_intent
        ┬─(general)→ generate_response → translate_output → END
        ├─(rag)→ retrieve_docs ─(KB none, no attachments)→ web_search ┐
        │                      └────────────────────────────────────→ generate_response → translate_output → END
        ├─(web_search)→ web_search → generate_response
        └─(generate_pdf)→ retrieve_docs → generate_document → generate_response
```

- `translate_input` / `translate_output` (`GraphNodes.ai_translator_node`, `app/services/translator_service.py`): one handler wired at both boundaries. Inbound translates a Sinhala query to English (the graph is English-only) and resolves `reply_language`; outbound translates the English answer back to Sinhala into `translated_response` (the English `response` is what gets persisted). Language comes from `ChatRequest.language` ("English" default / "Sinhala"); a Sinhala-script query auto-triggers translation even when the selector says English (`app/utils/language.py::resolve_translation_plan`). Disabled by `TRANSLATION_ENABLED=false`.
- `load_memory` (`GraphNodes`): pulls prior turns from Redis via `SessionService` (Postgres on a cache miss), then applies `budget_history` (`app/utils/history.py`): recent messages in full, older ones shortened, within `HISTORY_MAX_CHARS`. Every later node sees the budgeted history.
- `check_attachments` (`GraphNodes`): loads attachment status for the conversation and flags inline/groundable/pending attachments, plus `message_has_attachments` (a usable file linked to the current message) and `attachment_names`. Only `message_has_attachments` forces `rag`; otherwise the classifier decides, told which files exist. Inline attachment text is extracted once by the upload job and stored in `Attachment.extracted_text`; uploaded-file excerpts get their own share of the context budget (`ATTACHMENT_CONTEXT_CHARS` within `MAX_CONTEXT_CHARS`).
- `classify_intent` (`IntentService`): one structured-output call (`QueryAnalysis`) returns `intent`, `standalone_query`, up to 3 `sub_queries` and the question's `subject` (one of `KB_SUBJECTS`, or null), and uses `Command(goto=...)` to branch the graph directly (no separate conditional-edge function). With no history the original message is the query (except `generate_pdf`, whose query is the document topic); a file attached to the first message skips the call entirely; failures or a timeout fall back to `rag` with the original message. `web_search` is reserved for current/real-time information — the KB-miss fallback in `retrieve_docs` covers everything else.
- `retrieve_docs`: search against Qdrant via `RetrievalService` (each sub-query ranked separately, results interleaved and de-duplicated, then the top `RETRIEVAL_NEIGHBOUR_TOP_N` hits widened with their adjacent chunks from the same section, using `job_id` + document-global `chunk_index`). `retrieve_scoped` first searches books tagged with the student's grade (`ChatRequest.grade`, plus `RETRIEVAL_GRADE_LOOKBACK` grades below) and the classified subject, then the grade alone, then everything, and stops at the first `confident` result (`retrieval_scope` records which one was used; `RETRIEVAL_SCOPE_ENABLED=false` turns it off). It runs with a 15s timeout and a retry policy; retrieval failures degrade to an `error` status (H1) so a Qdrant outage doesn't hard-fail the chat. For `generate_pdf` it fetches `DOCUMENT_TOP_K_CHUNKS` and routes to `generate_document`, never to the web.
- `generate_document`: writes the study-document Markdown from the retrieved material with the same tag-and-cite rules, strips the tags from the PDF body, sends it to the MCP `generate_pdf` tool, and sets `sources` to the cited excerpts; `generate_response` then writes only the short chat reply.
- `generate_response`: builds the final answer with `ChatService`, selecting the prompt (`app/prompts/chat_prompts.py`) from `state["intent"]`. For `rag`/`web_search` the rules live in the system message and the tagged material (`[S1]`, `[S2]`, …) plus the question form the final human turn; `ChatService.finalize_citations` (`app/utils/citations.py`) strips the tags (or renumbers them when `CITATION_MARKERS_VISIBLE`) and sets `sources` to the cited excerpts only.

The SSE handler (`app/utils/message.py::chat_stream_handler`) drives the graph with `astream_events(..., version="v3")`, streams tokens emitted specifically by the `generate_response` node (passed through `CitationStreamFilter`, so the streamed text equals the stored content), then reads the final graph state via `stream.output()` to get sources/intent/etc. for the persisted assistant message and the terminal `done` SSE event. A 90s overall timeout wraps the streaming loop. When the reply will be Sinhala, live token streaming is suppressed and the fully translated answer is emitted as one `token` event before `done` (which also carries `translation_failed`). `Message.content` and the Redis history cache are always English (the only text any pipeline reads); the Sinhala display text is persisted next to it in `Message.translated_content` / `is_translated` and returned by the history endpoint so past Sinhala conversations render in Sinhala.

### Persistence model

- **Postgres** (SQLAlchemy async ORM, `app/models/`): `Conversation` 1—N `Message`, source of truth for chat history. `Conversation.title` is written by `TitleService` (`app/services/title_service.py`) on both message paths while it is still empty: a short `utility_llm` call over the first exchange, translated to Sinhala when the reply is, falling back to the truncated message on failure/timeout or when `TITLE_GENERATION_ENABLED=false`; a greeting-only turn leaves it null so the next turn retries. The title is returned as `conversation_title` in the `done` SSE event and the non-streaming `MessageResponse`. Messages store `message_metadata` (JSONB) containing `sources`, `rag_used`, `intent`, `rewritten_query`, `retrieval_status`, `retrieval_scope`. `content` is always English; `translated_content` + `is_translated` hold the optional Sinhala display text (see translation boundary above).
- **Shared chats** (`SharedChat`, `app/models/shared_chat_model.py`; `app/routes/share_routes.py`): read-only snapshots behind a random token. `POST /conversations/{id}/share` (owner only, by `user_id`) creates the link or refreshes its snapshot while keeping the token; `DELETE` on the same path revokes it; `GET /shared/{token}` is public. Snapshots deliberately hold no conversation/message/attachment ids (the live read endpoints are keyed by those and have no ownership checks), and the row cascades away when its conversation is deleted.
- **Redis** (`SessionService`, `app/services/session_service.py`): short-lived (`HISTORY_TTL`) cache of the last `MAX_HISTORY_MESSAGES * 2` messages per conversation, keyed as `{KEY_PREFIX}:{conversation_id}:history`, used to reconstruct LangChain message history for the graph without hitting Postgres on every turn. `append_message` uses optimistic locking (`WATCH`/`MULTI`) with a retry loop. When a Redis cache is cold, `get_messages` (paginated history endpoint) re-warms it from Postgres.
- **Qdrant** (`RetrievalService`): search over a single collection (`QDRANT_COLLECTION`), returning `SearchResult(text, score, metadata)`. The vector layout is read at startup: on a hybrid collection (named `dense` + sparse `bm25`, e.g. `pdf_knowledge_base_v2`) with the reranker attached, dense and BM25 (fastembed `Qdrant/bm25`) prefetches are fused with RRF before reranking; otherwise it is a plain dense search, score-thresholded by `SCORE_THRESHOLD`. Chunks carry `section_path` / `context_header` / `chapter` from ingestion, and `grade` / `subject` from the tags set at upload (editable later without re-indexing); the reranker scores `context_header + text`. Embeddings are produced by `EmbeddingGenerator` (`sentence-transformers`, prefixes queries with `"search_query: "`), and its output dimension is validated against `EMBEDDING_DIM` at startup — a mismatched embedding model will fail fast rather than silently corrupt search.

### Service wiring

All services (`EmbeddingGenerator`, `SessionService`, `RetrievalService`, `ChatService`, `IntentService`, `TranslatorService`, `TitleService`) and the compiled graph are constructed once in `app/main.py`'s `lifespan` and stashed on `app.state`; controllers/handlers pull them from `Request.app.state` rather than via FastAPI `Depends`. Only the DB session is wired through `Depends(get_async_session)` (`app/core/database.py`).

### Adding a new graph node/branch

Follow the existing pattern: add the node method to `GraphNodes` (or a new service class if it owns distinct LLM/business logic), register it in `create_assistant_graph` (`app/graph/workflow.py`), and extend `AgentState` (`app/schemas/agent_state.py`) with any new state keys. Branching nodes return `Command(update=..., goto=...)` rather than using `add_conditional_edges`.
