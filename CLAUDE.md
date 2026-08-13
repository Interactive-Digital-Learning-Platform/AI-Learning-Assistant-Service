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
- `POST /conversations/{id}/messages` — non-streaming path, calls `ChatService` directly without going through the graph (retrieval → response in a straight line, no intent classification). Kept for simple/synchronous use cases.

### LangGraph workflow (`app/graph/workflow.py`, `app/graph/nodes.py`)

The streaming path runs a compiled `StateGraph` over `AgentState` (`app/schemas/agent_state.py`):

```
START → load_memory → classify_intent →┬─(general)→ generate_response → END
                                        └─(rag)→ rewrite_query → retrieve_docs → generate_response → END
```

- `load_memory` (`GraphNodes`): pulls prior turns from Redis via `SessionService`.
- `classify_intent` (`IntentService`): structured-output LLM call classifying the query as `general` or `rag`, and uses `Command(goto=...)` to branch the graph directly (no separate conditional-edge function).
- `rewrite_query` (`rag` path only): rewrites the latest message into a standalone search query using history.
- `retrieve_docs`: vector search against Qdrant via `RetrievalService`, with a 15s timeout and a retry policy; retrieval failures are swallowed (falls back to empty context) so a Qdrant outage doesn't hard-fail the chat.
- `generate_response`: builds the final answer with `ChatService`, selecting the RAG or general system prompt (`app/prompts/chat_prompts.py`) based on `state["intent"]`.

The SSE handler (`app/utils/message.py::chat_stream_handler`) drives the graph with `astream_events(..., version="v3")`, streams tokens emitted specifically by the `generate_response` node, then reads the final graph state via `stream.output()` to get sources/intent/etc. for the persisted assistant message and the terminal `done` SSE event. A 90s overall timeout wraps the streaming loop.

### Persistence model

- **Postgres** (SQLAlchemy async ORM, `app/models/`): `Conversation` 1—N `Message`, source of truth for chat history. Messages store `message_metadata` (JSONB) containing `sources`, `rag_used`, `intent`, `rewritten_query`.
- **Redis** (`SessionService`, `app/services/session_service.py`): short-lived (`HISTORY_TTL`) cache of the last `MAX_HISTORY_MESSAGES * 2` messages per conversation, keyed as `{KEY_PREFIX}:{conversation_id}:history`, used to reconstruct LangChain message history for the graph without hitting Postgres on every turn. `append_message` uses optimistic locking (`WATCH`/`MULTI`) with a retry loop. When a Redis cache is cold, `get_messages` (paginated history endpoint) re-warms it from Postgres.
- **Qdrant** (`RetrievalService`): vector search over a single collection (`QDRANT_COLLECTION`), score-thresholded (`SCORE_THRESHOLD`), returning `SearchResult(text, score, metadata)`. Embeddings are produced by `EmbeddingGenerator` (`sentence-transformers`, prefixes queries with `"search_query: "`), and its output dimension is validated against `EMBEDDING_DIM` at startup — a mismatched embedding model will fail fast rather than silently corrupt search.

### Service wiring

All services (`EmbeddingGenerator`, `SessionService`, `RetrievalService`, `ChatService`, `IntentService`) and the compiled graph are constructed once in `app/main.py`'s `lifespan` and stashed on `app.state`; controllers/handlers pull them from `Request.app.state` rather than via FastAPI `Depends`. Only the DB session is wired through `Depends(get_async_session)` (`app/core/database.py`).

### Adding a new graph node/branch

Follow the existing pattern: add the node method to `GraphNodes` (or a new service class if it owns distinct LLM/business logic), register it in `create_assistant_graph` (`app/graph/workflow.py`), and extend `AgentState` (`app/schemas/agent_state.py`) with any new state keys. Branching nodes return `Command(update=..., goto=...)` rather than using `add_conditional_edges`.
