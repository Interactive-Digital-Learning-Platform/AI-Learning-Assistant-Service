import json
import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import settings
from app.core.redis import redis_instance

logger = logging.getLogger(__name__)


class SessionService:
    def __init__(self):
        self._redis = redis_instance

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"{settings.KEY_PREFIX}:{conversation_id}:history"

    async def get_langchain_history(
        self,
        conversation_id: str,
    ) -> list:

        key = self._key(conversation_id)
        data = await self._redis.get(key)

        if not data:
            logger.debug(f"Cache miss for conversation {conversation_id}")
            return []

        await self._redis.expire(key, settings.HISTORY_TTL)

        messages = json.loads(data)

        messages = messages[-(settings.MAX_HISTORY_MESSAGES * 2) :]

        lc_messages = []

        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        return lc_messages

    async def get_raw_history(self, conversation_id: str) -> list[dict]:
        """Return raw list of {role, content} dicts — for debugging."""
        key = self._key(conversation_id)
        data = await self._redis.get(key)
        return json.loads(data) if data else []

    async def cache_exists(self, conversation_id: str) -> bool:
        """Check whether this conversation has a warm cache."""
        return await self._redis.exists(self._key(conversation_id)) > 0

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:

        key = self._key(conversation_id)
        max_store = settings.MAX_HISTORY_MESSAGES * 2

        async with self._redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(key)
                    data = await pipe.get(key)
                    messages = json.loads(data) if data else []
                    messages.append({"role": role, "content": content})
                    if len(messages) > max_store:
                        messages = messages[-max_store:]
                    pipe.multi()
                    await pipe.set(key, json.dumps(messages), ex=settings.HISTORY_TTL)
                    await pipe.execute()
                    break
                except Exception:
                    continue

        logger.debug(
            f"Appended {role} message — "
            f"conversation={conversation_id} total={len(messages)}"
        )

    async def warm_cache(
        self,
        conversation_id: str,
        db_messages: list,
    ) -> None:

        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in db_messages
            if msg.role.value in ("user", "assistant")
        ]

        messages = messages[-(settings.MAX_HISTORY_MESSAGES * 2) :]

        key = self._key(conversation_id)
        await self._redis.set(key, json.dumps(messages), ex=settings.HISTORY_TTL)
        logger.info(
            f"Cache warmed — conversation={conversation_id} messages={len(messages)}"
        )

    async def clear(self, conversation_id: str) -> None:

        await self._redis.delete(self._key(conversation_id))
        logger.info(f"Cache cleared — conversation={conversation_id}")

    async def close(self) -> None:
        await self._redis.aclose()
