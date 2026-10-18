import json
import logging

from langchain_core.messages import AIMessage, HumanMessage
from redis.exceptions import WatchError

from app.core.config import settings
from app.core.redis import redis_instance

logger = logging.getLogger(__name__)

_APPEND_MAX_ATTEMPTS = 5


class SessionService:
    def __init__(self):
        self._redis = redis_instance

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"{settings.KEY_PREFIX}:{conversation_id}:history"

    @staticmethod
    def _to_langchain(messages: list[dict]) -> list:
        lc_messages = []

        for msg in messages[-(settings.MAX_HISTORY_MESSAGES * 2) :]:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        return lc_messages

    @staticmethod
    def _to_cache_entries(db_messages: list) -> list[dict]:
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in db_messages
            if msg.role.value in ("user", "assistant")
        ]

        return messages[-(settings.MAX_HISTORY_MESSAGES * 2) :]

    def history_from_db(self, db_messages: list) -> list:
        return self._to_langchain(self._to_cache_entries(db_messages))

    async def get_cached_history(self, conversation_id: str) -> list | None:
        key = self._key(conversation_id)
        data = await self._redis.get(key)

        if not data:
            logger.debug(f"Cache miss for conversation {conversation_id}")
            return None

        await self._redis.expire(key, settings.HISTORY_TTL)

        return self._to_langchain(json.loads(data))

    async def get_langchain_history(
        self,
        conversation_id: str,
    ) -> list:

        return await self.get_cached_history(conversation_id) or []

    async def get_raw_history(self, conversation_id: str) -> list[dict]:
        key = self._key(conversation_id)
        data = await self._redis.get(key)
        return json.loads(data) if data else []

    async def cache_exists(self, conversation_id: str) -> bool:
        return await self._redis.exists(self._key(conversation_id)) > 0

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:

        key = self._key(conversation_id)
        max_store = settings.MAX_HISTORY_MESSAGES * 2
        messages: list[dict] = []

        async with self._redis.pipeline(transaction=True) as pipe:
            for attempt in range(1, _APPEND_MAX_ATTEMPTS + 1):
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
                except WatchError:
                    if attempt == _APPEND_MAX_ATTEMPTS:
                        logger.warning(
                            f"History append gave up after {attempt} concurrent-write "
                            f"conflicts — conversation={conversation_id}"
                        )
                        raise

        logger.debug(
            f"Appended {role} message — "
            f"conversation={conversation_id} total={len(messages)}"
        )

    async def warm_cache(
        self,
        conversation_id: str,
        db_messages: list,
    ) -> None:

        messages = self._to_cache_entries(db_messages)

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
