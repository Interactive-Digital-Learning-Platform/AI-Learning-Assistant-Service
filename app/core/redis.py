import redis.asyncio as aioredis
from app.core.config import settings

redis_instance = aioredis.from_url(
    url=settings.REDIS_URL,
    decode_responses = True
)