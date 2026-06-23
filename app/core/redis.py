import redis.asyncio as aioredis
from app.core.config import REDIS_URL

redis_instance = aioredis.from_url(
    url=REDIS_URL,
    decode_responses = True
)