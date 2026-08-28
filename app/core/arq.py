from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.core.config import settings


def get_arq_redis_settings() -> RedisSettings:
    arq_redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    arq_redis_settings.database = settings.ATTACHMENT_QUEUE_REDIS_DB

    return arq_redis_settings

    
async def create_arq_pool() -> ArqRedis:
    arq_pool = await create_pool(get_arq_redis_settings())
    return arq_pool