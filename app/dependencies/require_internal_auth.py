import hmac

from fastapi import Header, HTTPException, status

from app.core.config import settings


async def require_internal_auth(
    x_internal_key: str | None = Header(default=None, alias="X-Internal-Key")
) -> None:

    if x_internal_key is None or not hmac.compare_digest(
        x_internal_key, settings.INTERNAL_SERVICE_KEY
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid service key"
        )