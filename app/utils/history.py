from langchain_core.messages import BaseMessage

from app.core.config import settings

_MIN_KEPT_CHARS = 200
_RESERVED_CHARS = 600

def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    cut = text.rfind(" ", 0, limit)
    cut = cut if cut > limit // 2 else limit
    return text[:cut].rstrip() + " …"


def budget_history(
    messages: list[BaseMessage],
    max_chars: int = settings.HISTORY_MAX_CHARS,
    full_messages: int = settings.HISTORY_FULL_MESSAGES,
    older_message_chars: int = settings.HISTORY_OLDER_MESSAGE_CHARS,
) -> list[BaseMessage]:
    kept: list[BaseMessage] = []
    used = 0
    message_cap = max(max_chars - _RESERVED_CHARS, _MIN_KEPT_CHARS)

    for position, message in enumerate(reversed(messages)):
        content = message.content if isinstance(message.content, str) else str(message.content)
        remaining = max_chars - used

        candidates = [_shorten(content, message_cap)] if position < full_messages else []
        candidates.append(_shorten(content, older_message_chars))

        chosen = next((c for c in candidates if len(c) <= remaining), None)

        if chosen is None:
            if position < 2 and remaining >= _MIN_KEPT_CHARS:
                chosen = _shorten(content, remaining)
            else:
                break

        kept.append(message.model_copy(update={"content": chosen}))
        used += len(chosen)

    return list(reversed(kept))
