import logging
import re
from asyncio import wait_for

from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.prompts.title_prompts import NO_TOPIC, title_prompt
from app.services.translator_service import TranslationError, TranslatorService

logger = logging.getLogger(__name__)

_MESSAGE_CHARS = 1000
_REPLY_CHARS = 1500
_COLUMN_CHARS = 255
_LABEL_PREFIX = re.compile(r"^(conversation\s+)?title\s*:\s*", re.IGNORECASE)
_WRAPPING_CHARS = " \t\"'`*#_“”‘’"
_TRAILING_CHARS = " .!?:;,"


class TitleService:
    def __init__(
        self,
        llm,
        translator: TranslatorService,
        enabled: bool = settings.TITLE_GENERATION_ENABLED,
        timeout_seconds: float = settings.TITLE_GENERATION_TIMEOUT_SECONDS,
        max_chars: int = settings.TITLE_MAX_CHARS,
    ):
        self.translator = translator
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.max_chars = max_chars
        self._chain = title_prompt | llm | StrOutputParser()

    async def generate(
        self,
        user_message: str,
        assistant_reply: str,
        reply_language: str = "English",
    ) -> str | None:
        if not self.enabled:
            return self.fallback(user_message)

        try:
            raw_title = await wait_for(
                self._chain.ainvoke(
                    {
                        "user_message": user_message[:_MESSAGE_CHARS],
                        "assistant_reply": assistant_reply[:_REPLY_CHARS],
                    }
                ),
                timeout=self.timeout_seconds,
            )
        except Exception:
            logger.exception("Title generation failed; using the first message")
            return self.fallback(user_message)

        title = self._clean(raw_title)

        if title.upper() == NO_TOPIC:
            logger.info("Title deferred — the message has no study topic yet")
            return None

        if not title:
            return self.fallback(user_message)

        if reply_language == "Sinhala":
            try:
                title = (await self.translator.to_sinhala(title)).strip() or title
            except TranslationError:
                logger.exception("Title translation failed; keeping the English title")

        return title[:_COLUMN_CHARS]

    def fallback(self, user_message: str) -> str:
        text = " ".join(user_message.split())

        if len(text) <= self.max_chars:
            return text

        return text[: self.max_chars - 3].rstrip() + "..."

    def _clean(self, raw_title: str) -> str:
        lines = [line for line in raw_title.splitlines() if line.strip()]

        if not lines:
            return ""

        title = _LABEL_PREFIX.sub("", lines[0].strip(_WRAPPING_CHARS))
        title = " ".join(title.strip(_WRAPPING_CHARS).rstrip(_TRAILING_CHARS).split())

        if len(title) > self.max_chars:
            title = title[: self.max_chars].rsplit(" ", 1)[0]

        return title
