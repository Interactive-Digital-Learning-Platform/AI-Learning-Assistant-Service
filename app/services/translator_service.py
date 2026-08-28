import logging
from asyncio import to_thread
from dataclasses import dataclass

from azure.ai.translation.text.models import InputTextItem

from app.core.azure_ai_translator import text_translator
from app.core.config import settings
from app.utils.language import chunk_text

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    pass


@dataclass(frozen=True)
class TranslationResult:
    text: str
    detected_language: str
    score: float


class TranslatorService:
    def __init__(self, client=text_translator):
        self._client = client
        self._char_limit = settings.AZURE_TRANSLATOR_CHAR_LIMIT

    async def to_english(self, text: str) -> TranslationResult:
        if not text or not text.strip():
            return TranslationResult(text=text, detected_language="", score=0.0)

        return await self._translate(text, to_language="en")

    async def to_sinhala(self, text: str) -> str:
        if not text or not text.strip():
            return text

        result = await self._translate(text, to_language="si", from_language="en")
        return result.text

    async def _translate(
        self, text: str, to_language: str, from_language: str | None = None
    ) -> TranslationResult:
        chunks = chunk_text(text, self._char_limit)

        translated_parts: list[str] = []
        detected_language = ""
        detected_score = 0.0

        for chunk in chunks:
            item, chunk_lang, chunk_score = await self._translate_chunk(
                chunk, to_language, from_language
            )
            translated_parts.append(item)

            if chunk_score > detected_score:
                detected_language = chunk_lang
                detected_score = chunk_score

        return TranslationResult(
            text="".join(translated_parts),
            detected_language=detected_language,
            score=detected_score,
        )

    async def _translate_chunk(
        self, chunk: str, to_language: str, from_language: str | None
    ) -> tuple[str, str, float]:
        try:
            response = await to_thread(
                self._client.translate,
                [InputTextItem(text=chunk)],
                to_language=[to_language],
                from_language=from_language,
            )
        except Exception as exc:
            logger.exception("Azure translation call failed")
            raise TranslationError(str(exc)) from exc

        if not response or not response[0].translations:
            raise TranslationError("Azure translation returned no content")

        first = response[0]
        detected = first.detected_language
        return (
            first.translations[0].text,
            detected.language if detected else "",
            detected.score if detected else 0.0,
        )
