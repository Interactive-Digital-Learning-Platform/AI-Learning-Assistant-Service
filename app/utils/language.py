import re
from dataclasses import dataclass

from app.core.config import settings

LANGUAGE_TO_CODE = {"English": "en", "Sinhala": "si"}

_SINHALA_START = chr(0x0D80)
_SINHALA_END = chr(0x0DFF)
_SEGMENT_SPLIT = re.compile("(\n\n+|(?<=[.!?" + chr(0x0DF4) + r"])\s+)")


@dataclass(frozen=True)
class TranslationPlan:
    translate_inbound: bool
    reply_language: str
    translate_outbound: bool
    source_language_hint: str


def contains_sinhala(text: str) -> bool:
    return any(_SINHALA_START <= ch <= _SINHALA_END for ch in text or "")


def resolve_translation_plan(selected_language: str, message: str) -> TranslationPlan:
    if not settings.TRANSLATION_ENABLED:
        return TranslationPlan(False, "English", False, "en")

    has_sinhala_script = contains_sinhala(message)
    is_sinhala = selected_language == "Sinhala" or has_sinhala_script
    reply_language = "Sinhala" if is_sinhala else "English"

    return TranslationPlan(
        translate_inbound=is_sinhala,
        reply_language=reply_language,
        translate_outbound=reply_language == "Sinhala",
        source_language_hint="si" if has_sinhala_script else "en",
    )


def chunk_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    segments = [s for s in _SEGMENT_SPLIT.split(text) if s]
    chunks: list[str] = []
    current = ""

    for segment in segments:
        if len(segment) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(segment), max_chars):
                piece = segment[start : start + max_chars]
                if len(piece) == max_chars:
                    chunks.append(piece)
                else:
                    current = piece
            continue

        if len(current) + len(segment) > max_chars:
            chunks.append(current)
            current = segment
        else:
            current += segment

    if current:
        chunks.append(current)

    return chunks
