import re
from dataclasses import dataclass, field
from typing import Any

from app.schemas.retrieval import SearchResult

_MARKER = re.compile(
    r"[ \t]*(?:[\[【]S\d+(?:\s*,\s*S?\d+)*[\]】]|(【)[^【】]*】)", re.IGNORECASE
)
_NUMBER = re.compile(r"\d+")
_MAX_PENDING_CHARS = 120


@dataclass
class CitationRewriter:
    visible: bool
    excerpt_count: int | None = None
    order: list[int] = field(default_factory=list)

    def rewrite(self, text: str) -> str:
        return _MARKER.sub(self._replace, text)

    def _replace(self, match: re.Match[str]) -> str:
        if match.group(1):
            return ""

        labels: list[str] = []

        for number in (int(n) for n in _NUMBER.findall(match.group(0))):
            if self.excerpt_count is not None and not 1 <= number <= self.excerpt_count:
                continue

            if number not in self.order:
                self.order.append(number)

            label = str(self.order.index(number) + 1)
            if label not in labels:
                labels.append(label)

        if not self.visible or not labels:
            return ""

        marker = match.group(0)
        leading = marker[: len(marker) - len(marker.lstrip(" \t"))]
        return f"{leading}[{', '.join(labels)}]"


class CitationStreamFilter:
    def __init__(self, visible: bool):
        self._rewriter = CitationRewriter(visible=visible)
        self._pending = ""

    def feed(self, token: str) -> str:
        self._pending += token
        cut = self._safe_cut(self._pending)
        ready, self._pending = self._pending[:cut], self._pending[cut:]
        return self._rewriter.rewrite(ready) if ready else ""

    def flush(self) -> str:
        ready, self._pending = self._pending, ""
        return self._rewriter.rewrite(ready) if ready else ""

    @staticmethod
    def _safe_cut(text: str) -> int:
        cut = len(text)
        opener = max(text.rfind("["), text.rfind("【"))

        if opener != -1:
            closer = "]" if text[opener] == "[" else "】"
            if closer not in text[opener:] and len(text) - opener <= _MAX_PENDING_CHARS:
                cut = opener

        while cut > 0 and text[cut - 1] in " \t":
            cut -= 1

        return cut


def source_list(chunks: list[SearchResult], mode: str) -> list[dict[str, Any]]:
    if mode == "web_search":
        return [
            {
                "title": c.metadata.get("title", ""),
                "url": c.metadata.get("url", ""),
                "snippet": c.text[:300],
                "provider": c.metadata.get("source"),
                "score": c.score,
            }
            for c in chunks
        ]

    return [
        {
            "filename": c.metadata.get("filename", ""),
            "page": c.metadata.get("page_start", 0),
            "score": c.score,
        }
        for c in chunks
    ]
