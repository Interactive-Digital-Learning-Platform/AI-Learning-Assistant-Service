from dataclasses import dataclass, field
from typing import Literal

RetrievalStatus = Literal["confident", "weak", "none", "error", "skipped"]


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict

    @property
    def contextual_text(self) -> str:
        header = self.metadata.get("context_header")
        return f"{header}\n\n{self.text}" if header else self.text


@dataclass
class RetrievalOutcome:
    status: RetrievalStatus
    results: list[SearchResult] = field(default_factory=list)
    best_score: float | None = None
    scope: dict[str, int | str] = field(default_factory=dict)
