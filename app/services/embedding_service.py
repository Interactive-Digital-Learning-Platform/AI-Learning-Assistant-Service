from app.core.config import EMBEDDING_MODEL, EMBEDDING_DEVICE, MAX_TOKENS
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        logger.info(f"Loading embedding model '{model_name}' on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Embedding model ready — dim={self.embedding_dimension()}")

    def embed_single(self, text: str) -> List[float]:
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        prefixed = self._truncate(prefixed, label="query")

        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def embedding_dimension(self) -> int:
        return self.model.get_embedding_dimension()

    def _truncate(self, text: str, label: str = "text") -> str:
        estimated = int(len(text.split()) * 1.3)
        if estimated > MAX_TOKENS:
            max_words = int(MAX_TOKENS / 1.3)
            text = " ".join(text.split()[:max_words])
            logger.warning(f"{label} truncated to {max_words} words")
        return text
