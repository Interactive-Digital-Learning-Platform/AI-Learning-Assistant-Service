import logging

from fastembed import SparseTextEmbedding
from qdrant_client import models as qmodels

from app.core.config import settings

logger = logging.getLogger(__name__)


class SparseEmbeddingGenerator:
    def __init__(
        self,
        model_name: str = settings.SPARSE_MODEL,
        cache_dir: str | None = settings.FASTEMBED_CACHE_DIR,
    ):
        self.model_name = model_name

        logger.info(f"Loading sparse model '{model_name}'...")
        self.model = SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir)
        logger.info("Sparse model ready")

    def embed_query(self, text: str) -> qmodels.SparseVector:
        embedding = next(iter(self.model.query_embed(text)))

        return qmodels.SparseVector(
            indices=embedding.indices.tolist(), values=embedding.values.tolist()
        )
