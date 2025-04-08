
from neo4j_graphrag.embeddings.base import Embedder
from .embeddings import EmbeddingModel
import logging  

logger = logging.getLogger(__name__)

class NeoJSEmbedder(Embedder):
    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0"):
        self.embedding_model = EmbeddingModel(provider="bedrock")
        self.model_id = model_id

    def embed_query(self, text: str) -> list:
        logger.debug(f"Embedding text: {text}")
        embedding = self.embedding_model.embed_text(text=text, model_id=self.model_id)
        logger.info("Embedding Completed")
        return embedding