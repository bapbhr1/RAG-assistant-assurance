# Recherche hybride : embeddings denses (fastembed) + recouvrement de mots-clés.

from __future__ import annotations

import re
import unicodedata

import numpy as np

from .models import Chunk, RetrievedChunk

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle",
    "en", "est", "et", "je", "la", "le", "les", "ma", "mes", "mon", "pour",
    "que", "quel", "quelle", "qui", "si", "sous", "sur", "un", "une",
}


def _tokens(text: str) -> set[str]:
    normalized = (
        unicodedata.normalize("NFKD", text.lower())
        .encode("ascii", "ignore")
        .decode()
    )
    return {
        word
        for word in re.findall(r"[a-z0-9]+", normalized)
        if len(word) > 2 and word not in STOPWORDS
    }


# Index en mémoire, suffisant tant qu'on reste sur quelques dizaines de clauses.
class HybridRetriever:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding

        # modèle déjà téléchargé en priorité, sinon on le récupère
        try:
            self.model = TextEmbedding(model_name=model_name, local_files_only=True)
        except Exception:
            self.model = TextEmbedding(model_name=model_name)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.token_sets: list[set[str]] = []

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1e-12, norms)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.embed(texts)), dtype=np.float32)

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Le corpus est vide.")
        self.chunks = chunks
        texts = [f"{c.product}. {c.article_title}. {c.text}" for c in chunks]
        self.embeddings = self._normalize(self._encode(texts))
        self.token_sets = [_tokens(text) for text in texts]

    def search(
        self, query: str, top_k: int = 5, product_line: str | None = None
    ) -> list[RetrievedChunk]:
        if self.embeddings is None:
            raise RuntimeError("L'index doit être construit avant la recherche.")
        mask = np.asarray(
            [not product_line or c.product_line == product_line for c in self.chunks]
        )
        if not mask.any():
            return []

        query_vector = self._normalize(self._encode([query]))[0]
        # cosinus ramené de [-1, 1] vers [0, 1] pour rester comparable au score lexical
        dense = np.clip((self.embeddings @ query_vector + 1.0) / 2.0, 0.0, 1.0)
        query_tokens = _tokens(query)
        lexical = np.asarray(
            [
                len(query_tokens & tokens) / max(len(query_tokens), 1)
                for tokens in self.token_sets
            ]
        )
        combined = 0.72 * dense + 0.28 * lexical
        combined = np.where(mask, combined, -np.inf)
        k = min(top_k, int(mask.sum()))
        indexes = np.argsort(-combined)[:k]
        return [
            RetrievedChunk(
                chunk=self.chunks[i], score=float(combined[i]),
                dense_score=float(dense[i]), lexical_score=float(lexical[i]),
            )
            for i in indexes
        ]
