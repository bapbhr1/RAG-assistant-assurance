# Recherche hybride : embeddings denses (fastembed) + recouvrement de mots-clés.
# BM25 et la fusion par rangs (RRF) sont gardés pour la comparaison d'evaluate.py ;
# sur le jeu d'évaluation, ils font moins bien que la fusion pondérée par défaut.

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np

from .models import Chunk, RetrievedChunk

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle",
    "en", "est", "et", "je", "la", "le", "les", "ma", "mes", "mon", "pour",
    "que", "quel", "quelle", "qui", "si", "sous", "sur", "un", "une",
}
FUSIONS = ("ponderee", "dense", "bm25", "rrf")

# pondération dense / lexical de la fusion par défaut
DENSE_WEIGHT = 0.72
LEXICAL_WEIGHT = 0.28
# constante usuelle de la fusion par rangs (Cormack et al., 2009)
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75


def _words(text: str) -> list[str]:
    normalized = (
        unicodedata.normalize("NFKD", text.lower())
        .encode("ascii", "ignore")
        .decode()
    )
    return [
        word
        for word in re.findall(r"[a-z0-9]+", normalized)
        if len(word) > 2 and word not in STOPWORDS
    ]


def _tokens(text: str) -> set[str]:
    return set(_words(text))


def _ranks(scores: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(scores))
    ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return ranks


class BM25:
    def __init__(self, documents: list[list[str]]) -> None:
        self.counts = [Counter(doc) for doc in documents]
        self.lengths = np.asarray([len(doc) for doc in documents], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if documents else 0.0
        frequencies = Counter(word for doc in documents for word in set(doc))
        total = len(documents)
        self.idf = {
            word: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for word, freq in frequencies.items()
        }

    def scores(self, query_words: list[str]) -> np.ndarray:
        result = np.zeros(len(self.counts), dtype=np.float32)
        norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lengths / max(self.avg_length, 1e-12))
        for word in set(query_words):
            idf = self.idf.get(word)
            if idf is None:
                continue
            tf = np.asarray([counts[word] for counts in self.counts], dtype=np.float32)
            result += idf * tf * (BM25_K1 + 1) / (tf + norm)
        return result


# Index en mémoire, suffisant tant qu'on reste sur quelques dizaines de clauses.
class HybridRetriever:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        fusion: str = "ponderee",
        embedding_model=None,
    ) -> None:
        if fusion not in FUSIONS:
            raise ValueError(f"Fusion inconnue : {fusion}")
        self.fusion = fusion
        if embedding_model is not None:
            self.model = embedding_model
        else:
            from fastembed import TextEmbedding

            # modèle déjà téléchargé en priorité, sinon on le récupère
            try:
                self.model = TextEmbedding(model_name=model_name, local_files_only=True)
            except Exception:
                self.model = TextEmbedding(model_name=model_name)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.token_sets: list[set[str]] = []
        self.bm25: BM25 | None = None

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
        self.bm25 = BM25([_words(text) for text in texts])

    def _combine(self, query: str, dense: np.ndarray, lexical: np.ndarray) -> np.ndarray:
        if self.fusion == "ponderee":
            return DENSE_WEIGHT * dense + LEXICAL_WEIGHT * lexical
        if self.fusion == "dense":
            return dense
        bm25 = self.bm25.scores(_words(query))
        if self.fusion == "bm25":
            return bm25
        return 1.0 / (RRF_K + _ranks(dense)) + 1.0 / (RRF_K + _ranks(bm25))

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
        combined = self._combine(query, dense, lexical)
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
