# Fixtures communes : clauses de test et faux modèles, sans réseau ni clé API.

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import groq
import httpx
import numpy as np
import pytest

from src.models import Chunk, RetrievedChunk


def make_chunk(chunk_id: str, text: str, product_line: str = "auto") -> Chunk:
    contract_id, article_id = chunk_id.split("::")
    return Chunk(
        chunk_id=chunk_id,
        contract_id=contract_id,
        product=f"Assurance {product_line}",
        product_line=product_line,
        article_id=article_id,
        article_title=f"Article {article_id}",
        text=text,
    )


@pytest.fixture
def franchise_chunk() -> Chunk:
    return make_chunk(
        "AUTO-1::A3",
        "L'assureur garantit les dommages subis par le véhicule assuré. "
        "La franchise applicable est de 350 EUR par sinistre. "
        "Cette garantie ne s'applique pas si le conducteur était en état d'ivresse.",
    )


@pytest.fixture
def retrieved(franchise_chunk: Chunk) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=franchise_chunk, score=0.80)]


class HashEmbedding:
    # sac de mots haché : assez proche d'un vrai modèle pour tester le classement
    def embed(self, texts: list[str]):
        for text in texts:
            vector = np.zeros(64, dtype=np.float32)
            for word in text.lower().split():
                digest = hashlib.md5(word.encode()).digest()
                vector[digest[0] % 64] += 1.0
            yield vector


@pytest.fixture
def hash_embedding() -> HashEmbedding:
    return HashEmbedding()


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    def search(self, query, top_k=5, product_line=None):
        return self.results


class FakeGroq:
    # imite client.chat.completions.create et renvoie un contenu prédéfini
    def __init__(self, content: str) -> None:
        self.content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        message = SimpleNamespace(content=self.content)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeGroqError:
    # imite client.chat.completions.create mais lève une erreur du SDK Groq
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        raise self.exc


def rate_limit_error() -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request, json={"error": {"message": "quota atteint"}})
    return groq.RateLimitError("quota atteint", response=response, body=None)


def connection_error() -> groq.APIConnectionError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.APIConnectionError(message="Connection error.", request=request)


def llm_payload(**overrides) -> str:
    payload = {
        "status": "PRIS_EN_CHARGE",
        "answer": "Les dommages sont couverts avec une franchise de 350 EUR.",
        "decision": "Indemniser après application de la franchise.",
        "conditions": ["Franchise de 350 EUR par sinistre"],
        "missing_information": [],
        "claims": [
            {
                "claim": "La franchise est de 350 EUR par sinistre.",
                "chunk_id": "AUTO-1::A3",
                "quote": "La franchise applicable est de 350 EUR par sinistre.",
            }
        ],
        "conflicts": [],
        "reasoning_summary": "Article 3 → dommages au véhicule → couverts.",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)
