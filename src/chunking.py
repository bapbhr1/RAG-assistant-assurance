# Découpage du corpus : une clause entière = une unité de recherche traçable.

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Chunk, Contract

MAX_CHUNK_CHARS = 900


def load_contracts(path: str | Path) -> list[Contract]:
    with Path(path).open(encoding="utf-8") as stream:
        return [Contract.model_validate(item) for item in json.load(stream)]


def _split(text: str) -> list[str]:
    # phrase par phrase, regroupées jusqu'à la limite de taille
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    blocks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > MAX_CHUNK_CHARS:
            blocks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        blocks.append(current)
    return blocks


def build_corpus(contracts: list[Contract]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for contract in contracts:
        for article in contract.articles:
            parts = (
                [article.text.strip()]
                if len(article.text) <= MAX_CHUNK_CHARS
                else _split(article.text)
            )
            for index, text in enumerate(parts):
                suffix = f"-{index + 1}" if len(parts) > 1 else ""
                chunks.append(
                    Chunk(
                        chunk_id=f"{contract.contract_id}::{article.article_id}{suffix}",
                        contract_id=contract.contract_id,
                        product=contract.product,
                        product_line=contract.product_line,
                        article_id=article.article_id,
                        article_title=article.title,
                        text=text,
                    )
                )
    return chunks
