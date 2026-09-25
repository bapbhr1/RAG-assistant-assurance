# Traçage optionnel avec Langfuse : recherche, appel LLM et contrôles de chaque question.
# Actif seulement si LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY sont définis ;
# sinon les observations ne font rien et le pipeline tourne à l'identique.

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator


class _NoObservation:
    def update(self, **kwargs: Any) -> None:
        pass


@lru_cache(maxsize=1)
def _client():
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import get_client
    except ImportError:
        return None
    return get_client()


def enabled() -> bool:
    return _client() is not None


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    client = _client()
    if client is None:
        yield _NoObservation()
        return
    with client.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
        yield obs


def flush() -> None:
    # à appeler en fin de script : l'envoi se fait en tâche de fond
    client = _client()
    if client is not None:
        client.flush()
