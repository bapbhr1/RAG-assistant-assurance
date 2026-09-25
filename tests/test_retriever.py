# Recherche hybride avec un faux modèle d'embeddings (aucun téléchargement)

import pytest
from conftest import make_chunk

from src.retriever import BM25, FUSIONS, HybridRetriever, _words

CHUNKS = [
    make_chunk("AUTO-1::A1", "La franchise bris de glace est de 90 EUR.", "auto"),
    make_chunk("AUTO-1::A2", "Le vol total est indemnisé après 30 jours.", "auto"),
    make_chunk("HAB-1::H1", "Le dégât des eaux est garanti avec une franchise.", "habitation"),
]


@pytest.fixture
def retriever(hash_embedding) -> HybridRetriever:
    instance = HybridRetriever(embedding_model=hash_embedding)
    instance.index(CHUNKS)
    return instance


def test_mots_normalises():
    assert _words("Le Dégât des EAUX") == ["degat", "eaux"]


def test_bm25_favorise_le_document_pertinent():
    scores = BM25([_words(c.text) for c in CHUNKS]).scores(_words("vol total"))
    assert scores.argmax() == 1
    assert scores[2] == 0.0


def test_filtre_par_branche(retriever):
    results = retriever.search("franchise", top_k=5, product_line="habitation")
    assert [r.chunk.chunk_id for r in results] == ["HAB-1::H1"]


def test_branche_inconnue(retriever):
    assert retriever.search("franchise", product_line="bateau") == []


@pytest.mark.parametrize("fusion", FUSIONS)
def test_chaque_fusion_trouve_le_vol(retriever, fusion):
    retriever.fusion = fusion
    results = retriever.search("vol total indemnisé", top_k=2)
    assert results[0].chunk.chunk_id == "AUTO-1::A2"


def test_fusion_inconnue(hash_embedding):
    with pytest.raises(ValueError):
        HybridRetriever(fusion="magique", embedding_model=hash_embedding)


def test_recherche_sans_index(hash_embedding):
    with pytest.raises(RuntimeError):
        HybridRetriever(embedding_model=hash_embedding).search("vol")
