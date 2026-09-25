# Découpage du corpus : une clause courte reste entière, une clause longue est
# coupée entre deux phrases, et chaque morceau garde son identifiant.

from pathlib import Path

from src.chunking import MAX_CHUNK_CHARS, build_corpus, load_contracts
from src.models import Contract, ContractArticle

CONTRACTS_PATH = Path(__file__).parent.parent / "data" / "synthetic_contracts.json"


def _contract(text: str) -> Contract:
    return Contract(
        contract_id="AUTO-1",
        product="Assurance Auto",
        product_line="auto",
        title="Contrat test",
        policyholder="Client test",
        articles=[ContractArticle(article_id="A1", title="Article 1", text=text)],
    )


def test_article_court_garde_un_seul_chunk():
    chunks = build_corpus([_contract("Une clause courte.")])
    assert [c.chunk_id for c in chunks] == ["AUTO-1::A1"]


def test_article_long_decoupe_entre_deux_phrases():
    sentence = "Cette phrase de test décrit une garantie contractuelle. "
    chunks = build_corpus([_contract(sentence * 40)])
    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks][:2] == ["AUTO-1::A1-1", "AUTO-1::A1-2"]
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_corpus_synthetique_complet():
    contracts = load_contracts(CONTRACTS_PATH)
    chunks = build_corpus(contracts)
    assert len(contracts) == 15
    assert len({(c.contract_id, c.article_id) for c in chunks}) == 112
    assert len({c.chunk_id for c in chunks}) == len(chunks)
