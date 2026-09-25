# Garde-fous de bout en bout : le modèle est simulé, les contrôles sont réels.

from conftest import (
    FakeGroq,
    FakeGroqError,
    FakeRetriever,
    connection_error,
    llm_payload,
    rate_limit_error,
)

from src.models import (
    CoverageStatus,
    EvidenceCheck,
    RAGQuery,
    RetrievedChunk,
)
from src.rag import RAGEngine, _confidence, _requires_review


def _engine(retrieved, content: str) -> RAGEngine:
    engine = RAGEngine(FakeRetriever(retrieved), api_key="test")
    engine.client = FakeGroq(content)
    return engine


def _ask(engine: RAGEngine):
    return engine.answer(RAGQuery(question="Les dommages à mon véhicule sont-ils couverts ?"))


def test_reponse_etayee_conservee(retrieved):
    response = _ask(_engine(retrieved, llm_payload()))
    assert response.status == CoverageStatus.COVERED
    assert len(response.citations) == 1
    assert response.confidence >= 0.85
    assert response.warnings == []


def test_citation_inventee_neutralise_la_decision(retrieved):
    claims = [
        {
            "claim": "Le véhicule de prêt est fourni 30 jours.",
            "chunk_id": "AUTO-1::A3",
            "quote": "Un véhicule de prêt est fourni pendant 30 jours.",
        }
    ]
    response = _ask(_engine(retrieved, llm_payload(claims=claims)))
    assert response.status == CoverageStatus.NEEDS_REVIEW
    assert response.citations == []
    assert "Décision automatique neutralisée" in response.warnings[-1]


def test_montant_invente_neutralise_la_decision(retrieved):
    claims = [
        {
            "claim": "La franchise est de 150 EUR.",
            "chunk_id": "AUTO-1::A3",
            "quote": "La franchise applicable est de 350 EUR par sinistre.",
        }
    ]
    response = _ask(_engine(retrieved, llm_payload(claims=claims)))
    assert response.status == CoverageStatus.NEEDS_REVIEW


def test_json_invalide_renvoie_vers_un_gestionnaire(retrieved):
    response = _ask(_engine(retrieved, "pas du JSON"))
    assert response.status == CoverageStatus.NEEDS_REVIEW
    assert response.confidence == 0.0
    assert response.warnings[0].startswith("Échec de la génération structurée")
    assert response.service_unavailable is False


def test_quota_groq_atteint_signale_indisponibilite(retrieved):
    engine = RAGEngine(FakeRetriever(retrieved), api_key="test")
    engine.client = FakeGroqError(rate_limit_error())
    response = _ask(engine)
    assert response.status == CoverageStatus.NEEDS_REVIEW
    assert response.service_unavailable is True
    assert "quota" in response.warnings[0].lower()


def test_groq_indisponible_signale_indisponibilite(retrieved):
    engine = RAGEngine(FakeRetriever(retrieved), api_key="test")
    engine.client = FakeGroqError(connection_error())
    response = _ask(engine)
    assert response.status == CoverageStatus.NEEDS_REVIEW
    assert response.service_unavailable is True
    assert "indisponible" in response.warnings[0].lower()


def test_aucune_clause_abstention_fiable():
    response = _ask(_engine([], llm_payload()))
    assert response.status == CoverageStatus.NEEDS_REVIEW
    assert response.confidence == 1.0


def test_sources_trop_eloignees(franchise_chunk):
    weak = [RetrievedChunk(chunk=franchise_chunk, score=0.55)]
    response = _ask(_engine(weak, llm_payload()))
    assert response.status == CoverageStatus.NEEDS_REVIEW


def test_informations_manquantes_imposent_une_validation(retrieved):
    checks = [EvidenceCheck(claim="x", supported=True)]
    assert not _requires_review(retrieved, checks, [], [])
    assert _requires_review(retrieved, checks, ["Date du sinistre"], [])
    assert _requires_review(retrieved, checks, [], ["Clauses contradictoires"])
    assert _requires_review(retrieved, [], [], [])


def test_modele_trop_affirmatif_penalise(retrieved):
    # le garde-fou corrige le statut, mais la fiabilité note ce que le modèle a proposé
    checks = [EvidenceCheck(claim="x", supported=False)]
    overconfident, breakdown = _confidence(
        retrieved, checks, [], [], CoverageStatus.COVERED
    )
    cautious, _ = _confidence(retrieved, checks, [], [], CoverageStatus.NEEDS_REVIEW)
    assert breakdown.uncertainty_handling == 0.0
    assert overconfident < cautious


def test_modele_trop_prudent_penalise(retrieved):
    checks = [EvidenceCheck(claim="x", supported=True)]
    _, breakdown = _confidence(retrieved, checks, [], [], CoverageStatus.NEEDS_REVIEW)
    assert breakdown.uncertainty_handling == 0.0
