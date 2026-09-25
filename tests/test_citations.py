# Contrôle des citations : tolérant à la typographie, pas aux reformulations,
# et la clause citée doit aussi contenir les montants annoncés.

import pytest

from src.models import ClaimEvidence
from src.rag import (
    _claim_numbers_supported,
    _numbers,
    _quote_matches_source,
    _validate_claims,
)

SOURCE = (
    "L'assureur prend en charge les frais dans la limite de 1 000 EUR. "
    "Un délai de carence de 30 jours s'applique. Aucune franchise n'est due."
)


@pytest.mark.parametrize(
    "quote",
    [
        "L'assureur prend en charge les frais dans la limite de 1 000 EUR.",
        "L’assureur prend en charge les frais",  # apostrophe courbe
        "l'ASSUREUR prend en charge les frais",  # casse
        "dans la limite de 1 000 EUR",  # espace insécable
        "L'assureur prend en charge … délai de carence de 30 jours",  # passages reliés
    ],
)
def test_citation_acceptee(quote):
    assert _quote_matches_source(quote, SOURCE)


@pytest.mark.parametrize(
    "quote",
    [
        "L'assureur rembourse les frais dans la limite de 1 000 EUR.",  # reformulation
        "dans la limite de 2 000 EUR",  # montant modifié
        "délai de carence de 30 jours … L'assureur prend en charge",  # ordre inversé
        "frais",  # trop court pour être probant
    ],
)
def test_citation_refusee(quote):
    assert not _quote_matches_source(quote, SOURCE)


def test_extraction_des_montants():
    assert _numbers("Plafond de 100 000 000 EUR, franchise de 10% et 0,5 point") == {
        "100000000", "10", "0.5",
    }
    assert _numbers("1 000 €") == _numbers("1000 EUR") == {"1000"}
    assert _numbers("un délai de dix jours, limité à deux par an") == {"10", "2"}


def test_identifiants_ignores():
    assert _numbers("Contrat SANTE-2024-010, article 6 (SP6) : 8 séances") == {"8"}
    assert _numbers("cesse au 1095e jour") == set()


def test_montant_absent_de_la_clause():
    assert _claim_numbers_supported("Plafond de 1 000 EUR.", SOURCE)
    assert not _claim_numbers_supported("Plafond de 1 500 EUR.", SOURCE)
    assert _claim_numbers_supported("Les frais sont pris en charge.", SOURCE)
    # montant repris d'une autre phrase de la même clause
    assert _claim_numbers_supported("Carence de 30 jours sur un plafond de 1 000 EUR.", SOURCE)


def test_montant_donne_par_la_question_tolere():
    quote = "une rente proportionnelle au taux d'invalidité"
    claim = "Pour un taux de 40 %, la rente est proportionnelle."
    assert not _claim_numbers_supported(claim, quote)
    assert _claim_numbers_supported(claim, quote, "Invalidité reconnue à 40 % ?")


def test_extrait_authentique_mais_montant_invente(retrieved):
    claims = [
        ClaimEvidence(
            claim="La franchise est de 150 EUR.",
            chunk_id="AUTO-1::A3",
            quote="La franchise applicable est de 350 EUR par sinistre.",
        )
    ]
    checks, citations, warnings = _validate_claims(claims, retrieved)
    assert not checks[0].supported
    assert citations == []
    assert warnings == ["Montant absent de la clause citée : La franchise est de 150 EUR."]


def test_identifiant_approximatif_resolu_par_le_texte(retrieved):
    claims = [
        ClaimEvidence(
            claim="La franchise est de 350 EUR.",
            chunk_id="A3",
            quote="La franchise applicable est de 350 EUR par sinistre.",
        )
    ]
    checks, citations, _ = _validate_claims(claims, retrieved)
    assert checks[0].supported
    assert citations[0].chunk_id == "AUTO-1::A3"
