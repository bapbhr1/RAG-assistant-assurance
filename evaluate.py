# Évaluation hors ligne du pipeline RAG.
# Retrieval et génération sont mesurés séparément : le retrieval décide de ce que
# le modèle peut inventer, la génération vérifie la fidélité une fois les bonnes
# clauses fournies. Le retrieval ne demande pas de clé API, la génération oui.
#
#   python evaluate.py                       # retrieval + génération si une clé Groq est trouvée
#   python evaluate.py --no-llm              # retrieval seul
#   python evaluate.py --top-k 5             # profondeur de recherche
#   python evaluate.py --compare-retrieval   # compare les méthodes de fusion
#   python evaluate.py --report eval_report.md

from __future__ import annotations

import argparse
import json
import os
import statistics
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from src import tracing
from src.chunking import build_corpus, load_contracts
from src.models import CoverageStatus, RAGQuery
from src.rag import SOURCE_SCORE_FLOOR, SOURCE_SCORE_RANGE, MIN_SOURCE_CORRESPONDENCE
from src.retriever import FUSIONS, HybridRetriever

ROOT = Path(__file__).parent
CONTRACTS_PATH = ROOT / "data" / "synthetic_contracts.json"
EVAL_PATH = ROOT / "data" / "eval_questions.json"
SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"

STATUS_ORDER = [
    CoverageStatus.COVERED,
    CoverageStatus.PARTIAL,
    CoverageStatus.NOT_COVERED,
    CoverageStatus.NEEDS_REVIEW,
]
STATUS_SHORT = {
    CoverageStatus.COVERED: "Pris en charge",
    CoverageStatus.PARTIAL: "Partiel",
    CoverageStatus.NOT_COVERED: "Non pris en charge",
    CoverageStatus.NEEDS_REVIEW: "À vérifier",
}
FUSION_LABELS = {
    "ponderee": "Pondérée dense + lexical (défaut)",
    "dense": "Dense seule",
    "bm25": "BM25 seul",
    "rrf": "RRF dense + BM25",
}


@dataclass
class RetrievalMetrics:
    evaluated: int = 0
    hit_rate: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    misses: list[str] = field(default_factory=list)
    # score de la meilleure clause, pour caler le seuil de correspondance
    top_scores_in_corpus: list[float] = field(default_factory=list)
    top_scores_out_of_corpus: list[float] = field(default_factory=list)


@dataclass
class GenerationMetrics:
    status_accuracy: float = 0.0
    false_coverage_rate: float = 0.0
    abstention_recall: float = 0.0
    needless_escalation_rate: float = 0.0
    citation_rate: float = 0.0
    faithfulness: float = 0.0
    escalation_rate: float = 0.0
    confusion: dict[tuple[CoverageStatus, CoverageStatus], int] = field(default_factory=dict)
    mistakes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _load_eval() -> list[dict]:
    with EVAL_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def _api_key() -> str | None:
    # variable d'environnement d'abord, puis secrets Streamlit comme pour l'application
    secrets: dict = {}
    if SECRETS_PATH.exists():
        with SECRETS_PATH.open("rb") as stream:
            secrets = tomllib.load(stream)
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        if name in secrets:
            os.environ.setdefault(name, str(secrets[name]))
    return os.environ.get("GROQ_API_KEY") or secrets.get("GROQ_API_KEY")


def _relevant_keys(item: dict) -> set[tuple[str, str]]:
    return {(ref["contract_id"], ref["article_id"]) for ref in item["relevant"]}


def evaluate_retrieval(
    retriever: HybridRetriever, questions: list[dict], top_k: int
) -> RetrievalMetrics:
    hits = 0
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    misses: list[str] = []
    in_corpus: list[float] = []
    out_of_corpus: list[float] = []

    for item in questions:
        expected = _relevant_keys(item)
        retrieved = retriever.search(item["question"], top_k, item["product_line"])
        top_score = retrieved[0].score if retrieved else 0.0
        # les questions hors corpus n'ont pas de clause attendue : elles ne
        # comptent que pour la calibration du seuil
        if not expected:
            out_of_corpus.append(top_score)
            continue
        in_corpus.append(top_score)
        ranked = [(r.chunk.contract_id, r.chunk.article_id) for r in retrieved]

        found = [rank for rank, key in enumerate(ranked, start=1) if key in expected]
        if found:
            hits += 1
            reciprocal_ranks.append(1.0 / found[0])
        else:
            reciprocal_ranks.append(0.0)
            misses.append(item["id"])

        covered = len(expected & set(ranked))
        recalls.append(covered / len(expected))

    total = len(recalls)
    return RetrievalMetrics(
        evaluated=total,
        hit_rate=hits / total,
        recall=statistics.fmean(recalls),
        mrr=statistics.fmean(reciprocal_ranks),
        misses=misses,
        top_scores_in_corpus=in_corpus,
        top_scores_out_of_corpus=out_of_corpus,
    )


def compare_retrieval(
    retriever: HybridRetriever, questions: list[dict], top_k: int
) -> dict[str, RetrievalMetrics]:
    default = retriever.fusion
    results = {}
    for fusion in FUSIONS:
        retriever.fusion = fusion
        results[fusion] = evaluate_retrieval(retriever, questions, top_k)
    retriever.fusion = default
    return results


def _is_false_coverage(expected: CoverageStatus, obtained: CoverageStatus) -> bool:
    # erreur coûteuse : accorder une garantie (même partielle) que l'attendu refuse,
    # réserve au gestionnaire, ou limite
    granted = {CoverageStatus.COVERED, CoverageStatus.PARTIAL}
    if expected in (CoverageStatus.NOT_COVERED, CoverageStatus.NEEDS_REVIEW):
        return obtained in granted
    if expected == CoverageStatus.PARTIAL:
        return obtained == CoverageStatus.COVERED
    return False


def evaluate_generation(
    retriever: HybridRetriever, questions: list[dict], top_k: int, api_key: str
) -> GenerationMetrics:
    from src.rag import RAGEngine

    engine = RAGEngine(retriever, api_key=api_key)

    correct_status = 0
    with_citation = 0
    escalated = 0
    false_coverage = 0
    at_risk = 0
    abstained_when_expected = 0
    expected_review = 0
    needless_escalations = 0
    decidable = 0
    faithfulness_scores: list[float] = []
    confusion: dict[tuple[CoverageStatus, CoverageStatus], int] = {}
    mistakes: list[str] = []
    errors: list[str] = []

    for item in questions:
        try:
            response = engine.answer(
                RAGQuery(
                    question=item["question"],
                    product_line=item["product_line"],
                    top_k=top_k,
                )
            )
        except Exception as exc:  # pragma: no cover
            errors.append(f"{item['id']} : {exc}")
            continue

        expected = CoverageStatus(item["expected_status"])
        obtained = response.status
        confusion[(expected, obtained)] = confusion.get((expected, obtained), 0) + 1
        if obtained == expected:
            correct_status += 1
        else:
            mistakes.append(
                f"{item['id']} : attendu {STATUS_SHORT[expected]}, obtenu {STATUS_SHORT[obtained]}"
            )
        if obtained == CoverageStatus.NEEDS_REVIEW:
            escalated += 1
        if expected != CoverageStatus.COVERED:
            at_risk += 1
            false_coverage += _is_false_coverage(expected, obtained)
        if expected == CoverageStatus.NEEDS_REVIEW:
            expected_review += 1
            abstained_when_expected += obtained == CoverageStatus.NEEDS_REVIEW
        else:
            decidable += 1
            needless_escalations += obtained == CoverageStatus.NEEDS_REVIEW
        if response.citations and expected != CoverageStatus.NEEDS_REVIEW:
            with_citation += 1
        if response.evidence_checks:
            supported = sum(check.supported for check in response.evidence_checks)
            faithfulness_scores.append(supported / len(response.evidence_checks))

    total = len(questions)
    return GenerationMetrics(
        status_accuracy=correct_status / total,
        false_coverage_rate=false_coverage / at_risk if at_risk else 0.0,
        abstention_recall=abstained_when_expected / expected_review if expected_review else 0.0,
        needless_escalation_rate=needless_escalations / decidable if decidable else 0.0,
        citation_rate=with_citation / decidable if decidable else 0.0,
        faithfulness=statistics.fmean(faithfulness_scores) if faithfulness_scores else 0.0,
        escalation_rate=escalated / total,
        confusion=confusion,
        mistakes=mistakes,
        errors=errors,
    )


def _score_range(scores: list[float]) -> str:
    if not scores:
        return "-"
    return f"{min(scores):.2f} / {statistics.median(scores):.2f} / {max(scores):.2f}"


def _format_report(
    top_k: int,
    questions: list[dict],
    retrieval: RetrievalMetrics,
    comparison: dict[str, RetrievalMetrics] | None,
    generation: GenerationMetrics | None,
) -> str:
    counts = {
        status: sum(item["expected_status"] == status.value for item in questions)
        for status in STATUS_ORDER
    }
    lines = [
        "# Rapport d'évaluation RAG",
        "",
        f"- Questions annotées : **{len(questions)}** "
        f"({counts[CoverageStatus.COVERED]} prises en charge, "
        f"{counts[CoverageStatus.PARTIAL]} partielles, "
        f"{counts[CoverageStatus.NOT_COVERED]} refus, "
        f"{counts[CoverageStatus.NEEDS_REVIEW]} à vérifier dont "
        f"{len(retrieval.top_scores_out_of_corpus)} hors corpus)",
        f"- Profondeur de recherche : **top-{top_k}**",
        "",
        "## Retrieval (sans appel modèle)",
        "",
        f"Mesuré sur les {retrieval.evaluated} questions qui ont au moins une clause attendue.",
        "",
        "| Métrique | Valeur | Lecture |",
        "| --- | --- | --- |",
        f"| Hit-rate@{top_k} | {retrieval.hit_rate:.0%} | au moins une clause attendue est remontée |",
        f"| Recall@{top_k} | {retrieval.recall:.0%} | proportion des clauses attendues remontées |",
        f"| MRR | {retrieval.mrr:.2f} | rang moyen de la première clause pertinente |",
    ]
    if retrieval.misses:
        lines += ["", f"Questions sans clause pertinente dans le top-{top_k} : "
                  + ", ".join(retrieval.misses) + "."]

    floor = SOURCE_SCORE_FLOOR + MIN_SOURCE_CORRESPONDENCE * SOURCE_SCORE_RANGE
    lines += [
        "",
        "### Score de la meilleure clause",
        "",
        "| Questions | min / médiane / max |",
        "| --- | --- |",
        f"| Avec clause attendue | {_score_range(retrieval.top_scores_in_corpus)} |",
        f"| Hors corpus | {_score_range(retrieval.top_scores_out_of_corpus)} |",
        "",
        f"En dessous de {floor:.2f}, la réponse part en validation humaine.",
    ]

    if comparison is not None:
        lines += [
            "",
            "### Comparaison des méthodes de fusion",
            "",
            f"| Fusion | Hit-rate@{top_k} | Recall@{top_k} | MRR |",
            "| --- | --- | --- | --- |",
        ]
        for fusion, metrics in comparison.items():
            lines.append(
                f"| {FUSION_LABELS[fusion]} | {metrics.hit_rate:.0%} | "
                f"{metrics.recall:.0%} | {metrics.mrr:.2f} |"
            )

    if generation is not None:
        lines += [
            "",
            "## Génération (avec contrôles déterministes)",
            "",
            "| Métrique | Valeur | Lecture |",
            "| --- | --- | --- |",
            f"| Exactitude du statut | {generation.status_accuracy:.0%} | décision conforme à l'attendu |",
            f"| Fausse prise en charge | {generation.false_coverage_rate:.0%} | garantie accordée à tort, parmi les questions qui ne sont pas des prises en charge complètes |",
            f"| Abstention correcte | {generation.abstention_recall:.0%} | questions à vérifier bien renvoyées vers un gestionnaire |",
            f"| Escalade inutile | {generation.needless_escalation_rate:.0%} | questions tranchables renvoyées vers un gestionnaire |",
            f"| Taux de citations | {generation.citation_rate:.0%} | questions tranchables appuyées par ≥ 1 extrait vérifié |",
            f"| Fidélité | {generation.faithfulness:.0%} | affirmations retrouvées littéralement, montants compris |",
            f"| Taux d'escalade | {generation.escalation_rate:.0%} | réponses renvoyées vers un gestionnaire |",
            "",
            "### Matrice de confusion",
            "",
            "| Attendu \\ Obtenu | " + " | ".join(STATUS_SHORT[s] for s in STATUS_ORDER) + " |",
            "| --- | " + " | ".join("---" for _ in STATUS_ORDER) + " |",
        ]
        for expected in STATUS_ORDER:
            cells = [str(generation.confusion.get((expected, obtained), 0)) for obtained in STATUS_ORDER]
            lines.append(f"| {STATUS_SHORT[expected]} | " + " | ".join(cells) + " |")
        if generation.mistakes:
            lines += ["", "Écarts avec l'attendu :", ""]
            lines += [f"- {mistake}" for mistake in generation.mistakes]
        if generation.errors:
            lines += ["", "Erreurs de génération :", ""]
            lines += [f"- {error}" for error in generation.errors]
    else:
        lines += [
            "",
            "## Génération",
            "",
            "_Non évaluée (aucune clé `GROQ_API_KEY` ou option `--no-llm`)._",
        ]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation hors ligne du pipeline RAG.")
    parser.add_argument("--top-k", type=int, default=5, help="Profondeur de recherche (défaut : 5).")
    parser.add_argument("--no-llm", action="store_true", help="Évaluer uniquement le retrieval.")
    parser.add_argument(
        "--compare-retrieval", action="store_true", help="Comparer les méthodes de fusion."
    )
    parser.add_argument("--report", type=Path, help="Chemin d'un rapport Markdown à écrire.")
    args = parser.parse_args()

    questions = _load_eval()
    retriever = HybridRetriever()
    retriever.index(build_corpus(load_contracts(CONTRACTS_PATH)))

    retrieval = evaluate_retrieval(retriever, questions, args.top_k)
    comparison = (
        compare_retrieval(retriever, questions, args.top_k)
        if args.compare_retrieval
        else None
    )

    generation: GenerationMetrics | None = None
    api_key = _api_key()
    if not args.no_llm and api_key:
        generation = evaluate_generation(retriever, questions, args.top_k, api_key)
        tracing.flush()

    report = _format_report(args.top_k, questions, retrieval, comparison, generation)
    print(report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
        print(f"Rapport écrit dans {args.report}.")


if __name__ == "__main__":
    main()
