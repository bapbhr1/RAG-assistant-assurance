# Évaluation hors ligne du pipeline RAG.
# Retrieval et génération sont mesurés séparément : le retrieval décide de ce que
# le modèle peut inventer, la génération vérifie la fidélité une fois les bonnes
# clauses fournies. Le retrieval ne demande pas de clé API, la génération oui.
#
#   python evaluate.py                 # retrieval + génération si GROQ_API_KEY est défini
#   python evaluate.py --no-llm        # retrieval seul
#   python evaluate.py --top-k 5       # profondeur de recherche
#   python evaluate.py --report eval_report.md

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from src.chunking import build_corpus, load_contracts
from src.models import CoverageStatus, RAGQuery
from src.retriever import HybridRetriever

ROOT = Path(__file__).parent
CONTRACTS_PATH = ROOT / "data" / "synthetic_contracts.json"
EVAL_PATH = ROOT / "data" / "eval_questions.json"


@dataclass
class RetrievalMetrics:
    hit_rate: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    misses: list[str] = field(default_factory=list)


@dataclass
class GenerationMetrics:
    status_accuracy: float = 0.0
    citation_rate: float = 0.0
    faithfulness: float = 0.0
    escalation_rate: float = 0.0
    errors: list[str] = field(default_factory=list)


def _load_eval() -> list[dict]:
    with EVAL_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def _relevant_keys(item: dict) -> set[tuple[str, str]]:
    return {(ref["contract_id"], ref["article_id"]) for ref in item["relevant"]}


def evaluate_retrieval(
    retriever: HybridRetriever, questions: list[dict], top_k: int
) -> RetrievalMetrics:
    hits = 0
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    misses: list[str] = []

    for item in questions:
        expected = _relevant_keys(item)
        retrieved = retriever.search(item["question"], top_k, item["product_line"])
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

    total = len(questions)
    return RetrievalMetrics(
        hit_rate=hits / total,
        recall=statistics.fmean(recalls),
        mrr=statistics.fmean(reciprocal_ranks),
        misses=misses,
    )


def evaluate_generation(
    retriever: HybridRetriever, questions: list[dict], top_k: int, api_key: str
) -> GenerationMetrics:
    from src.rag import RAGEngine

    engine = RAGEngine(retriever, api_key=api_key)

    correct_status = 0
    with_citation = 0
    escalated = 0
    faithfulness_scores: list[float] = []
    errors: list[str] = []

    for item in questions:
        try:
            response = engine.answer(
                RAGQuery(question=item["question"], product_line=item["product_line"])
            )
        except Exception as exc:  # pragma: no cover
            errors.append(f"{item['id']} : {exc}")
            continue

        expected_status = CoverageStatus(item["expected_status"])
        if response.status == expected_status:
            correct_status += 1
        if response.status == CoverageStatus.NEEDS_REVIEW:
            escalated += 1
        if response.citations:
            with_citation += 1
        if response.evidence_checks:
            supported = sum(check.supported for check in response.evidence_checks)
            faithfulness_scores.append(supported / len(response.evidence_checks))

    total = len(questions)
    return GenerationMetrics(
        status_accuracy=correct_status / total,
        citation_rate=with_citation / total,
        faithfulness=statistics.fmean(faithfulness_scores) if faithfulness_scores else 0.0,
        escalation_rate=escalated / total,
        errors=errors,
    )


def _format_report(
    top_k: int,
    total: int,
    retrieval: RetrievalMetrics,
    generation: GenerationMetrics | None,
) -> str:
    lines = [
        "# Rapport d'évaluation RAG",
        "",
        f"- Questions annotées : **{total}**",
        f"- Profondeur de recherche : **top-{top_k}**",
        "",
        "## Retrieval (sans appel modèle)",
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

    if generation is not None:
        lines += [
            "",
            "## Génération (avec contrôles déterministes)",
            "",
            "| Métrique | Valeur | Lecture |",
            "| --- | --- | --- |",
            f"| Exactitude du statut | {generation.status_accuracy:.0%} | décision conforme à l'attendu |",
            f"| Taux de citations | {generation.citation_rate:.0%} | réponses appuyées par ≥ 1 extrait vérifié |",
            f"| Fidélité | {generation.faithfulness:.0%} | part des affirmations retrouvées littéralement |",
            f"| Taux d'escalade | {generation.escalation_rate:.0%} | réponses renvoyées vers un gestionnaire |",
        ]
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
    parser.add_argument("--report", type=Path, help="Chemin d'un rapport Markdown à écrire.")
    args = parser.parse_args()

    questions = _load_eval()
    retriever = HybridRetriever()
    retriever.index(build_corpus(load_contracts(CONTRACTS_PATH)))

    retrieval = evaluate_retrieval(retriever, questions, args.top_k)

    generation: GenerationMetrics | None = None
    api_key = os.environ.get("GROQ_API_KEY")
    if not args.no_llm and api_key:
        generation = evaluate_generation(retriever, questions, args.top_k, api_key)

    report = _format_report(args.top_k, len(questions), retrieval, generation)
    print(report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
        print(f"Rapport écrit dans {args.report}.")


if __name__ == "__main__":
    main()
