# Modèles Pydantic : corpus contractuel et réponse métier.

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ContractArticle(BaseModel):
    article_id: str
    title: str
    text: str


class Contract(BaseModel):
    contract_id: str
    product: str
    product_line: str
    title: str
    policyholder: str
    articles: list[ContractArticle]


class Chunk(BaseModel):
    chunk_id: str
    contract_id: str
    product: str
    product_line: str
    article_id: str
    article_title: str
    text: str

    def to_context_string(self) -> str:
        return (
            f"[SOURCE {self.chunk_id} | contrat={self.contract_id} | "
            f"produit={self.product} | article={self.article_id} - {self.article_title}]\n"
            f"{self.text}"
        )


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0


class CoverageStatus(str, Enum):
    COVERED = "PRIS_EN_CHARGE"
    NOT_COVERED = "NON_PRIS_EN_CHARGE"
    PARTIAL = "PARTIELLEMENT_PRIS_EN_CHARGE"
    NEEDS_REVIEW = "A_VERIFIER_PAR_GESTIONNAIRE"


class Citation(BaseModel):
    chunk_id: str
    contract_id: str
    article_id: str
    article_title: str
    quote: str


class ClaimEvidence(BaseModel):
    claim: str
    chunk_id: str
    quote: str


class EvidenceCheck(BaseModel):
    claim: str
    supported: bool
    citation: Citation | None = None


class ConversationTurn(BaseModel):
    question: str
    answer: str


class RAGQuery(BaseModel):
    question: str = Field(min_length=3)
    product_line: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


class LLMAnswer(BaseModel):
    status: CoverageStatus
    answer: str
    decision: str
    conditions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    claims: list[ClaimEvidence] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    reasoning_summary: str


class ConfidenceBreakdown(BaseModel):
    factual_support: float = Field(ge=0.0, le=1.0)
    uncertainty_handling: float = Field(ge=0.0, le=1.0)
    source_handling: float = Field(ge=0.0, le=1.0)


class RAGResponse(LLMAnswer):
    citations: list[Citation] = Field(default_factory=list)
    evidence_checks: list[EvidenceCheck] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: ConfidenceBreakdown
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
