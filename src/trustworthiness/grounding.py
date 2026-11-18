"""
Faithfulness, Grounding, Hallucination Detection & Uncertainty Calibration.

Provides:
1. Retrieval with exact cosine relevance scoring & optional chapter metadata filtering.
2. Strict Grounding Prompt engineered to enforce inline citations ([S1], [S2], ...)
   and deterministic abstention when textbook evidence is absent.
3. Pre-generation Abstention Gate (prevents hallucinations on out-of-context queries).
4. Post-generation Claim-Level Faithfulness & Citation Integrity Verification using
   local sentence embeddings + lexical overlap (zero extra LLM cost/latency).
5. Composite Trust Score (0.0 - 1.0) and calibrated High/Medium/Low Trust badges.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Strict Grounded System Prompt
# ---------------------------------------------------------------------------
GROUNDED_SYSTEM_PROMPT = """You are StudyPal, a trustworthy AI study assistant.

STRICT GROUNDING RULES:
1. Answer ONLY using the numbered textbook excerpts ([S1], [S2], ...) provided in the Context section below.
2. Cite the supporting excerpt inline using [S1], [S2], etc. at the end of every factual claim or sentence.
3. NEVER fabricate citations, page numbers, definitions, or facts that are not supported by the Context.
4. If the Context does not contain enough information to answer the question, explicitly say:
   "I cannot answer this reliably from the provided textbook excerpts."
   and briefly state what topic IS covered in the retrieved excerpts (if any).
5. If the Context only partially answers the question, answer only the supported part with citations and clearly note what is missing.
6. Maintain a clear, neutral, unbiased, student-friendly academic tone.
"""

TRANSPARENT_FALLBACK_SYSTEM_PROMPT = """You are StudyPal, a trustworthy and helpful AI study assistant.

TRANSPARENT FALLBACK RULES:
1. The student's question was not strongly matched in the currently indexed textbook PDF excerpts.
2. Provide a clear, accurate, well-structured educational explanation using established academic knowledge for the selected subject.
3. DO NOT fabricate or invent [S1], [S2], or textbook page citations for facts that do not appear in the provided excerpts.
4. If any provided textbook excerpt partially relates to the question, you may cite that specific excerpt accurately; otherwise answer clearly without fake citations.
5. Keep the explanation student-friendly, neutral, and factually careful — explicitly note any caveats or areas where the student should double-check their syllabus textbook.
"""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class ClaimVerification:
    sentence: str
    status: str  # "Supported" | "Partially Supported" | "Unsupported"
    semantic_score: float
    lexical_score: float
    combined_score: float
    best_source_id: Optional[str] = None
    cited_ids: List[str] = field(default_factory=list)


@dataclass
class TrustMetrics:
    retrieval_score: float
    faithfulness_score: float
    citation_score: float
    safety_score: float
    composite_trust_score: float
    trust_level: str  # "High Trust" | "Medium Trust" | "Low Trust" | "Abstained" | "Blocked"
    supported_claims: int
    partial_claims: int
    unsupported_claims: int
    total_claims: int
    invalid_citations: List[str] = field(default_factory=list)
    uncertainty_warning: str = ""
    claims: List[ClaimVerification] = field(default_factory=list)


@dataclass
class GroundedResponse:
    answer: str
    abstained: bool
    sources: List[Dict[str, Any]]
    metrics: TrustMetrics
    decision_trace: List[Dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Vector Math & Text Helpers
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "not", "with",
    "by", "from", "as", "it", "its", "this", "that", "these", "those",
    "can", "may", "will", "would", "could", "should", "has", "have", "had",
    "do", "does", "did", "into", "also", "such", "which", "who", "whom",
    "what", "where", "when", "why", "how", "more", "most", "very", "than",
    "then", "there", "their", "they", "them", "we", "our", "you", "your",
    "about", "between", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "off", "over", "under", "again", "further",
    "once", "here", "all", "any", "both", "each", "few", "other", "some",
    "only", "own", "same", "so", "too", "just", "because", "while", "if",
}

_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")

_ABSTENTION_PHRASES = (
    "cannot answer this reliably from the provided textbook",
    "not enough information in the provided textbook",
    "does not contain enough information",
    "not covered in the provided textbook",
    "i cannot find sufficient evidence",
    "insufficient textbook evidence",
)


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute bounded cosine similarity in [0.0, 1.0] for two embedding vectors."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    sim = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, sim))


def _content_tokens(text: str) -> set:
    clean = _CITATION_PATTERN.sub("", text.lower())
    tokens = re.findall(r"[a-z0-9]{3,}", clean)
    return {t for t in tokens if t not in _STOPWORDS}


def lexical_overlap_score(sentence: str, chunk_text: str) -> float:
    """Compute proportion of sentence content tokens grounded in a source chunk."""
    sent_tokens = _content_tokens(sentence)
    if not sent_tokens:
        return 1.0
    chunk_tokens = _content_tokens(chunk_text)
    if not chunk_tokens:
        return 0.0
    overlap = len(sent_tokens & chunk_tokens)
    return overlap / len(sent_tokens)


def is_abstention_text(text: str) -> bool:
    lower = (text or "").lower()
    return any(phrase in lower for phrase in _ABSTENTION_PHRASES)


# ---------------------------------------------------------------------------
# Retrieval with Exact Cosine Relevance Scores
# ---------------------------------------------------------------------------
def retrieve_with_scores(
    vectorstore: Any,
    embeddings: Any,
    query: str,
    k: int = 4,
    chapter_filter: Optional[str] = None,
) -> List[Tuple[Any, float]]:
    """
    Retrieve top-k chunks from ChromaDB and compute exact cosine similarity
    in [0.0, 1.0] against the query embedding so scores are calibrated and
    independent of vectorstore distance conventions.
    """
    fetch_k = max(k * 2, 6)
    search_kwargs: Dict[str, Any] = {"k": fetch_k}
    if chapter_filter and chapter_filter != "All Chapters":
        search_kwargs["filter"] = {"chapter": chapter_filter}

    try:
        docs = vectorstore.similarity_search(query, **search_kwargs)
    except Exception:
        # Fallback without filter if legacy DB lacks chapter metadata
        docs = vectorstore.similarity_search(query, k=fetch_k)

    if not docs:
        return []

    query_vec = embeddings.embed_query(query)
    doc_texts = [d.page_content for d in docs]
    doc_vecs = embeddings.embed_documents(doc_texts)

    scored: List[Tuple[Any, float]] = []
    seen_snippets = set()
    for doc, vec in zip(docs, doc_vecs):
        # Deduplicate near-identical chunks
        fingerprint = re.sub(r"\s+", " ", doc.page_content.strip())[:160]
        if fingerprint in seen_snippets:
            continue
        seen_snippets.add(fingerprint)
        sim = cosine_similarity(query_vec, vec)
        scored.append((doc, round(sim, 4)))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def build_grounded_context(scored_docs: List[Tuple[Any, float]]) -> str:
    """Format retrieved chunks as numbered [S1], [S2], ... excerpts with provenance."""
    blocks: List[str] = []
    for idx, (doc, score) in enumerate(scored_docs, start=1):
        meta = getattr(doc, "metadata", {}) or {}
        source_file = meta.get("source_file") or meta.get("source") or "Textbook"
        chapter = meta.get("chapter", "")
        page = meta.get("page", "N/A")
        header = f"[S{idx}] File: {source_file}"
        if chapter:
            header += f" | Chapter: {chapter}"
        header += f" | Page: {page} | Relevance: {score:.2f}"
        blocks.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def build_abstention_response(
    query: str,
    subject: str,
    chapter: str,
    best_score: float,
    threshold: float,
    scored_docs: List[Tuple[Any, float]],
) -> Tuple[str, TrustMetrics]:
    """Deterministic abstention when textbook evidence is below threshold."""
    if not scored_docs:
        reason_detail = (
            f"No indexed textbook chunks were found for **{subject}** ({chapter}). "
            "Please run the vectorization script to index the textbook PDFs."
        )
    else:
        reason_detail = (
            f"The highest textbook similarity score was **{best_score:.0%}**, which is below "
            f"the Trustworthiness Grounding Threshold (**{threshold:.0%}**)."
        )

    answer = (
        "### 🛡️ Grounding Guard: Safe Abstention\n\n"
        "I cannot answer this question reliably from the provided textbook excerpts.\n\n"
        f"- **Selected Scope:** {subject} — *{chapter}*\n"
        f"- **Why I Abstained:** {reason_detail}\n"
        "- **Recommendation:** Ask a question covered in the selected chapter, switch to "
        "**All Chapters**, or verify your question topic against the textbook syllabus."
    )

    metrics = TrustMetrics(
        retrieval_score=round(best_score, 4),
        faithfulness_score=1.0,  # Honest abstention introduces zero hallucinated claims
        citation_score=1.0,
        safety_score=1.0,
        composite_trust_score=round(best_score, 4),
        trust_level="Abstained",
        supported_claims=0,
        partial_claims=0,
        unsupported_claims=0,
        total_claims=0,
        uncertainty_warning=(
            "StudyPal abstained rather than risking hallucination because retrieved textbook "
            "evidence did not meet the minimum relevance threshold."
        ),
        claims=[],
    )
    return answer, metrics


# ---------------------------------------------------------------------------
# Post-Generation Faithfulness & Citation Verification
# ---------------------------------------------------------------------------
def split_into_claims(answer: str) -> List[str]:
    """Split an LLM response into atomic factual sentences/bullets for verification."""
    if not answer:
        return []

    raw_lines = answer.splitlines()
    claims: List[str] = []

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip markdown headers & dividers
        if stripped.startswith("#") or stripped.startswith("---"):
            continue
        # Remove leading markdown bullet / numbering
        stripped = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", stripped)
        # Split into sentences while preserving abbreviations/decimals
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[*])", stripped)
        for s in sentences:
            s = s.strip()
            # Filter out tiny non-factual fragments (e.g., "Summary:", "Note:")
            plain = _CITATION_PATTERN.sub("", s).strip(" *:-_.")
            if len(plain.split()) >= 5:
                claims.append(s)

    return claims


def verify_citations(answer: str, num_sources: int) -> Tuple[float, List[str], List[str]]:
    """
    Check citation validity & coverage:
    - Detects hallucinated/out-of-range citation IDs (e.g. [S9] when only 4 sources exist).
    - Computes citation coverage across factual claims.
    Returns (citation_score, valid_cited_ids, invalid_cited_ids).
    """
    found_nums = [int(m) for m in _CITATION_PATTERN.findall(answer or "")]
    valid_ids: List[str] = []
    invalid_ids: List[str] = []

    for n in found_nums:
        sid = f"S{n}"
        if 1 <= n <= num_sources:
            if sid not in valid_ids:
                valid_ids.append(sid)
        else:
            if sid not in invalid_ids:
                invalid_ids.append(sid)

    claims = split_into_claims(answer)
    if not claims or num_sources == 0:
        integrity = 0.0 if invalid_ids else 1.0
        return integrity, valid_ids, invalid_ids

    cited_claims = sum(1 for c in claims if _CITATION_PATTERN.search(c))
    coverage = cited_claims / len(claims)

    # Penalize any fabricated citation ID heavily
    validity = 1.0 if not found_nums else (
        len([n for n in found_nums if 1 <= n <= num_sources]) / len(found_nums)
    )
    if invalid_ids:
        score = max(0.0, 0.4 * coverage + 0.6 * validity - 0.25 * len(invalid_ids))
    else:
        # Reward good citation coverage while allowing summary transition sentences
        score = min(1.0, 0.35 + 0.65 * coverage) if valid_ids else 0.25

    return round(score, 4), valid_ids, invalid_ids


def evaluate_faithfulness(
    answer: str,
    source_texts: List[str],
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    supported_threshold: float = 0.46,
    partial_threshold: float = 0.33,
) -> Tuple[float, List[ClaimVerification]]:
    """
    Verify each sentence/claim in the LLM answer against the retrieved textbook chunks.

    Uses hybrid semantic similarity (local sentence-transformer cosine similarity)
    plus lexical content-word overlap. Works even if embed_fn is None (falls back
    gracefully to lexical overlap for fast unit testing).
    """
    claims = split_into_claims(answer)
    if not claims:
        return 1.0, []

    if not source_texts:
        verifications = [
            ClaimVerification(
                sentence=c,
                status="Unsupported",
                semantic_score=0.0,
                lexical_score=0.0,
                combined_score=0.0,
                best_source_id=None,
                cited_ids=[f"S{n}" for n in _CITATION_PATTERN.findall(c)],
            )
            for c in claims
        ]
        return 0.0, verifications

    claim_vecs: Optional[List[List[float]]] = None
    source_vecs: Optional[List[List[float]]] = None
    if embed_fn is not None:
        clean_claims = [_CITATION_PATTERN.sub("", c).strip() for c in claims]
        all_vecs = embed_fn(clean_claims + source_texts)
        claim_vecs = all_vecs[: len(claims)]
        source_vecs = all_vecs[len(claims) :]

    verifications: List[ClaimVerification] = []
    for idx, claim in enumerate(claims):
        cited = [f"S{n}" for n in _CITATION_PATTERN.findall(claim)]
        best_combined = -1.0
        best_sem = 0.0
        best_lex = 0.0
        best_sid = "S1"

        for s_idx, src_text in enumerate(source_texts, start=1):
            lex = lexical_overlap_score(claim, src_text)
            if claim_vecs is not None and source_vecs is not None:
                sem = cosine_similarity(claim_vecs[idx], source_vecs[s_idx - 1])
                combined = 0.72 * sem + 0.28 * lex
            else:
                sem = lex
                combined = lex

            if combined > best_combined:
                best_combined = combined
                best_sem = sem
                best_lex = lex
                best_sid = f"S{s_idx}"

        if best_combined >= supported_threshold:
            status = "Supported"
        elif best_combined >= partial_threshold:
            status = "Partially Supported"
        else:
            status = "Unsupported"

        verifications.append(
            ClaimVerification(
                sentence=claim,
                status=status,
                semantic_score=round(best_sem, 4),
                lexical_score=round(best_lex, 4),
                combined_score=round(best_combined, 4),
                best_source_id=best_sid,
                cited_ids=cited,
            )
        )

    # Faithfulness score combines support ratio and average grounding alignment
    weighted_support = sum(
        1.0 if v.status == "Supported" else (0.55 if v.status == "Partially Supported" else 0.0)
        for v in verifications
    ) / len(verifications)
    mean_alignment = sum(min(1.0, v.combined_score / 0.70) for v in verifications) / len(
        verifications
    )
    faithfulness = round(max(0.0, min(1.0, 0.65 * weighted_support + 0.35 * mean_alignment)), 4)

    return faithfulness, verifications


def compute_trust_metrics(
    *,
    answer: str,
    scored_docs: List[Tuple[Any, float]],
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    safety_passed: bool = True,
) -> TrustMetrics:
    """
    Compute the Composite Trust Score and calibrated Trust Level:

        Composite Trust = 0.30 * Retrieval_Relevance
                        + 0.40 * Faithfulness
                        + 0.20 * Citation_Integrity
                        + 0.10 * Safety_Compliance
    """
    if scored_docs:
        # Weight top retrieved chunk higher since primary answer often comes from top 1-2 chunks
        scores = [s for _, s in scored_docs]
        weights = [1.0 / math.log2(i + 2) for i in range(len(scores))]
        retrieval_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    else:
        retrieval_score = 0.0

    source_texts = [d.page_content for d, _ in scored_docs]
    citation_score, _valid_ids, invalid_ids = verify_citations(answer, len(source_texts))

    if is_abstention_text(answer):
        return TrustMetrics(
            retrieval_score=round(retrieval_score, 4),
            faithfulness_score=1.0,
            citation_score=1.0 if not invalid_ids else citation_score,
            safety_score=1.0 if safety_passed else 0.0,
            composite_trust_score=round(min(0.65, retrieval_score + 0.20), 4),
            trust_level="Abstained",
            supported_claims=0,
            partial_claims=0,
            unsupported_claims=0,
            total_claims=0,
            invalid_citations=invalid_ids,
            uncertainty_warning=(
                "The assistant explicitly abstained because the retrieved textbook excerpts "
                "did not sufficiently support an answer."
            ),
            claims=[],
        )

    faithfulness_score, claims = evaluate_faithfulness(
        answer=answer,
        source_texts=source_texts,
        embed_fn=embed_fn,
    )

    safety_score = 1.0 if safety_passed else 0.0

    # Normalize retrieval score smoothly (in MiniLM cosine space, 0.72+ is very strong)
    calibrated_retrieval = min(1.0, retrieval_score / 0.72) if retrieval_score > 0 else 0.0

    composite = (
        0.30 * calibrated_retrieval
        + 0.40 * faithfulness_score
        + 0.20 * citation_score
        + 0.10 * safety_score
    )
    if invalid_ids:
        composite = max(0.0, composite - 0.15)
    composite = round(max(0.0, min(1.0, composite)), 4)

    supported_n = sum(1 for c in claims if c.status == "Supported")
    partial_n = sum(1 for c in claims if c.status == "Partially Supported")
    unsupported_n = sum(1 for c in claims if c.status == "Unsupported")

    if not safety_passed:
        level = "Blocked"
    elif composite >= 0.75 and unsupported_n == 0 and not invalid_ids:
        level = "High Trust"
    elif composite >= 0.50:
        level = "Medium Trust"
    else:
        level = "Low Trust"

    warnings: List[str] = []
    if invalid_ids:
        warnings.append(
            f"Fabricated or out-of-range citation(s) detected: {', '.join(invalid_ids)}."
        )
    if unsupported_n > 0:
        warnings.append(
            f"{unsupported_n} of {len(claims)} sentence(s) had low grounding against retrieved excerpts — verify with the textbook sources below."
        )
    if retrieval_score < 0.40:
        warnings.append(
            f"Borderline textbook retrieval similarity ({retrieval_score:.0%}); double-check chapter selection."
        )

    return TrustMetrics(
        retrieval_score=round(retrieval_score, 4),
        faithfulness_score=faithfulness_score,
        citation_score=citation_score,
        safety_score=safety_score,
        composite_trust_score=composite,
        trust_level=level,
        supported_claims=supported_n,
        partial_claims=partial_n,
        unsupported_claims=unsupported_n,
        total_claims=len(claims),
        invalid_citations=invalid_ids,
        uncertainty_warning=" ".join(warnings),
        claims=claims,
    )


def build_fallback_trust_metrics(
    *,
    answer: str,
    scored_docs: List[Tuple[Any, float]],
    threshold: float,
    safety_passed: bool = True,
) -> TrustMetrics:
    """
    Build epistemically honest TrustMetrics for Transparent General-Knowledge Fallback.
    Never inflates textbook faithfulness or allows fabricated [S#] citations.
    """
    top_score = scored_docs[0][1] if scored_docs else 0.0
    _, valid_ids, invalid_ids = verify_citations(answer, len(scored_docs))
    # In general fallback mode, avoiding fabricated citations is honest behavior
    citation_integrity = 0.0 if invalid_ids else 1.0
    safety_score = 1.0 if safety_passed else 0.0

    # Transparently calibrated moderate score reflecting general-knowledge (non-textbook) basis
    composite = round(
        max(0.0, min(0.72, 0.35 + 0.30 * top_score + 0.15 * safety_score - 0.25 * len(invalid_ids))),
        4,
    )

    warning = (
        f"Transparent General-Knowledge Fallback: highest textbook similarity was {top_score:.0%} "
        f"(below your {threshold:.0%} grounding threshold). Answer is generated from general academic "
        "knowledge rather than your indexed PDF pages — please cross-check with your full textbook."
    )
    if invalid_ids:
        warning += f" Invalid citation(s) flagged: {', '.join(invalid_ids)}."

    return TrustMetrics(
        retrieval_score=round(top_score, 4),
        faithfulness_score=round(top_score, 4),
        citation_score=citation_integrity,
        safety_score=safety_score,
        composite_trust_score=composite,
        trust_level="Blocked" if not safety_passed else "General Knowledge (Verify)",
        supported_claims=0,
        partial_claims=0,
        unsupported_claims=0,
        total_claims=0,
        invalid_citations=invalid_ids,
        uncertainty_warning=warning,
        claims=[],
    )
