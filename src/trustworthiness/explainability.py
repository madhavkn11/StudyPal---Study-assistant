"""
Explainability, Source Provenance & Trustworthy AI System Card.

Provides:
- Verbatim source provenance extraction (file, chapter, exact page, similarity %)
- Step-by-step Decision Trace ("Why did StudyPal give this answer?")
- Cryptographic dataset provenance (SHA-256 hashes & page counts of indexed PDFs)
- Comprehensive AI System / Model Card mapped to NIST AI RMF 1.0 and the
  EU High-Level Expert Group 7 Key Requirements for Trustworthy AI.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SourceProvenance:
    source_id: str
    source_file: str
    subject: str
    chapter: str
    page: Any
    chunk_index: Any
    similarity_score: float
    excerpt: str
    file_sha256: str = ""


def format_source_provenance(
    scored_docs: List[Tuple[Any, float]]
) -> List[Dict[str, Any]]:
    """Convert retrieved (Document, similarity) pairs into transparent citation objects."""
    provenance: List[Dict[str, Any]] = []
    for idx, (doc, score) in enumerate(scored_docs, start=1):
        meta = getattr(doc, "metadata", {}) or {}
        raw_source = meta.get("source_file") or meta.get("source") or "Unknown Textbook"
        source_file = os.path.basename(str(raw_source))
        page = meta.get("page", "N/A")
        chapter = meta.get("chapter") or os.path.splitext(source_file)[0]
        provenance.append(
            {
                "source_id": f"S{idx}",
                "source_file": source_file,
                "subject": meta.get("subject", "Textbook"),
                "chapter": chapter,
                "page": page,
                "chunk_index": meta.get("chunk_index", idx - 1),
                "similarity_score": round(float(score), 4),
                "excerpt": doc.page_content.strip(),
                "file_sha256": meta.get("file_sha256", ""),
            }
        )
    return provenance


def build_decision_trace(
    *,
    pii_redacted_count: int,
    pii_entities: List[str],
    guardrail_allowed: bool,
    guardrail_category: str,
    retrieved_count: int,
    top_retrieval_score: float,
    threshold: float,
    strict_grounding: bool,
    abstained: bool,
    used_general_fallback: bool = False,
    faithfulness_score: Optional[float] = None,
    trust_level: str = "",
) -> List[Dict[str, str]]:
    """Build a human-readable pipeline trace explaining every trustworthiness gate."""
    trace: List[Dict[str, str]] = []

    if pii_redacted_count > 0:
        trace.append(
            {
                "stage": "1. Privacy Shield (PII)",
                "status": "REDACTED",
                "detail": f"Masked {pii_redacted_count} sensitive item(s) ({', '.join(pii_entities)}) locally before external API call.",
            }
        )
    else:
        trace.append(
            {
                "stage": "1. Privacy Shield (PII)",
                "status": "PASS",
                "detail": "No personal identifiable information detected in query.",
            }
        )

    if not guardrail_allowed:
        trace.append(
            {
                "stage": "2. Security & Safety Guardrails",
                "status": "BLOCKED",
                "detail": f"Blocked under category '{guardrail_category}'.",
            }
        )
        return trace

    trace.append(
        {
            "stage": "2. Security & Safety Guardrails",
            "status": "PASS",
            "detail": "No prompt-injection, jailbreak, or unsafe content patterns detected.",
        }
    )

    trace.append(
        {
            "stage": "3. Textbook Semantic Retrieval",
            "status": "OK" if retrieved_count > 0 else "EMPTY",
            "detail": (
                f"Retrieved {retrieved_count} chunk(s); top cosine relevance = {top_retrieval_score:.0%} "
                f"(abstention threshold = {threshold:.0%}, strict mode = {'ON' if strict_grounding else 'OFF'})."
            ),
        }
    )

    if abstained:
        trace.append(
            {
                "stage": "4. Grounding & Abstention Gate",
                "status": "ABSTAINED",
                "detail": "Evidence below threshold in Strict Abstention mode — abstained to prevent hallucination.",
            }
        )
        return trace

    if used_general_fallback:
        trace.append(
            {
                "stage": "4. Epistemic Fallback (General Academic Knowledge)",
                "status": "DISCLOSED",
                "detail": (
                    f"Top textbook similarity ({top_retrieval_score:.0%}) was below threshold ({threshold:.0%}); "
                    "answered via Transparent Fallback without fabricating textbook citations."
                ),
            }
        )
        trace.append(
            {
                "stage": "5. Output Safety & Provenance Disclosure",
                "status": trust_level or "DISCLOSED",
                "detail": "Verified output safety & zero fabricated citations; labeled response for student verification.",
            }
        )
        return trace

    trace.append(
        {
            "stage": "4. Grounded Generation (Groq LLM)",
            "status": "OK",
            "detail": "Generated response constrained strictly to numbered textbook excerpts [S1..Sk] at temperature=0.",
        }
    )

    trace.append(
        {
            "stage": "5. Claim Faithfulness & Citation Audit",
            "status": trust_level or "VERIFIED",
            "detail": (
                f"Post-generation verification scored faithfulness at {(faithfulness_score or 0.0):.0%} "
                f"and calibrated final badge: {trust_level}."
            ),
        }
    )
    return trace


def _sha256_short(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:12]


def get_dataset_provenance(project_root: str) -> List[Dict[str, Any]]:
    """Inspect all source PDF textbooks in data/ and return provenance metadata."""
    data_dir = os.path.join(project_root, "data")
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(data_dir):
        return rows

    try:
        import pypdf
    except Exception:
        pypdf = None

    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(".pdf"):
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, project_root).replace("\\", "/")
            size_kb = round(os.path.getsize(full_path) / 1024, 1)
            pages: Any = "N/A"
            if pypdf is not None:
                try:
                    pages = len(pypdf.PdfReader(full_path).pages)
                except Exception:
                    pages = "N/A"
            subject = (
                "Biology (Class 12)"
                if "biology" in rel_path.lower()
                else ("Operating Systems (B.Tech)" if "operating_systems" in rel_path.lower() else "General")
            )
            rows.append(
                {
                    "subject": subject,
                    "file": fname,
                    "relative_path": rel_path,
                    "pages": pages,
                    "size_kb": size_kb,
                    "sha256_short": _sha256_short(full_path),
                }
            )
    return rows


def get_system_card(project_root: str) -> Dict[str, Any]:
    """Return a structured Trustworthy AI Model & System Card for transparency."""
    return {
        "system_name": "StudyPal Trustworthy RAG Study Assistant",
        "version": "2.0 (Trustworthy AI Edition)",
        "intended_use": (
            "Educational question-answering strictly grounded in student textbooks "
            "(Class 12 Biology & B.Tech Operating Systems) with inline verifiable citations, "
            "confidence calibration, and optional educational YouTube video links."
        ),
        "out_of_scope_use": [
            "Medical, legal, financial, or clinical decision-making.",
            "Answering questions outside the indexed textbook curriculum without verification.",
            "Academic dishonesty or bypassing graded examination policies.",
            "Processing sensitive personal data (PII is automatically redacted locally).",
        ],
        "architecture": {
            "llm": "Groq Cloud — openai/gpt-oss-120b (configurable via GROQ_MODEL, temperature = 0 for deterministic grounding)",
            "embeddings": "Local HuggingFace sentence-transformers/all-MiniLM-L6-v2 (384-dim, runs on CPU)",
            "vector_store": "Local ChromaDB with cosine similarity & chapter/page provenance metadata",
            "video_lookup": "yt-dlp metadata search using PII-sanitized academic queries (optional toggle)",
        },
        "data_flow_privacy": [
            "PDF textbooks, embeddings, ChromaDB vector index, and audit logs remain 100% local.",
            "Privacy Shield scans & redacts emails, phone numbers, SSNs, IDs, credit cards, IPs, and API keys locally BEFORE any text leaves the device.",
            "Only the PII-redacted question and retrieved textbook excerpts are sent to the Groq inference API.",
            "Audit logs store only sanitized queries and aggregate trustworthiness metrics — never raw PII.",
        ],
        "trustworthiness_frameworks": [
            {
                "pillar": "Valid, Reliable & Faithful",
                "nist_rmf": "Valid & Reliable",
                "eu_ai_req": "Technical Robustness & Safety",
                "implementation": (
                    "Strict citation-enforced RAG prompt, deterministic out-of-context abstention gate, "
                    "claim-by-claim semantic + lexical faithfulness verifier, citation validity audit, and Composite Trust Score."
                ),
            },
            {
                "pillar": "Explainable & Transparent",
                "nist_rmf": "Explainable & Interpretable; Accountable & Transparent",
                "eu_ai_req": "Transparency",
                "implementation": (
                    "Inline [S#] citations, verbatim textbook excerpts with exact file/chapter/page/similarity score, "
                    "sentence-level claim attribution table, step-by-step Decision Trace, and this Model & Data Card."
                ),
            },
            {
                "pillar": "Safe, Secure & Resilient",
                "nist_rmf": "Safe; Secure & Resilient",
                "eu_ai_req": "Technical Robustness & Safety",
                "implementation": (
                    "Multi-rule prompt-injection, jailbreak, delimiter-smuggling, and system-prompt-exfiltration defense; "
                    "domain-aware input/output content safety filter; YouTube query sanitization."
                ),
            },
            {
                "pillar": "Privacy-Enhanced",
                "nist_rmf": "Privacy-Enhanced",
                "eu_ai_req": "Privacy & Data Governance",
                "implementation": (
                    "Local PII detection & redaction (with Luhn-validated credit card checks) prior to external API calls "
                    "and audit logging; local vector storage; cryptographic SHA-256 dataset provenance."
                ),
            },
            {
                "pillar": "Fairness & Bias Mitigation",
                "nist_rmf": "Fair with Harmful Bias Managed",
                "eu_ai_req": "Diversity, Non-discrimination & Fairness",
                "implementation": (
                    "Domain-aware guardrails calibrated against false positives on legitimate Biology/OS terminology; "
                    "neutral academic prompting grounded in curriculum textbooks rather than unverified web opinions."
                ),
            },
            {
                "pillar": "Human Agency, Oversight & Accountability",
                "nist_rmf": "Govern & Measure",
                "eu_ai_req": "Human Agency & Oversight; Accountability",
                "implementation": (
                    "User-adjustable grounding strictness & thresholds, per-answer 👍/👎 human feedback loop with error categorization, "
                    "exportable JSONL/CSV audit logs, and built-in Automated Trustworthiness Evaluation Benchmark."
                ),
            },
        ],
        "known_limitations": [
            "Retrieval quality depends on PDF text extraction fidelity (scanned image-only pages or complex equations may lose structure).",
            "Sentence-level embedding faithfulness is an efficient semantic proxy and may occasionally flag valid multi-sentence synthesis as 'Partially Supported' — always consult the cited textbook excerpt.",
            "External LLM availability depends on Groq API connectivity and rate limits.",
        ],
        "datasets": get_dataset_provenance(project_root),
    }
