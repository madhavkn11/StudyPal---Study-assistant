"""
Trustworthy AI package for StudyPal.

Implements end-to-end trustworthiness aligned with NIST AI RMF 1.0 and the
EU High-Level Expert Group on AI (7 Key Requirements for Trustworthy AI):
- Safety, Security & Privacy Guardrails (guardrails.py)
- Faithfulness, Grounding & Uncertainty Calibration (grounding.py)
- Explainability, Provenance & System Transparency Card (explainability.py)
- Accountability, Audit Logging & Human-in-the-Loop Feedback (audit.py)
- Automated Trustworthiness Evaluation & Red-Team Suite (evaluator.py)
"""

from .guardrails import (
    GuardrailResult,
    PIIRedactionResult,
    OutputAuditResult,
    redact_pii,
    check_prompt_injection,
    check_content_safety,
    validate_user_input,
    audit_llm_output,
    sanitize_search_query,
)
from .grounding import (
    ClaimVerification,
    TrustMetrics,
    GroundedResponse,
    GROUNDED_SYSTEM_PROMPT,
    split_into_claims,
    verify_citations,
    evaluate_faithfulness,
    compute_trust_metrics,
    build_abstention_response,
    build_grounded_context,
    retrieve_with_scores,
)
from .explainability import (
    SourceProvenance,
    format_source_provenance,
    build_decision_trace,
    get_dataset_provenance,
    get_system_card,
)
from .audit import (
    TrustAuditLogger,
    DEFAULT_AUDIT_LOG_PATH,
    DEFAULT_FEEDBACK_LOG_PATH,
)
from .pipeline import (
    PipelineConfig,
    PipelineResult,
    TrustworthyRAGPipeline,
)
from .evaluator import (
    run_trustworthiness_benchmark,
    save_benchmark_report,
)

__all__ = [
    "GuardrailResult",
    "PIIRedactionResult",
    "OutputAuditResult",
    "redact_pii",
    "check_prompt_injection",
    "check_content_safety",
    "validate_user_input",
    "audit_llm_output",
    "sanitize_search_query",
    "ClaimVerification",
    "TrustMetrics",
    "GroundedResponse",
    "GROUNDED_SYSTEM_PROMPT",
    "split_into_claims",
    "verify_citations",
    "evaluate_faithfulness",
    "compute_trust_metrics",
    "build_abstention_response",
    "build_grounded_context",
    "retrieve_with_scores",
    "SourceProvenance",
    "format_source_provenance",
    "build_decision_trace",
    "get_dataset_provenance",
    "get_system_card",
    "TrustAuditLogger",
    "DEFAULT_AUDIT_LOG_PATH",
    "DEFAULT_FEEDBACK_LOG_PATH",
    "PipelineConfig",
    "PipelineResult",
    "TrustworthyRAGPipeline",
    "run_trustworthiness_benchmark",
    "save_benchmark_report",
]
