"""
End-to-End Trustworthy RAG Orchestration Pipeline for StudyPal.

Coordinates all trustworthiness stages in one clean, auditable flow:
1. Input validation, local PII redaction, prompt-injection & safety guardrails
2. Chapter-scoped semantic retrieval with exact cosine similarity scoring
3. Pre-generation out-of-context abstention gate
4. Grounded LLM synthesis (strict inline [S#] citation enforcement, temp=0)
5. Output safety & prompt-leakage audit
6. Post-generation claim-by-claim faithfulness & citation integrity scoring
7. Explainable decision trace & privacy-preserving JSONL audit logging
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .audit import TrustAuditLogger
from .explainability import build_decision_trace, format_source_provenance
from .grounding import (
    GROUNDED_SYSTEM_PROMPT,
    TrustMetrics,
    build_abstention_response,
    build_grounded_context,
    compute_trust_metrics,
    is_abstention_text,
    retrieve_with_scores,
)
from .guardrails import (
    GuardrailResult,
    audit_llm_output,
    sanitize_search_query,
    validate_user_input,
)


@dataclass
class PipelineConfig:
    strict_grounding: bool = True
    min_retrieval_score: float = 0.30
    top_k: int = 4
    enable_pii_redaction: bool = True
    max_history_turns: int = 4


@dataclass
class PipelineResult:
    interaction_id: str
    decision: str  # "answered" | "abstained" | "blocked_injection" | "blocked_safety" | "blocked_validation" | "error"
    answer: str
    sanitized_query: str
    youtube_query: str
    guardrail: GuardrailResult
    metrics: TrustMetrics
    sources: List[Dict[str, Any]] = field(default_factory=list)
    decision_trace: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    show_videos: bool = False


class TrustworthyRAGPipeline:
    """Self-contained Trustworthy RAG engine used by Streamlit UI and CLI Evaluator."""

    def __init__(
        self,
        *,
        vectorstore: Any,
        embeddings: Any,
        llm: Any,
        subject: str,
        chapter: str = "All Chapters",
        chapter_filter: Optional[str] = None,
        audit_logger: Optional[TrustAuditLogger] = None,
    ) -> None:
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.llm = llm
        self.subject = subject
        self.chapter = chapter
        self.chapter_filter = chapter_filter
        self.audit_logger = audit_logger or TrustAuditLogger()

    def _build_messages(
        self,
        sanitized_query: str,
        context_block: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_turns: int = 4,
    ) -> List[Any]:
        messages: List[Any] = [SystemMessage(content=GROUNDED_SYSTEM_PROMPT)]

        if chat_history:
            recent = chat_history[-(max_turns * 2) :]
            for msg in recent:
                role = msg.get("role")
                content = msg.get("content", "")
                if not content:
                    continue
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content[:1200]))

        user_prompt = (
            f"Subject: {self.subject}\n"
            f"Topic/Chapter Scope: {self.chapter}\n\n"
            f"Context (Numbered Textbook Excerpts):\n"
            f"{context_block}\n\n"
            f"Student Question: {sanitized_query}\n\n"
            "Instructions: Answer clearly and concisely using ONLY the numbered textbook excerpts "
            "above, citing [S1], [S2], etc. inline for every factual statement. If the excerpts do not "
            "support an answer, explicitly abstain."
        )
        messages.append(HumanMessage(content=user_prompt))
        return messages

    def answer_question(
        self,
        raw_query: str,
        *,
        chat_history: Optional[List[Dict[str, str]]] = None,
        config: Optional[PipelineConfig] = None,
        log_to_audit: bool = True,
    ) -> PipelineResult:
        cfg = config or PipelineConfig()
        t0 = time.perf_counter()
        interaction_id = self.audit_logger.new_interaction_id()

        # ------------------------------------------------------------------
        # Stage 1 & 2: Privacy (PII) + Security/Safety Guardrails
        # ------------------------------------------------------------------
        guard = validate_user_input(
            raw_query, enable_pii_redaction=cfg.enable_pii_redaction
        )
        warnings = list(guard.warnings)

        if not guard.allowed:
            decision = f"blocked_{guard.block_category or 'guardrail'}"
            trace = build_decision_trace(
                pii_redacted_count=guard.pii_redaction_count,
                pii_entities=guard.pii_entities,
                guardrail_allowed=False,
                guardrail_category=guard.block_category,
                retrieved_count=0,
                top_retrieval_score=0.0,
                threshold=cfg.min_retrieval_score,
                strict_grounding=cfg.strict_grounding,
                abstained=False,
                trust_level="Blocked",
            )
            metrics = TrustMetrics(
                retrieval_score=0.0,
                faithfulness_score=1.0,
                citation_score=1.0,
                safety_score=0.0,
                composite_trust_score=0.0,
                trust_level="Blocked",
                supported_claims=0,
                partial_claims=0,
                unsupported_claims=0,
                total_claims=0,
                uncertainty_warning=guard.block_reason,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if log_to_audit:
                self.audit_logger.log_interaction(
                    interaction_id=interaction_id,
                    subject=self.subject,
                    chapter=self.chapter,
                    sanitized_query=guard.sanitized_query,
                    decision=decision,
                    strict_grounding=cfg.strict_grounding,
                    pii_redacted_count=guard.pii_redaction_count,
                    pii_entities=guard.pii_entities,
                    guardrail_rules=guard.matched_rules,
                    composite_trust_score=0.0,
                    trust_level="Blocked",
                    latency_ms=latency_ms,
                )
            return PipelineResult(
                interaction_id=interaction_id,
                decision=decision,
                answer=f"### 🛡️ Guardrail Notice\n\n{guard.block_reason}",
                sanitized_query=guard.sanitized_query,
                youtube_query="",
                guardrail=guard,
                metrics=metrics,
                sources=[],
                decision_trace=trace,
                warnings=warnings,
                latency_ms=latency_ms,
                show_videos=False,
            )

        # ------------------------------------------------------------------
        # Stage 3: Semantic Retrieval with Exact Cosine Relevance
        # ------------------------------------------------------------------
        scored_docs: List[Tuple[Any, float]] = retrieve_with_scores(
            vectorstore=self.vectorstore,
            embeddings=self.embeddings,
            query=guard.sanitized_query,
            k=cfg.top_k,
            chapter_filter=self.chapter_filter,
        )
        sources = format_source_provenance(scored_docs)
        top_score = scored_docs[0][1] if scored_docs else 0.0

        # ------------------------------------------------------------------
        # Stage 4: Pre-Generation Grounding / Abstention Gate
        # ------------------------------------------------------------------
        should_abstain = (not scored_docs) or (
            cfg.strict_grounding and top_score < cfg.min_retrieval_score
        )
        if should_abstain:
            abstention_answer, metrics = build_abstention_response(
                query=guard.sanitized_query,
                subject=self.subject,
                chapter=self.chapter,
                best_score=top_score,
                threshold=cfg.min_retrieval_score,
                scored_docs=scored_docs,
            )
            trace = build_decision_trace(
                pii_redacted_count=guard.pii_redaction_count,
                pii_entities=guard.pii_entities,
                guardrail_allowed=True,
                guardrail_category="",
                retrieved_count=len(scored_docs),
                top_retrieval_score=top_score,
                threshold=cfg.min_retrieval_score,
                strict_grounding=cfg.strict_grounding,
                abstained=True,
                trust_level="Abstained",
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if log_to_audit:
                self.audit_logger.log_interaction(
                    interaction_id=interaction_id,
                    subject=self.subject,
                    chapter=self.chapter,
                    sanitized_query=guard.sanitized_query,
                    decision="abstained",
                    strict_grounding=cfg.strict_grounding,
                    pii_redacted_count=guard.pii_redaction_count,
                    pii_entities=guard.pii_entities,
                    retrieval_score=top_score,
                    faithfulness_score=1.0,
                    citation_score=1.0,
                    composite_trust_score=metrics.composite_trust_score,
                    trust_level="Abstained",
                    latency_ms=latency_ms,
                )
            return PipelineResult(
                interaction_id=interaction_id,
                decision="abstained",
                answer=abstention_answer,
                sanitized_query=guard.sanitized_query,
                youtube_query="",
                guardrail=guard,
                metrics=metrics,
                sources=sources,
                decision_trace=trace,
                warnings=warnings,
                latency_ms=latency_ms,
                show_videos=False,
            )

        # ------------------------------------------------------------------
        # Stage 5: Grounded LLM Synthesis & Output Safety Audit
        # ------------------------------------------------------------------
        context_block = build_grounded_context(scored_docs)
        messages = self._build_messages(
            sanitized_query=guard.sanitized_query,
            context_block=context_block,
            chat_history=chat_history,
            max_turns=cfg.max_history_turns,
        )

        llm_response = self.llm.invoke(messages)
        raw_answer = getattr(llm_response, "content", str(llm_response)).strip()

        out_audit = audit_llm_output(raw_answer)
        if out_audit.flags:
            warnings.append(f"Output audit flags: {', '.join(out_audit.flags)}")
        final_answer = out_audit.audited_text

        # ------------------------------------------------------------------
        # Stage 6: Post-Generation Claim Faithfulness & Trust Calibration
        # ------------------------------------------------------------------
        embed_fn = getattr(self.embeddings, "embed_documents", None)
        metrics = compute_trust_metrics(
            answer=final_answer,
            scored_docs=scored_docs,
            embed_fn=embed_fn,
            safety_passed=out_audit.safe,
        )

        llm_abstained = is_abstention_text(final_answer)
        decision = (
            "blocked_output_safety"
            if not out_audit.safe
            else ("abstained" if llm_abstained else "answered")
        )

        trace = build_decision_trace(
            pii_redacted_count=guard.pii_redaction_count,
            pii_entities=guard.pii_entities,
            guardrail_allowed=True,
            guardrail_category="",
            retrieved_count=len(scored_docs),
            top_retrieval_score=top_score,
            threshold=cfg.min_retrieval_score,
            strict_grounding=cfg.strict_grounding,
            abstained=llm_abstained,
            faithfulness_score=metrics.faithfulness_score,
            trust_level=metrics.trust_level,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        cited_labels = [f"{s['source_file']} p.{s['page']}" for s in sources]

        if log_to_audit:
            self.audit_logger.log_interaction(
                interaction_id=interaction_id,
                subject=self.subject,
                chapter=self.chapter,
                sanitized_query=guard.sanitized_query,
                decision=decision,
                strict_grounding=cfg.strict_grounding,
                pii_redacted_count=guard.pii_redaction_count,
                pii_entities=guard.pii_entities,
                guardrail_rules=out_audit.flags,
                retrieval_score=metrics.retrieval_score,
                faithfulness_score=metrics.faithfulness_score,
                citation_score=metrics.citation_score,
                composite_trust_score=metrics.composite_trust_score,
                trust_level=metrics.trust_level,
                unsupported_claims=metrics.unsupported_claims,
                sources_cited=cited_labels,
                latency_ms=latency_ms,
            )

        yt_query = (
            sanitize_search_query(guard.sanitized_query, subject=self.subject)
            if decision == "answered"
            else ""
        )

        return PipelineResult(
            interaction_id=interaction_id,
            decision=decision,
            answer=final_answer,
            sanitized_query=guard.sanitized_query,
            youtube_query=yt_query,
            guardrail=guard,
            metrics=metrics,
            sources=sources,
            decision_trace=trace,
            warnings=warnings,
            latency_ms=latency_ms,
            show_videos=(decision == "answered"),
        )
