"""
Accountability, Audit Logging & Human-in-the-Loop Feedback for StudyPal.

Persists structured, privacy-preserving JSONL audit logs (never storing raw PII)
and human oversight feedback so system behavior can be inspected, exported, and
continuously audited.
"""

from __future__ import annotations

import csv
import io
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
DEFAULT_LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")
DEFAULT_AUDIT_LOG_PATH = os.path.join(DEFAULT_LOGS_DIR, "trust_audit.jsonl")
DEFAULT_FEEDBACK_LOG_PATH = os.path.join(DEFAULT_LOGS_DIR, "feedback.jsonl")


class TrustAuditLogger:
    """Append-only JSONL logger for AI trustworthiness audit trails & user feedback."""

    def __init__(
        self,
        audit_path: str = DEFAULT_AUDIT_LOG_PATH,
        feedback_path: str = DEFAULT_FEEDBACK_LOG_PATH,
    ) -> None:
        self.audit_path = audit_path
        self.feedback_path = feedback_path
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.feedback_path), exist_ok=True)

    @staticmethod
    def new_interaction_id() -> str:
        return str(uuid.uuid4())[:12]

    def log_interaction(
        self,
        *,
        interaction_id: str,
        subject: str,
        chapter: str,
        sanitized_query: str,
        decision: str,  # "answered" | "abstained" | "blocked_injection" | "blocked_safety" | "blocked_validation" | "error"
        strict_grounding: bool,
        pii_redacted_count: int = 0,
        pii_entities: Optional[List[str]] = None,
        guardrail_rules: Optional[List[str]] = None,
        retrieval_score: float = 0.0,
        faithfulness_score: float = 0.0,
        citation_score: float = 0.0,
        composite_trust_score: float = 0.0,
        trust_level: str = "N/A",
        unsupported_claims: int = 0,
        sources_cited: Optional[List[str]] = None,
        latency_ms: float = 0.0,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "interaction_id": interaction_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "subject": subject,
            "chapter": chapter,
            "sanitized_query": sanitized_query,
            "decision": decision,
            "strict_grounding": strict_grounding,
            "pii_redacted_count": pii_redacted_count,
            "pii_entities": pii_entities or [],
            "guardrail_rules": guardrail_rules or [],
            "retrieval_score": round(float(retrieval_score), 4),
            "faithfulness_score": round(float(faithfulness_score), 4),
            "citation_score": round(float(citation_score), 4),
            "composite_trust_score": round(float(composite_trust_score), 4),
            "trust_level": trust_level,
            "unsupported_claims": int(unsupported_claims),
            "sources_cited": sources_cited or [],
            "latency_ms": round(float(latency_ms), 1),
        }
        with open(self.audit_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def log_feedback(
        self,
        *,
        interaction_id: str,
        rating: str,  # "positive" | "negative"
        reason: str = "",
        comment: str = "",
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "interaction_id": interaction_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rating": rating,
            "reason": reason,
            "comment": comment[:400],
        }
        with open(self.feedback_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    @staticmethod
    def _read_jsonl(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def load_audit_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        records = self._read_jsonl(self.audit_path)
        if limit is not None and limit > 0:
            return records[-limit:]
        return records

    def load_feedback_records(self) -> List[Dict[str, Any]]:
        return self._read_jsonl(self.feedback_path)

    def get_summary_metrics(self) -> Dict[str, Any]:
        records = self.load_audit_records()
        feedback = self.load_feedback_records()

        total = len(records)
        answered = [r for r in records if r.get("decision") == "answered"]
        abstained = [r for r in records if r.get("decision") == "abstained"]
        blocked = [
            r for r in records if str(r.get("decision", "")).startswith("blocked_")
        ]
        pii_events = sum(int(r.get("pii_redacted_count", 0)) for r in records)

        avg_trust = (
            sum(float(r.get("composite_trust_score", 0.0)) for r in answered)
            / len(answered)
            if answered
            else 0.0
        )
        avg_faithfulness = (
            sum(float(r.get("faithfulness_score", 0.0)) for r in answered)
            / len(answered)
            if answered
            else 0.0
        )
        avg_retrieval = (
            sum(float(r.get("retrieval_score", 0.0)) for r in answered)
            / len(answered)
            if answered
            else 0.0
        )
        avg_latency = (
            sum(float(r.get("latency_ms", 0.0)) for r in records) / total
            if total
            else 0.0
        )

        # Deduplicate feedback by interaction_id (latest wins)
        latest_fb: Dict[str, Dict[str, Any]] = {}
        for f in feedback:
            latest_fb[f.get("interaction_id", "")] = f
        pos_fb = sum(1 for f in latest_fb.values() if f.get("rating") == "positive")
        neg_fb = sum(1 for f in latest_fb.values() if f.get("rating") == "negative")
        total_fb = pos_fb + neg_fb
        approval_rate = (pos_fb / total_fb) if total_fb else None

        return {
            "total_interactions": total,
            "answered_count": len(answered),
            "abstained_count": len(abstained),
            "blocked_count": len(blocked),
            "pii_redactions_total": pii_events,
            "avg_trust_score": round(avg_trust, 4),
            "avg_faithfulness_score": round(avg_faithfulness, 4),
            "avg_retrieval_score": round(avg_retrieval, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "feedback_total": total_fb,
            "feedback_positive": pos_fb,
            "feedback_negative": neg_fb,
            "human_approval_rate": round(approval_rate, 4) if approval_rate is not None else None,
        }

    def export_audit_csv(self) -> str:
        records = self.load_audit_records()
        fieldnames = [
            "interaction_id",
            "timestamp_utc",
            "subject",
            "chapter",
            "sanitized_query",
            "decision",
            "strict_grounding",
            "pii_redacted_count",
            "retrieval_score",
            "faithfulness_score",
            "citation_score",
            "composite_trust_score",
            "trust_level",
            "unsupported_claims",
            "latency_ms",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)
        return buf.getvalue()
