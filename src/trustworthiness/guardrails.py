"""
Safety, Security & Privacy Guardrails for StudyPal.

Covers:
1. Privacy-by-Design: Deterministic PII detection & redaction before any text
   leaves the local machine to external APIs (Groq LLM, YouTube search).
2. Security & Robustness: Multi-layered prompt-injection, jailbreak, delimiter
   smuggling, and system-prompt exfiltration detection.
3. Input & Output Safety: Domain-aware harmful content filtering calibrated so
   legitimate academic Biology & Operating Systems terminology (e.g., "process
   kill", "deadlock", "starvation", "human reproduction") is never falsely
   blocked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

MAX_QUERY_LENGTH = 1500
MIN_QUERY_LENGTH = 2


@dataclass
class PIIRedactionResult:
    redacted_text: str
    entities_found: List[str] = field(default_factory=list)
    redaction_count: int = 0

    @property
    def has_pii(self) -> bool:
        return self.redaction_count > 0


@dataclass
class GuardrailResult:
    allowed: bool
    sanitized_query: str
    block_category: str = ""  # "", "validation", "prompt_injection", "safety"
    block_reason: str = ""
    pii_entities: List[str] = field(default_factory=list)
    pii_redaction_count: int = 0
    matched_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class OutputAuditResult:
    safe: bool
    audited_text: str
    flags: List[str] = field(default_factory=list)
    pii_redacted_in_output: bool = False


# ---------------------------------------------------------------------------
# 1. Privacy / PII Detection & Redaction
# ---------------------------------------------------------------------------
_PII_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    (
        "API_KEY",
        re.compile(
            r"\b(?:gsk_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{20,})\b"
        ),
        "[REDACTED_API_KEY]",
    ),
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "[REDACTED_CARD]",
    ),
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    (
        "AADHAAR",
        re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
        "[REDACTED_ID]",
    ),
    (
        "PHONE",
        re.compile(
            r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
        ),
        "[REDACTED_PHONE]",
    ),
    (
        "IP_ADDRESS",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
        ),
        "[REDACTED_IP]",
    ),
]


def _luhn_check(digits: str) -> bool:
    """Validate candidate credit card numbers with Luhn algorithm to avoid false positives."""
    nums = [int(c) for c in digits if c.isdigit()]
    if not (13 <= len(nums) <= 19):
        return False
    checksum = 0
    parity = len(nums) % 2
    for idx, d in enumerate(nums):
        if idx % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def redact_pii(text: str) -> PIIRedactionResult:
    """Detect and redact PII from text before sending to external services or logs."""
    if not text:
        return PIIRedactionResult(redacted_text="")

    redacted = text
    entities: List[str] = []
    total_count = 0

    for label, pattern, token in _PII_PATTERNS:
        if label == "CREDIT_CARD":
            def _replace_card(match: re.Match) -> str:
                nonlocal total_count
                candidate = match.group(0)
                if _luhn_check(candidate):
                    total_count += 1
                    if label not in entities:
                        entities.append(label)
                    return token
                return candidate

            redacted = pattern.sub(_replace_card, redacted)
            continue

        matches = list(pattern.finditer(redacted))
        if matches:
            total_count += len(matches)
            if label not in entities:
                entities.append(label)
            redacted = pattern.sub(token, redacted)

    return PIIRedactionResult(
        redacted_text=redacted,
        entities_found=entities,
        redaction_count=total_count,
    )


# ---------------------------------------------------------------------------
# 2. Prompt Injection & Jailbreak Defense
# ---------------------------------------------------------------------------
_PROMPT_INJECTION_RULES: List[Tuple[str, re.Pattern]] = [
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|bypass|override|abandon)\b.{0,40}\b(?:previous|prior|above|all|system|original|safety|grounding)\b.{0,30}\b(?:instructions?|prompts?|rules?|constraints?|guidelines?|directives?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "jailbreak_persona",
        re.compile(
            r"\b(?:you are now|act as|pretend to be|switch to|enter)\b.{0,35}\b(?:dan\b|do anything now|unrestricted|uncensored|jailbroken|developer mode|evil ai|no rules)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"\b(?:reveal|show|print|output|repeat|dump|leak|tell me)\b.{0,35}\b(?:your\s+)?(?:system prompt|hidden instructions?|initial prompt|developer prompt|secret instructions?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_smuggling",
        re.compile(
            r"(?:<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[SYSTEM\]|###\s*System\s*:|BEGIN\s+SYSTEM\s+PROMPT)",
            re.IGNORECASE,
        ),
    ),
    (
        "forced_hallucination",
        re.compile(
            r"\b(?:make up|fabricate|invent|hallucinate)\b.{0,35}\b(?:facts?|citations?|sources?|references?|page numbers?)\b",
            re.IGNORECASE,
        ),
    ),
]


def check_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """Return (is_attack, matched_rule_names)."""
    matched: List[str] = []
    for rule_name, pattern in _PROMPT_INJECTION_RULES:
        if pattern.search(text):
            matched.append(rule_name)
    return (len(matched) > 0, matched)


# ---------------------------------------------------------------------------
# 3. Domain-Aware Content Safety Filter
# ---------------------------------------------------------------------------
# Carefully scoped to high-harm actionable requests while avoiding false
# positives on academic OS ("kill process", "deadlock", "abort") and
# Biology ("reproduction", "cancer", "mutation", "immune system") topics.
_UNSAFE_CONTENT_RULES: List[Tuple[str, re.Pattern]] = [
    (
        "self_harm",
        re.compile(
            r"\b(?:how to|ways to|steps to|help me)\b.{0,30}\b(?:commit suicide|kill myself|self[- ]harm|hurt myself|end my life)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "weapons_explosives",
        re.compile(
            r"\b(?:how to|instructions? (?:for|to)|recipe (?:for|to)|build|make|synthesize)\b.{0,35}\b(?:a bomb|explosives?|nerve agent|ricin|bioweapon|pipe bomb|molotov)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cyberattack_actionable",
        re.compile(
            r"\b(?:write|create|give me|generate)\b.{0,30}\b(?:ransomware|keylogger|trojan|malware|phishing kit|ddos script|sql injection payload to hack)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "hate_violence",
        re.compile(
            r"\b(?:promote violence against|incite hatred against|exterminate)\b",
            re.IGNORECASE,
        ),
    ),
]


def check_content_safety(text: str) -> Tuple[bool, List[str]]:
    """Return (is_unsafe, matched_categories)."""
    matched: List[str] = []
    for category, pattern in _UNSAFE_CONTENT_RULES:
        if pattern.search(text):
            matched.append(category)
    return (len(matched) > 0, matched)


# ---------------------------------------------------------------------------
# 4. Unified Input Validation Pipeline
# ---------------------------------------------------------------------------
def validate_user_input(query: str, enable_pii_redaction: bool = True) -> GuardrailResult:
    """
    Run full input trustworthiness pipeline:
    1. Length & format validation
    2. PII detection & redaction (privacy-by-design)
    3. Prompt-injection & jailbreak detection
    4. Content safety check
    """
    if query is None or not isinstance(query, str):
        return GuardrailResult(
            allowed=False,
            sanitized_query="",
            block_category="validation",
            block_reason="Please enter a valid text question.",
            matched_rules=["empty_input"],
        )

    cleaned = query.strip()
    if len(cleaned) < MIN_QUERY_LENGTH:
        return GuardrailResult(
            allowed=False,
            sanitized_query="",
            block_category="validation",
            block_reason="Your question is too short. Please ask a complete textbook question.",
            matched_rules=["too_short"],
        )

    if len(cleaned) > MAX_QUERY_LENGTH:
        return GuardrailResult(
            allowed=False,
            sanitized_query="",
            block_category="validation",
            block_reason=(
                f"Your question exceeds the maximum allowed length of {MAX_QUERY_LENGTH} characters "
                "to ensure reliable retrieval and prevent context overflow."
            ),
            matched_rules=["too_long"],
        )

    # Step 1: Redact PII first so downstream checks & audit logs never store raw PII
    pii_entities: List[str] = []
    pii_count = 0
    sanitized = cleaned
    warnings: List[str] = []

    if enable_pii_redaction:
        pii_res = redact_pii(cleaned)
        sanitized = pii_res.redacted_text
        pii_entities = pii_res.entities_found
        pii_count = pii_res.redaction_count
        if pii_res.has_pii:
            warnings.append(
                f"Privacy Shield redacted {pii_count} sensitive item(s) "
                f"({', '.join(pii_entities)}) before processing."
            )

    # Step 2: Prompt injection & jailbreak detection
    is_injection, inj_rules = check_prompt_injection(sanitized)
    if is_injection:
        return GuardrailResult(
            allowed=False,
            sanitized_query=sanitized,
            block_category="prompt_injection",
            block_reason=(
                "Security Guardrail blocked this request because it resembles a prompt-injection "
                "or instruction-override attempt. Please ask a genuine study question about the selected textbook."
            ),
            pii_entities=pii_entities,
            pii_redaction_count=pii_count,
            matched_rules=inj_rules,
            warnings=warnings,
        )

    # Step 3: Content safety filter
    is_unsafe, safety_rules = check_content_safety(sanitized)
    if is_unsafe:
        return GuardrailResult(
            allowed=False,
            sanitized_query=sanitized,
            block_category="safety",
            block_reason=(
                "Safety Guardrail blocked this request because it asks for harmful or unsafe content. "
                "StudyPal is restricted to safe educational assistance grounded in your textbooks."
            ),
            pii_entities=pii_entities,
            pii_redaction_count=pii_count,
            matched_rules=safety_rules,
            warnings=warnings,
        )

    return GuardrailResult(
        allowed=True,
        sanitized_query=sanitized,
        pii_entities=pii_entities,
        pii_redaction_count=pii_count,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 5. Output Safety & Leak Audit + YouTube Query Sanitization
# ---------------------------------------------------------------------------
_PROMPT_LEAK_MARKERS = [
    "STRICT GROUNDING RULES:",
    "NEVER FABRICATE CITATIONS",
    "YOU ARE STUDYPAL, A TRUSTWORTHY",
]


def audit_llm_output(answer: str) -> OutputAuditResult:
    """Verify the LLM output has no system-prompt leaks, raw PII, or unsafe content."""
    if not answer:
        return OutputAuditResult(safe=True, audited_text="")

    flags: List[str] = []
    audited = answer

    # Redact any accidental PII in output
    pii_res = redact_pii(audited)
    if pii_res.has_pii:
        audited = pii_res.redacted_text
        flags.append(f"output_pii_redacted:{','.join(pii_res.entities_found)}")

    # Check system prompt leakage
    upper_text = audited.upper()
    for marker in _PROMPT_LEAK_MARKERS:
        if marker in upper_text:
            flags.append("system_prompt_leak_detected")
            return OutputAuditResult(
                safe=False,
                audited_text=(
                    "The generated response was withheld by the Output Guardrail to prevent "
                    "internal instruction leakage. Please rephrase your textbook question."
                ),
                flags=flags,
                pii_redacted_in_output=pii_res.has_pii,
            )

    # Check unsafe content in output
    is_unsafe, unsafe_cats = check_content_safety(audited)
    if is_unsafe:
        flags.extend([f"unsafe_output:{c}" for c in unsafe_cats])
        return OutputAuditResult(
            safe=False,
            audited_text=(
                "The generated response was withheld by the Output Safety Guardrail. "
                "Please ask an educational question grounded in the selected textbook."
            ),
            flags=flags,
            pii_redacted_in_output=pii_res.has_pii,
        )

    return OutputAuditResult(
        safe=True,
        audited_text=audited,
        flags=flags,
        pii_redacted_in_output=pii_res.has_pii,
    )


def sanitize_search_query(query: str, subject: str = "") -> str:
    """Prepare a privacy-safe, subject-contextualized query for YouTube search."""
    pii_res = redact_pii(query or "")
    cleaned = re.sub(r"\[REDACTED_[A-Z_]+\]", "", pii_res.redacted_text)
    cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rsplit(" ", 1)[0]
    if subject:
        short_subject = subject.split("(")[0].strip()
        if short_subject.lower() not in cleaned.lower():
            cleaned = f"{cleaned} {short_subject} lecture"
    return cleaned.strip()
