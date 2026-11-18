import json
import os
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from chatbot_utility import get_chapter_list, map_chapter_to_filter
from get_yt_video import get_yt_video_link
from trustworthiness import (
    PipelineConfig,
    TrustAuditLogger,
    TrustworthyRAGPipeline,
    get_system_card,
    run_trustworthiness_benchmark,
    save_benchmark_report,
)

# Load environment variables
load_dotenv()
DEVICE = os.getenv("DEVICE", "cpu")

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

SUBJECTS_LIST = [
    "Biology (Class 12)",
    "Operating Systems (B.Tech)",
]


# ---------------------------------------------------------------------------
# Cached Resources (Fast, Deterministic Initialization)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading local sentence embedding model...")
def load_embeddings() -> HuggingFaceEmbeddings:
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": DEVICE, "local_files_only": True},
        )
    except Exception:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": DEVICE},
        )


def get_vector_db_path(chapter: str, subject: str) -> str:
    if subject == "Operating Systems (B.Tech)":
        return os.path.join(parent_dir, "vector_db", "operating_systems_vector_db")

    if chapter == "All Chapters":
        return os.path.join(parent_dir, "vector_db", "class_12_biology_vector_db")

    chapter_db = os.path.join(parent_dir, "chapters_vector_db", chapter)
    if os.path.exists(chapter_db):
        return chapter_db
    return os.path.join(parent_dir, "vector_db", "class_12_biology_vector_db")


@st.cache_resource(show_spinner="Opening textbook vector store...")
def load_vectorstore(vector_db_path: str) -> Chroma:
    embeddings = load_embeddings()
    return Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings,
    )


@st.cache_resource(show_spinner=False)
def load_llm() -> ChatGroq:
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(model=model_name, temperature=0)


def build_pipeline(selected_chapter: str, selected_subject: str) -> TrustworthyRAGPipeline:
    vector_db_path = get_vector_db_path(selected_chapter, selected_subject)
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(vector_db_path)
    llm = load_llm()

    # Apply metadata chapter filtering when using a full-subject vector store
    using_chapter_store = "chapters_vector_db" in vector_db_path
    chapter_filter = (
        None
        if using_chapter_store
        else map_chapter_to_filter(selected_chapter, selected_subject)
    )

    return TrustworthyRAGPipeline(
        vectorstore=vectorstore,
        embeddings=embeddings,
        llm=llm,
        subject=selected_subject,
        chapter=selected_chapter,
        chapter_filter=chapter_filter,
        audit_logger=TrustAuditLogger(),
    )


# ---------------------------------------------------------------------------
# UI Rendering Helpers
# ---------------------------------------------------------------------------
def _trust_badge(level: str, score: float) -> str:
    if level == "High Trust":
        return f"🟢 **High Trust** ({score:.0%})"
    if level == "Medium Trust":
        return f"🟡 **Medium Trust — Verify Sources** ({score:.0%})"
    if level == "Abstained":
        return "🛡️ **Safe Abstention** (Prevented Hallucination)"
    if level == "Blocked":
        return "🚫 **Blocked by Guardrail**"
    return f"🔴 **Low Trust** ({score:.0%})"


def _claim_status_icon(status: str) -> str:
    if status == "Supported":
        return "✅ Supported"
    if status == "Partially Supported":
        return "⚠️ Partially Supported"
    return "❌ Unsupported"


def render_trust_panel(msg: Dict[str, Any], msg_idx: int, audit_logger: TrustAuditLogger) -> None:
    """Render trustworthiness metrics, cited excerpts, claim verification, trace, and feedback."""
    trust_data = msg.get("trust")
    if not trust_data:
        return

    metrics = trust_data["metrics"]
    sources = trust_data.get("sources", [])
    trace = trust_data.get("decision_trace", [])
    warnings = trust_data.get("warnings", [])
    interaction_id = trust_data.get("interaction_id", f"msg-{msg_idx}")
    latency_ms = trust_data.get("latency_ms", 0.0)

    # 1. Trustworthiness Status Bar
    st.markdown(
        _trust_badge(metrics["trust_level"], metrics["composite_trust_score"])
        + f" &nbsp;·&nbsp; Latency: `{latency_ms:.0f} ms` &nbsp;·&nbsp; Audit ID: `{interaction_id}`"
    )

    if metrics["trust_level"] not in ("Blocked",):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Composite Trust", f"{metrics['composite_trust_score']:.0%}")
        c2.metric("Faithfulness", f"{metrics['faithfulness_score']:.0%}")
        c3.metric("Retrieval Relevance", f"{metrics['retrieval_score']:.0%}")
        c4.metric("Citation Integrity", f"{metrics['citation_score']:.0%}")

    # 2. Warnings & Privacy Notices
    for w in warnings:
        st.info(f"🛡️ {w}")
    if metrics.get("uncertainty_warning"):
        if metrics["trust_level"] in ("Abstained", "Medium Trust"):
            st.warning(f"⚠️ {metrics['uncertainty_warning']}")
        elif metrics["trust_level"] in ("Low Trust", "Blocked"):
            st.error(f"⚠️ {metrics['uncertainty_warning']}")

    # 3. Cited Textbook Sources & Provenance
    if sources:
        with st.expander(f"📚 Cited Textbook Sources & Provenance ({len(sources)} excerpts)"):
            for src in sources:
                st.markdown(
                    f"**[{src['source_id']}] {src['source_file']}** — "
                    f"Chapter: `{src['chapter']}` | Page: **`{src['page']}`** | "
                    f"Relevance: **`{src['similarity_score']:.0%}`**"
                    + (f" | SHA-256: `{src['file_sha256']}`" if src.get("file_sha256") else "")
                )
                st.progress(min(1.0, max(0.0, float(src["similarity_score"]))))
                st.caption(src["excerpt"])
                st.divider()

    # 4. Claim-by-Claim Faithfulness & Decision Trace
    claims = metrics.get("claims", [])
    if claims or trace:
        with st.expander("🔍 Claim-by-Claim Faithfulness Verification & Decision Trace"):
            if claims:
                st.markdown(
                    f"**Claim Grounding Summary:** "
                    f"✅ {metrics['supported_claims']} Supported · "
                    f"⚠️ {metrics['partial_claims']} Partially Supported · "
                    f"❌ {metrics['unsupported_claims']} Unsupported"
                )
                claim_rows = [
                    {
                        "Status": _claim_status_icon(c["status"]),
                        "Matched Source": c["best_source_id"] or "—",
                        "Cited": ", ".join(c["cited_ids"]) if c["cited_ids"] else "None",
                        "Semantic": f"{c['semantic_score']:.0%}",
                        "Lexical": f"{c['lexical_score']:.0%}",
                        "Claim": c["sentence"],
                    }
                    for c in claims
                ]
                st.dataframe(claim_rows, use_container_width=True, hide_index=True)

            if trace:
                st.markdown("**Trustworthiness Pipeline Decision Trace**")
                st.dataframe(trace, use_container_width=True, hide_index=True)

    # 5. Human-in-the-Loop Feedback
    fb_key = f"feedback_done_{interaction_id}"
    if st.session_state.get(fb_key):
        st.caption(f"✅ Feedback recorded ({st.session_state[fb_key]}) — thank you for improving StudyPal!")
    else:
        fb_col1, fb_col2, fb_col3 = st.columns([1.2, 1.2, 3.0])
        with fb_col1:
            if st.button("👍 Accurate", key=f"up_{interaction_id}_{msg_idx}"):
                audit_logger.log_feedback(
                    interaction_id=interaction_id,
                    rating="positive",
                    reason="accurate_and_grounded",
                )
                st.session_state[fb_key] = "Helpful & Accurate"
                st.rerun()
        with fb_col2:
            if st.button("👎 Flag Issue", key=f"down_{interaction_id}_{msg_idx}"):
                st.session_state[f"show_issue_form_{interaction_id}"] = True
        with fb_col3:
            if st.session_state.get(f"show_issue_form_{interaction_id}"):
                reason = st.selectbox(
                    "Issue Type",
                    [
                        "Factual Error / Hallucination",
                        "Missing or Wrong Citation",
                        "Off-Topic / Wrong Chapter",
                        "Unclear Explanation",
                    ],
                    key=f"reason_{interaction_id}_{msg_idx}",
                    label_visibility="collapsed",
                )
                if st.button("Submit Feedback", key=f"submit_fb_{interaction_id}_{msg_idx}"):
                    audit_logger.log_feedback(
                        interaction_id=interaction_id,
                        rating="negative",
                        reason=reason,
                    )
                    st.session_state[fb_key] = f"Flagged: {reason}"
                    st.session_state[f"show_issue_form_{interaction_id}"] = False
                    st.rerun()


def serialize_metrics(metrics_obj: Any) -> Dict[str, Any]:
    return {
        "retrieval_score": metrics_obj.retrieval_score,
        "faithfulness_score": metrics_obj.faithfulness_score,
        "citation_score": metrics_obj.citation_score,
        "safety_score": metrics_obj.safety_score,
        "composite_trust_score": metrics_obj.composite_trust_score,
        "trust_level": metrics_obj.trust_level,
        "supported_claims": metrics_obj.supported_claims,
        "partial_claims": metrics_obj.partial_claims,
        "unsupported_claims": metrics_obj.unsupported_claims,
        "total_claims": metrics_obj.total_claims,
        "invalid_citations": metrics_obj.invalid_citations,
        "uncertainty_warning": metrics_obj.uncertainty_warning,
        "claims": [
            {
                "sentence": c.sentence,
                "status": c.status,
                "semantic_score": c.semantic_score,
                "lexical_score": c.lexical_score,
                "combined_score": c.combined_score,
                "best_source_id": c.best_source_id,
                "cited_ids": c.cited_ids,
            }
            for c in metrics_obj.claims
        ],
    }


def render_videos(video_refs: Optional[List[Any]]) -> None:
    if not video_refs:
        return
    st.divider()
    st.subheader("🎥 Recommended Educational Videos")
    st.caption("Retrieved using a privacy-sanitized, subject-scoped search query.")
    for title, link in video_refs:
        st.markdown(f"**{title}**")
        st.video(link)
        st.link_button("▶ Watch on YouTube", link, use_container_width=True)


# ---------------------------------------------------------------------------
# Streamlit App Layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StudyPal — Trustworthy AI Study Assistant",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ StudyPal — Trustworthy AI Study Assistant")
st.caption(
    "Grounded Textbook RAG · Verifiable Inline Citations · Hallucination & Faithfulness Verification · "
    "Privacy Shield & Security Guardrails · Auditability"
)

audit_logger = TrustAuditLogger()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_history" not in st.session_state:
    st.session_state.video_history = []

# ---------------------------------------------------------------------------
# Sidebar: Subject Selection + Trustworthiness Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📖 Study Settings")

    selected_subject = st.selectbox(
        "Select Subject",
        SUBJECTS_LIST,
        index=0,
    )

    chapter_list = get_chapter_list(selected_subject)
    if selected_subject != "Operating Systems (B.Tech)" and "All Chapters" not in chapter_list:
        chapter_list = ["All Chapters"] + chapter_list

    selected_chapter = st.selectbox(
        "Select Topic / Chapter",
        chapter_list,
        index=0,
    )

    if (
        st.session_state.get("selected_subject") != selected_subject
        or st.session_state.get("selected_chapter") != selected_chapter
        or "rag_pipeline" not in st.session_state
    ):
        st.session_state.rag_pipeline = build_pipeline(selected_chapter, selected_subject)
        st.session_state.selected_subject = selected_subject
        st.session_state.selected_chapter = selected_chapter
        st.session_state.chat_history = []
        st.session_state.video_history = []

    st.divider()
    st.header("🛡️ Trustworthiness Controls")

    strict_grounding = st.toggle(
        "Strict Grounding & Abstention",
        value=True,
        help="Automatically abstain instead of guessing when textbook similarity is below threshold.",
    )
    min_retrieval_score = st.slider(
        "Min Grounding Relevance Threshold",
        min_value=0.15,
        max_value=0.60,
        value=0.30,
        step=0.05,
        help="Higher values enforce stricter textbook alignment and stronger out-of-context abstention.",
    )
    top_k = st.slider(
        "Textbook Excerpts Retrieved (Top-K)",
        min_value=2,
        max_value=6,
        value=4,
    )
    enable_pii_redaction = st.toggle(
        "Privacy Shield (Local PII Redaction)",
        value=True,
        help="Redacts emails, phone numbers, SSNs, IDs, cards, IPs, and API keys locally before sending queries externally.",
    )
    enable_videos = st.toggle(
        "YouTube Video Recommendations",
        value=True,
        help="Fetches related educational videos using only the PII-sanitized query.",
    )

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.video_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_dashboard, tab_card = st.tabs(
    [
        "💬 Trustworthy Study Chat",
        "📊 Trust & Evaluation Dashboard",
        "📋 Transparency & Model Card",
    ]
)

# ===========================================================================
# TAB 1: TRUSTWORTHY STUDY CHAT
# ===========================================================================
with tab_chat:
    st.info(
        f"**Active Scope:** `{selected_subject}` → `{selected_chapter}` | "
        f"**Strict Grounding:** `{'ON' if strict_grounding else 'OFF'}` (`≥ {min_retrieval_score:.0%}`) | "
        f"**Privacy Shield:** `{'ON' if enable_pii_redaction else 'OFF'}` — "
        "Every answer is verified claim-by-claim against cited textbook pages."
    )

    for idx, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_trust_panel(message, idx, audit_logger)
                if idx < len(st.session_state.video_history):
                    render_videos(st.session_state.video_history[idx])

    user_input = st.chat_input(
        f"Ask a question from {selected_subject} ({selected_chapter})..."
    )

    if user_input:
        pipeline_cfg = PipelineConfig(
            strict_grounding=strict_grounding,
            min_retrieval_score=min_retrieval_score,
            top_k=top_k,
            enable_pii_redaction=enable_pii_redaction,
        )

        # Keep clean chat history for conversational memory before appending new turn
        prior_turns = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history
        ]

        with st.chat_message("assistant"):
            with st.spinner("🛡️ Verifying guardrails, retrieving textbook pages, and checking faithfulness..."):
                try:
                    result = st.session_state.rag_pipeline.answer_question(
                        user_input,
                        chat_history=prior_turns,
                        config=pipeline_cfg,
                        log_to_audit=True,
                    )
                except Exception as exc:
                    st.error(
                        f"An error occurred while generating a trustworthy response: {exc}"
                    )
                    result = None

        if result is not None:
            # Store PII-sanitized user query in session history so raw PII is never retained
            display_user_query = result.sanitized_query or user_input
            st.session_state.chat_history.append(
                {"role": "user", "content": display_user_query}
            )
            st.session_state.video_history.append(None)

            video_refs: List[Any] = []
            if enable_videos and result.show_videos and result.youtube_query:
                try:
                    titles, links = get_yt_video_link(result.youtube_query)
                    video_refs = list(zip(titles, links))
                except Exception:
                    video_refs = []

            assistant_msg = {
                "role": "assistant",
                "content": result.answer,
                "trust": {
                    "interaction_id": result.interaction_id,
                    "decision": result.decision,
                    "metrics": serialize_metrics(result.metrics),
                    "sources": result.sources,
                    "decision_trace": result.decision_trace,
                    "warnings": result.warnings,
                    "latency_ms": result.latency_ms,
                },
            }
            st.session_state.chat_history.append(assistant_msg)
            st.session_state.video_history.append(video_refs)
            st.rerun()

# ===========================================================================
# TAB 2: TRUST & EVALUATION DASHBOARD
# ===========================================================================
with tab_dashboard:
    st.subheader("📈 Live Trustworthiness & Audit Metrics")
    summary = audit_logger.get_summary_metrics()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Audited Queries", summary["total_interactions"])
    k2.metric("Avg Composite Trust", f"{summary['avg_trust_score']:.0%}")
    k3.metric("Avg Faithfulness", f"{summary['avg_faithfulness_score']:.0%}")
    k4.metric("Avg Retrieval Score", f"{summary['avg_retrieval_score']:.0%}")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Safe Abstentions", summary["abstained_count"])
    k6.metric("Guardrail Blocks", summary["blocked_count"])
    k7.metric("PII Items Redacted", summary["pii_redactions_total"])
    approval_str = (
        f"{summary['human_approval_rate']:.0%} ({summary['feedback_positive']}/{summary['feedback_total']})"
        if summary["human_approval_rate"] is not None
        else "No feedback yet"
    )
    k8.metric("Human Approval Rate", approval_str)

    st.divider()
    st.subheader("🧪 Automated Trustworthiness & Red-Team Benchmark")
    st.write(
        "Run StudyPal's built-in evaluation suite across **Red-Team Security** (prompt injection, "
        "jailbreaks, safety), **Domain Fairness** (0% false positives on Biology/OS terminology), "
        "**Privacy (PII Redaction)**, **Hallucination Verification**, and **Retrieval & Out-of-Context Abstention**."
    )

    if st.button("▶ Run Trustworthiness Benchmark Now", type="primary"):
        with st.spinner("Running Security, Privacy, Faithfulness, Retrieval & Abstention benchmarks..."):
            embeddings = load_embeddings()
            report = run_trustworthiness_benchmark(
                project_root=parent_dir,
                embeddings=embeddings,
                abstention_threshold=min_retrieval_score,
                include_retrieval=True,
            )
            save_benchmark_report(report, os.path.join(parent_dir, "reports"))
            st.session_state.latest_benchmark = report

    report = st.session_state.get("latest_benchmark")
    if report is None:
        saved_report_path = os.path.join(
            parent_dir, "reports", "trustworthiness_evaluation.json"
        )
        if os.path.exists(saved_report_path):
            try:
                with open(saved_report_path, "r", encoding="utf-8") as fh:
                    report = json.load(fh)
            except Exception:
                report = None

    if report is not None:
        sec = report["security_and_fairness"]
        priv = report["privacy"]
        faith = report["faithfulness_verifier"]
        ret = report.get("retrieval_and_abstention")

        st.success(
            f"**Overall Trustworthiness Benchmark Score: "
            f"{report['overall_trustworthiness_benchmark_score']:.1%}** "
            f"(Generated: `{report['generated_at_utc']}`)"
        )
        b1, b2, b3 = st.columns(3)
        b1.metric(
            "Red-Team Block Rate",
            f"{sec['attack_block_rate']:.0%}",
            f"{sec['attacks_blocked']}/{sec['attacks_total']} attacks blocked",
        )
        b2.metric(
            "Benign Domain Allow Rate",
            f"{1.0 - sec['false_positive_rate']:.0%}",
            f"FPR: {sec['false_positive_rate']:.0%}",
        )
        b3.metric(
            "PII Redaction Accuracy",
            f"{priv['pii_accuracy']:.0%}",
            f"{priv['passed']}/{priv['total']} cases passed",
        )

        b4, b5, b6 = st.columns(3)
        b4.metric(
            "Faithfulness Separation",
            f"+{faith['separation_margin']:.2f}",
            f"Faithful {faith['faithful_answer_score']:.2f} vs Hallucinated {faith['hallucinated_answer_score']:.2f}",
        )
        if ret is not None:
            b5.metric(
                "In-Domain Retrieval Hit Rate",
                f"{ret['in_domain_hit_rate']:.0%}",
                f"Avg top sim: {ret['in_domain_avg_top_score']:.2f}",
            )
            b6.metric(
                "Out-of-Context Abstention",
                f"{ret['out_of_context_abstention_rate']:.0%}",
                f"Avg OOC sim: {ret['out_of_context_avg_top_score']:.2f}",
            )

        with st.expander("View Detailed Benchmark Breakdown"):
            st.markdown("**Red-Team Attack Tests**")
            st.dataframe(sec["attack_details"], use_container_width=True, hide_index=True)
            st.markdown("**PII Privacy Tests**")
            st.dataframe(priv["details"], use_container_width=True, hide_index=True)
            if ret is not None:
                st.markdown("**In-Domain Textbook Retrieval Tests**")
                st.dataframe(ret["in_domain_details"], use_container_width=True, hide_index=True)
                st.markdown("**Out-of-Context Abstention Tests**")
                st.dataframe(ret["out_of_context_details"], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🧾 Privacy-Preserving Audit Log")
    records = audit_logger.load_audit_records(limit=50)
    if records:
        st.dataframe(list(reversed(records)), use_container_width=True, hide_index=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇ Export Audit Log (CSV)",
                data=audit_logger.export_audit_csv(),
                file_name="studypal_trust_audit.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇ Export Audit Log (JSON)",
                data=json.dumps(records, indent=2),
                file_name="studypal_trust_audit.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.caption("No interactions logged yet. Ask a question in the Study Chat tab to populate the audit trail.")

# ===========================================================================
# TAB 3: TRANSPARENCY & MODEL CARD
# ===========================================================================
with tab_card:
    card = get_system_card(parent_dir)
    st.subheader(f"📋 {card['system_name']} — {card['version']}")
    st.write(card["intended_use"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🏗️ System Architecture")
        for k, v in card["architecture"].items():
            st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")

        st.markdown("### 🔒 Privacy & Data-Flow Guarantees")
        for item in card["data_flow_privacy"]:
            st.markdown(f"- {item}")

    with col_b:
        st.markdown("### 🚫 Out-of-Scope Uses")
        for item in card["out_of_scope_use"]:
            st.markdown(f"- {item}")

        st.markdown("### ⚠️ Known Limitations")
        for item in card["known_limitations"]:
            st.markdown(f"- {item}")

    st.divider()
    st.markdown("### 🏛️ Alignment with NIST AI RMF 1.0 & EU Trustworthy AI 7 Pillars")
    st.dataframe(card["trustworthiness_frameworks"], use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📚 Indexed Textbook Dataset Provenance (SHA-256 Verified)")
    if card["datasets"]:
        st.dataframe(card["datasets"], use_container_width=True, hide_index=True)
    else:
        st.warning("No textbook PDFs found in `data/`.")
