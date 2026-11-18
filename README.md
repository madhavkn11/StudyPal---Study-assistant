# StudyPal – Trustworthy AI-Powered Study Assistant

> Ask questions from your textbook. Get verifiable, cited answers. Inspect faithfulness in real time. Actually understand it.

StudyPal is a **Trustworthy Retrieval-Augmented Generation (RAG)** study assistant that lets students chat with their textbooks with full **explainability, hallucination protection, privacy-by-design, security guardrails, and auditability** — aligned with **NIST AI RMF 1.0** and the **EU Trustworthy AI 7 Pillars**.

---

## Trustworthy AI Features

- **Grounded Inline Citations & Provenance** — Every factual claim cites exact textbook excerpts (`[S1]`, `[S2]`, ...) with file name, chapter, **1-indexed page number**, cosine relevance bar, and SHA-256 dataset fingerprint.
- **Hallucination Detection & Claim-by-Claim Faithfulness Verification** — Every sentence of the LLM answer is verified locally against retrieved textbook chunks (`Supported` / `Partially Supported` / `Unsupported`) alongside citation validity auditing.
- **Calibrated Composite Trust Score & Safe Abstention** — Computes a transparent 0–100% Composite Trust Score (`High Trust` / `Medium Trust` / `Low Trust`) and **deterministically abstains** on out-of-context queries instead of hallucinating.
- **Privacy Shield (Local PII Redaction)** — Automatically detects and masks emails, phone numbers, SSNs, Aadhaar-style IDs, Luhn-validated credit cards, IP addresses, and API keys *locally* before any query reaches Groq or YouTube.
- **Security & Safety Guardrails** — Blocks prompt injections, jailbreaks (`DAN`), delimiter smuggling, forced-hallucination attacks, system-prompt exfiltration, and unsafe requests while maintaining **0% false positives** on academic Biology & Operating Systems terminology.
- **Explainable Decision Trace & Model Card** — Interactive 5-stage pipeline trace for every answer plus a built-in **Transparency & Model Card** tab detailing architecture, data provenance, privacy boundaries, and known limitations.
- **Human-in-the-Loop Feedback & Exportable Audit Trail** — Per-answer 👍/👎 feedback with issue categorization, privacy-preserving JSONL audit logging, CSV/JSON export, and a live **Trust & Evaluation Dashboard**.
- **Automated Trustworthiness & Red-Team Benchmark** — Built-in reproducible benchmark (`100.0%` across Security, Fairness, Privacy, Faithfulness Separation, In-Domain Retrieval, and Out-of-Context Abstention). See [TRUSTWORTHINESS.md](TRUSTWORTHINESS.md) for the full technical & compliance report.

---

## Supported Subjects

| Level | Subject | Indexed Chapters / Scope |
|---|---|---|
| Class 12 | Biology | `Ch 6` (Evolution), `Ch 12` (Ecosystem), `All Chapters` |
| B.Tech | Operating Systems | `OS Part 1` – `OS Part 4` (with chapter metadata filtering), `All Chapters` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend & Trust Dashboard | Streamlit |
| LLM | Groq API — `openai/gpt-oss-120b` (`temperature=0`, configurable via `GROQ_MODEL`) |
| Orchestration & Trust Pipeline | LangChain + Custom `src/trustworthiness/` Suite |
| Local Embeddings & Faithfulness | HuggingFace `all-MiniLM-L6-v2` (local CPU, offline-capable) |
| Vector Database | ChromaDB (cosine space + page/chapter/SHA-256 metadata) |
| Document Processing | PyPDFLoader (`pypdf`) + Recursive Character Text Splitter |
| Video Recommendations | `yt-dlp` (optional, uses PII-sanitized queries) |

---

## Project Structure

```text
StudyPal/
│
├── data/
│   ├── class_12/biology/         # Class 12 Biology PDFs (Ch 6, Ch 12)
│   └── operating_systems/        # B.Tech OS textbook PDFs (OS Part 1-4)
│
├── src/
│   ├── main.py                   # Streamlit app (Study Chat, Trust Dashboard, Model Card)
│   ├── chatbot_utility.py        # Subject/chapter discovery & metadata filter mapping
│   ├── get_yt_video.py           # YouTube video recommendation lookup
│   ├── vectorize_book.py         # Page/chapter/SHA-256 provenance vectorization
│   ├── vectorize_script.py       # Biology vectorization script
│   ├── vectorize_os.py           # Operating Systems vectorization script
│   ├── evaluate_trustworthiness.py # CLI automated trustworthiness & red-team benchmark
│   └── trustworthiness/
│       ├── guardrails.py         # Local PII redaction, prompt-injection & safety guardrails
│       ├── grounding.py          # Grounded prompt, abstention gate, faithfulness & trust score
│       ├── explainability.py     # Provenance formatter, decision trace, NIST/EU Model Card
│       ├── audit.py              # Privacy-preserving JSONL audit logger & feedback capture
│       ├── pipeline.py           # End-to-end TrustworthyRAGPipeline orchestrator
│       └── evaluator.py          # Automated 6-dimension trustworthiness benchmark suite
│
├── tests/
│   └── test_trustworthiness.py   # Automated unit & integration tests
│
├── reports/
│   ├── trustworthiness_evaluation.json
│   └── trustworthiness_report.md
│
├── vector_db/                    # Local ChromaDB stores (full subject)
├── chapters_vector_db/           # Local ChromaDB stores (chapter-wise)
├── logs/                         # Local JSONL audit & feedback logs (gitignored)
├── TRUSTWORTHINESS.md            # Full Trustworthy AI Governance & Evaluation Report
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/madhavkn11/StudyPal---Study-assistant.git
cd StudyPal---Study-assistant
python -m venv .venv
```

Activate the environment:
```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
DEVICE=cpu
```

### 4. Build the Provenance-Enriched Vector Databases

```bash
python src/vectorize_script.py
python src/vectorize_os.py
```

This extracts page-accurate text, attaches chapter/page/SHA-256 provenance metadata, and indexes into `vector_db/` and `chapters_vector_db/`.

### 5. Run the App

```bash
streamlit run src/main.py
```

Open `http://localhost:8501` in your browser and explore all three tabs:
1. **💬 Trustworthy Study Chat**
2. **📊 Trust & Evaluation Dashboard**
3. **📋 Transparency & Model Card**

---

## Running Verification Tests & Trustworthiness Benchmark

**Unit & Integration Tests (12 tests):**
```bash
python -m unittest discover -s tests -v
```

**Automated Trustworthiness & Red-Team Benchmark:**
```bash
python src/evaluate_trustworthiness.py
```

Generates fresh reports in `reports/trustworthiness_evaluation.json` and `reports/trustworthiness_report.md`.

---

## Author

**Madhav Karthik Nambi**  
GitHub: `madhavkn11`  

Live link: https://studypal---study-assistant.streamlit.app/
