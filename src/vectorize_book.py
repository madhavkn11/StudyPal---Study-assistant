"""
Textbook PDF Vectorization Utility with Rich Provenance Metadata.

Indexes PDF textbooks into ChromaDB with per-chunk provenance:
- source_file: PDF filename (e.g., "Ch 6.pdf")
- subject: Human-readable subject label
- chapter: Chapter identifier matching UI filters (e.g., "Ch 6", "OS Part 1")
- page: 1-indexed page number in the PDF
- chunk_index: Sequential chunk number within the source file
- file_sha256: Short SHA-256 fingerprint for data provenance & tamper detection
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from typing import List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)
data_dir = os.path.join(parent_dir, "data")
vector_db_dir = os.path.join(parent_dir, "vector_db")
chapters_vector_db_dir = os.path.join(parent_dir, "chapters_vector_db")

_EMBEDDING_CACHE: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDING_CACHE
    if _EMBEDDING_CACHE is None:
        device = os.getenv("DEVICE", "cpu")
        try:
            _EMBEDDING_CACHE = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": device, "local_files_only": True},
            )
        except Exception:
            _EMBEDDING_CACHE = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": device},
            )
    return _EMBEDDING_CACHE


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _sha256_short(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:12]


def _clean_text(text: str) -> str:
    # Normalize PDF extraction artifacts while preserving paragraph boundaries
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_and_chunk_pdf(
    pdf_path: str,
    subject_label: str,
) -> List[Document]:
    """Load a PDF page-by-page using PyPDFLoader and enrich every chunk with provenance metadata."""
    loader = PyPDFLoader(pdf_path)
    raw_pages = loader.load()

    filename = os.path.basename(pdf_path)
    chapter_name = os.path.splitext(filename)[0]
    rel_source = os.path.relpath(pdf_path, parent_dir).replace("\\", "/")
    file_hash = _sha256_short(pdf_path)

    cleaned_pages: List[Document] = []
    for page_doc in raw_pages:
        cleaned = _clean_text(page_doc.page_content or "")
        if len(cleaned) < 25:
            continue
        # PyPDFLoader uses 0-based page index; convert to 1-indexed human-readable page number
        raw_page = page_doc.metadata.get("page", 0)
        page_num = int(raw_page) + 1 if isinstance(raw_page, (int, float)) else 1
        cleaned_pages.append(
            Document(
                page_content=cleaned,
                metadata={
                    "source": rel_source,
                    "source_file": filename,
                    "subject": subject_label,
                    "chapter": chapter_name,
                    "page": page_num,
                    "file_sha256": file_hash,
                },
            )
        )

    chunks = text_splitter.split_documents(cleaned_pages)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
    return chunks


def _reset_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def vectorize_book_and_store_to_db(
    class_subject_name: str,
    vector_db_name: str,
    subject_label: Optional[str] = None,
) -> int:
    book_dir = os.path.join(data_dir, class_subject_name)
    vector_db_path = os.path.join(vector_db_dir, vector_db_name)

    print(f"Indexing book_dir={book_dir} (exists={os.path.exists(book_dir)})")
    if not os.path.exists(book_dir):
        raise FileNotFoundError(f"Textbook directory not found: {book_dir}")

    if subject_label is None:
        subject_label = (
            "Biology (Class 12)"
            if "biology" in class_subject_name.lower()
            else (
                "Operating Systems (B.Tech)"
                if "operating_systems" in class_subject_name.lower()
                else class_subject_name
            )
        )

    pdf_files = sorted(
        f for f in os.listdir(book_dir) if f.lower().endswith(".pdf")
    )
    all_chunks: List[Document] = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(book_dir, pdf_file)
        file_chunks = load_and_chunk_pdf(pdf_path, subject_label=subject_label)
        print(f"  Loaded {pdf_file}: {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    if not all_chunks:
        print(f"Warning: no text chunks extracted from {book_dir}")
        return 0

    _reset_dir(vector_db_path)
    Chroma.from_documents(
        documents=all_chunks,
        embedding=get_embeddings(),
        persist_directory=vector_db_path,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Saved {len(all_chunks)} chunks to {vector_db_path}")
    return len(all_chunks)


def vectorize_chapters(
    class_subject_name: str,
    subject_label: Optional[str] = None,
) -> int:
    book_dir = os.path.join(data_dir, class_subject_name)
    if not os.path.exists(book_dir):
        raise FileNotFoundError(f"Textbook directory not found: {book_dir}")

    if subject_label is None:
        subject_label = (
            "Biology (Class 12)"
            if "biology" in class_subject_name.lower()
            else class_subject_name
        )

    total = 0
    for chapter in sorted(os.listdir(book_dir)):
        if not chapter.lower().endswith(".pdf"):
            continue

        chapter_name = os.path.splitext(chapter)[0]
        chapter_pdf_path = os.path.join(book_dir, chapter)
        chunks = load_and_chunk_pdf(chapter_pdf_path, subject_label=subject_label)
        if not chunks:
            continue

        target_dir = os.path.join(chapters_vector_db_dir, chapter_name)
        _reset_dir(target_dir)
        Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=target_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        total += len(chunks)
        print(f"  Chapter '{chapter_name}' vectorized ({len(chunks)} chunks)")

    return total
