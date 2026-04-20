"""
Chunk metadata sözleşmesi ve doğrulama yardımcıları.

Bu modülün amacı, ekipteki herkesin Chroma'ya aynı alan adlarıyla veri yazmasını
zorunlu hale getirmektir. Böylece retrieval, writer ve değerlendirme adımları
ortak bir veri standardı üzerinde çalışır.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, Iterable, List

# Child/Parent ayrımı RAG akışında zorunlu olduğu için sabit olarak tutulur.
DOC_TYPE_PARENT = "parent"
DOC_TYPE_CHILD = "child"
VALID_DOC_TYPES = {DOC_TYPE_PARENT, DOC_TYPE_CHILD}

# Takımın kullanacağı tek metadata şeması.
# Not: Chroma metadata değerleri düz tiplerde tutulmalıdır.
STANDARD_METADATA_FIELDS = (
    "chunk_id",
    "parent_id",
    "doc_type",
    "repo_url",
    "commit_hash",
    "file_path",
    "language",
    "start_line",
    "end_line",
    "content_hash",
    "symbol",
)

# Bu alanlar olmadan anlamlı retrieval kurulamaz.
REQUIRED_FIELDS = (
    "doc_type",
    "repo_url",
    "file_path",
    "text",
)


def _safe_str(value: Any, default: str = "") -> str:
    """Değeri güvenli biçimde string'e çevirir."""
    if value is None:
        return default
    return str(value)


def _safe_int(value: Any, default: int = -1) -> int:
    """Değeri güvenli biçimde integer'a çevirir."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _content_hash(text: str) -> str:
    """Chunk text içeriğinden deterministic SHA-256 üretir."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_chunk(raw_chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ham chunk verisini takım standardına normalize eder.

    Geriye dönük uyumluluk için eski alan adlarını da (ör. source_repo) kabul eder.
    """
    text = _safe_str(raw_chunk.get("text")).strip()

    # Geriye dönük alan adı desteği:
    # - source_repo -> repo_url
    # - chunk_type  -> doc_type
    repo_url = _safe_str(raw_chunk.get("repo_url") or raw_chunk.get("source_repo")).strip()
    doc_type = _safe_str(raw_chunk.get("doc_type") or raw_chunk.get("chunk_type")).strip().lower()

    chunk_id = _safe_str(raw_chunk.get("chunk_id")).strip() or str(uuid.uuid4())
    parent_id = _safe_str(raw_chunk.get("parent_id")).strip()
    commit_hash = _safe_str(raw_chunk.get("commit_hash")).strip()
    file_path = _safe_str(raw_chunk.get("file_path")).strip()
    language = _safe_str(raw_chunk.get("language")).strip().lower()
    symbol = _safe_str(raw_chunk.get("symbol")).strip()

    start_line = _safe_int(raw_chunk.get("start_line"), -1)
    end_line = _safe_int(raw_chunk.get("end_line"), -1)

    content_hash = _safe_str(raw_chunk.get("content_hash")).strip() or _content_hash(text)

    return {
        "chunk_id": chunk_id,
        "parent_id": parent_id,
        "doc_type": doc_type,
        "repo_url": repo_url,
        "commit_hash": commit_hash,
        "file_path": file_path,
        "language": language,
        "start_line": start_line,
        "end_line": end_line,
        "content_hash": content_hash,
        "symbol": symbol,
        "text": text,
    }


def validate_chunk(chunk: Dict[str, Any]) -> None:
    """
    Normalize edilmiş chunk için zorunlu kuralları doğrular.

    Kurallardan biri sağlanmazsa ValueError fırlatır.
    """
    for field in REQUIRED_FIELDS:
        value = chunk.get(field)
        if isinstance(value, str):
            value = value.strip()
        if value in ("", None):
            raise ValueError(f"Zorunlu alan eksik: {field}")

    if chunk["doc_type"] not in VALID_DOC_TYPES:
        raise ValueError(
            f"Gecersiz doc_type: {chunk['doc_type']}. "
            f"Kabul edilenler: {sorted(VALID_DOC_TYPES)}"
        )

    # Child chunk parent'a bağlı olmalı; parent chunk ise parent_id boş olabilir.
    if chunk["doc_type"] == DOC_TYPE_CHILD and not chunk.get("parent_id"):
        raise ValueError("Child chunk icin parent_id zorunludur.")

    start_line = chunk.get("start_line", -1)
    end_line = chunk.get("end_line", -1)
    if start_line >= 0 and end_line >= 0 and start_line > end_line:
        raise ValueError("Line araligi gecersiz: start_line end_line'dan buyuk olamaz.")


def build_chroma_payload(raw_chunks: Iterable[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Ham chunk listesi için Chroma upsert payload'u üretir.

    Returns:
        {
            "ids": [...],
            "documents": [...],
            "metadatas": [...],
        }
    """
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for raw in raw_chunks:
        normalized = normalize_chunk(raw)
        validate_chunk(normalized)

        ids.append(normalized["chunk_id"])
        documents.append(normalized["text"])
        metadatas.append({k: normalized[k] for k in STANDARD_METADATA_FIELDS})

    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
    }
