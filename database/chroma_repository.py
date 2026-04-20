"""
Chroma veri katmanı.

Bu modül, chunk verilerini Chroma'ya yazmadan önce ortak sözleşmeye göre
doğrular. Böylece takım içi alan adı farklılıklarından doğan entegrasyon
hataları erken aşamada yakalanır.
"""

from __future__ import annotations

from typing import Any, Iterable

from database.chroma_client import get_collection
from database.chunk_contract import build_chroma_payload

# Koleksiyonu başlat
collection = get_collection()

def upsert_chunks(chunks: Iterable[dict[str, Any]]) -> None:
    """
    Chunk listesini standart metadata şemasıyla Chroma'ya yazar.

    Not:
        - `database.chunk_contract` içindeki zorunlu alanlar denetlenir.
        - Geçersiz chunk görüldüğünde ValueError fırlatılır.
    """
    payload = build_chroma_payload(chunks)
    if not payload["ids"]:
        return

    collection.upsert(
        ids=payload["ids"],
        documents=payload["documents"],
        metadatas=payload["metadatas"],
    )
    print(f"{len(payload['ids'])} parça başarıyla kaydedildi.")


def search_by_embedding(query_embedding: list[float], top_k: int = 5) -> dict[str, Any]:
    """Vektör bazlı arama yapar ve ham Chroma sonucunu döndürür."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results


def delete_by_repo(repo_url: str) -> None:
    """
    Belirli bir repoya ait tüm verileri siler.

    Standarda göre filtre alanı `repo_url` olarak kullanılır.
    """
    collection.delete(where={"repo_url": repo_url})
    print(f"{repo_url} reposuna ait veriler temizlendi.")
