"""
Chroma veri katmanı.

Bu modül, chunk verilerini Chroma'ya yazmadan önce ortak sözleşmeye göre
doğrular. Böylece takım içi alan adı farklılıklarından doğan entegrasyon
hataları erken aşamada yakalanır.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Iterable, Optional

from database.chroma_client import get_collection
from database.chunk_contract import build_chroma_payload

# Basit query timing ve sağlık log'u için (Scrum Master isteği)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Koleksiyonu başlat
collection = get_collection()

def upsert_chunks(chunks: Iterable[dict[str, Any]]) -> None:
    """
    Chunk listesini standart metadata şemasıyla Chroma'ya yazar.
    Idempotent yapı chunk_contract içindeki build_chroma_payload ile sağlanır.
    
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
    logger.info("%s chunk upsert edildi.", len(payload["ids"]))

def search_by_embedding(query_embedding: list[float], top_k: int = 5, where_filter: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Vektör bazlı arama yapar ve ham Chroma sonucunu döndürür.
    Scrum Master İsteği: where_filter eklendi, süre ölçümü eklendi.
    """
    start_time = time.time()
    
    # Eğer filtre verilmemişse None geç, verilmişse kullan
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k
    }
    if where_filter:
        query_params["where"] = where_filter

    results = collection.query(**query_params)
    
    elapsed_ms = (time.time() - start_time) * 1000
    filter_msg = f" | Filtre: {where_filter}" if where_filter else ""
    logger.info("Arama %.2f ms surdu.%s", elapsed_ms, filter_msg)
    
    return results

def delete_by_repo(repo_url: str, commit_hash: Optional[str] = None) -> None:
    """
    Belirli bir repoya ait tüm verileri siler.
    Scrum Master İsteği: Gerekirse commit_hash bazlı filtreli silmeyi destekle.
    """
    where_clause = {"repo_url": repo_url}
    if commit_hash:
        where_clause["commit_hash"] = commit_hash
        
    collection.delete(where=where_clause)
    logger.info("Veriler temizlendi -> Repo: %s | Commit: %s", repo_url, commit_hash or "Tumu")

def collection_stats() -> dict[str, int]:
    """
    Scrum Master İsteği: Koleksiyon count, repo bazlı count dönsun.
    """
    count = collection.count()
    logger.info("Mevcut koleksiyon boyutu: %s chunk", count)
    return {"total_count": count}