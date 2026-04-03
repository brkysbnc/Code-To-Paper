import uuid
from database.chroma_client import get_collection

# Koleksiyonu başlat
collection = get_collection()

def upsert_chunks(chunks):
    """Kod parçalarını metadata ile birlikte kaydeder."""
    # Gelen listeyi ChromaDB'nin anlayacağı parçalara ayırıyoruz
    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [c.get('text', '') for c in chunks]
    metadatas = [{
        "source_repo": c.get('source_repo', ''),
        "file_path": c.get('file_path', ''),
        "content_hash": c.get('content_hash', '')
    } for c in chunks]
    
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"{len(chunks)} parça başarıyla kaydedildi.")

def search_by_embedding(query_embedding, top_k=5):
    """Vektör bazlı arama yapar."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results

def delete_by_repo(source_repo):
    """Belirli bir repoya ait tüm verileri siler."""
    collection.delete(where={"source_repo": source_repo})
    print(f"{source_repo} reposuna ait veriler temizlendi.")
