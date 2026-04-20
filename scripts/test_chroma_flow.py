import sys
import os

# Proje kök dizinini Python yoluna ekle (Database klasörünü bulabilmesi için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.chroma_repository import upsert_chunks, search_by_embedding

def run_test():
    print("ChromaDB testi baslatiliyor...")

    # 1) Parent + Child örneği:
    # Parent geniş bağlamı, child ise arama isabetini temsil eder.
    parent_id = "parent-geometry-calculate-area"
    sample_chunks = [
        {
            "chunk_id": parent_id,
            "doc_type": "parent",
            "parent_id": "",
            "text": "def calculate_area(radius):\n    pi = 3.14\n    return pi * radius ** 2",
            "repo_url": "https://github.com/user/math-lib",
            "commit_hash": "abc123def456",
            "file_path": "geometry.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "content_hash": "",
            "symbol": "calculate_area",
        },
        {
            "chunk_id": "child-geometry-calculate-area-return",
            "doc_type": "child",
            "parent_id": parent_id,
            "text": "return pi * radius ** 2",
            "repo_url": "https://github.com/user/math-lib",
            "commit_hash": "abc123def456",
            "file_path": "geometry.py",
            "language": "python",
            "start_line": 3,
            "end_line": 3,
            "content_hash": "",
            "symbol": "calculate_area",
        },
    ]

    # 2) Kaydetme (sözleşme doğrulaması dahil)
    try:
        upsert_chunks(sample_chunks)

        # 3) Arama (test için sabit boyutlu mock embedding)
        print("Kaydedilen veri araniyor...")
        mock_embedding = [0.1] * 384
        results = search_by_embedding(mock_embedding, top_k=1)

        if results:
            print("Test basarili! Veri bulundu.")
            print("Bulunan İçerik:", results["documents"][0][0])
    except Exception as e:
        print(f"Bir hata olustu: {e}")

if __name__ == "__main__":
    run_test()
    