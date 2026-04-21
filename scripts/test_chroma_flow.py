import sys
import os

# Proje dizinini yola ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.chroma_repository import upsert_chunks, search_by_embedding, delete_by_repo, collection_stats

def run_production_test():
    print("Code-to-Paper production testi baslatiliyor...\n")
    
    test_repo = "https://github.com/brkysbnc/Code-To-Paper-Test"
    
    # --- AŞAMA 1: Temizlik ---
    print("Eski test verileri temizleniyor...")
    delete_by_repo(test_repo)
    
    # --- AŞAMA 2: Veri Hazırlama (Sözleşmeye Uygun Parent/Child) ---
    parent_chunk = {
        "text": "class Calculator:\n    def add(a, b):\n        return a+b",
        "repo_url": test_repo,
        "commit_hash": "main",
        "file_path": "backend/calc.py",
        "content_hash": "hash_parent_001",
        "doc_type": "parent",
        "language": "python",
        "start_line": 1,
        "end_line": 3
    }
    
    child_chunk = {
        "text": "def add(a, b):\n        return a+b",
        "repo_url": test_repo,
        "commit_hash": "main",
        "file_path": "backend/calc.py",
        "content_hash": "hash_child_001",
        "doc_type": "child",
        "parent_id": "hash_parent_001", # Child için parent_id zorunludur
        "language": "python",
        "start_line": 2,
        "end_line": 3
    }
    
    mock_data = [parent_chunk, child_chunk]
    
    try:
        # --- AŞAMA 3: Upsert & Idempotent (Çift Kayıt) Testi ---
        print("\nVeriler kaydediliyor...")
        upsert_chunks(mock_data)
        
        print("Duplicate testi icin ayni veriler tekrar gonderiliyor...")
        upsert_chunks(mock_data)

        # Idempotent davranisi dogrulamak icin toplam kayit sayisini kontrol et.
        idempotent_check = search_by_embedding(
            [0.1] * 384,
            top_k=10,
            where_filter={"repo_url": test_repo},
        )
        upserted_count = len(idempotent_check.get("ids", [[]])[0])
        assert upserted_count == 2, (
            f"Idempotent test FAIL: beklenen 2 kayit, bulunan {upserted_count}"
        )
        print("Idempotent upsert dogrulandi.")
        
        # --- AŞAMA 4: Filtreli Sorgular ---
        print("\nFiltreli sorgular test ediliyor...")
        mock_embedding = [0.1] * 384
        
        # 1. doc_type=child filtresi
        child_results = search_by_embedding(mock_embedding, top_k=5, where_filter={"$and": [{"doc_type": "child"}, {"repo_url": test_repo}]})
        child_count = len(child_results.get('ids', [[]])[0])
        assert child_count >= 1, "doc_type=child filtresi veri bulamadı!"
        print("doc_type=child filtresi calisiyor.")
        
        # 2. repo_url filtresi
        repo_results = search_by_embedding(mock_embedding, top_k=5, where_filter={"repo_url": test_repo})
        repo_count = len(repo_results.get('ids', [[]])[0])
        assert repo_count >= 2, "repo_url filtresi eksik veri getirdi!"
        print("repo_url filtresi calisiyor.")

        # --- AŞAMA 5: İstatistik ---
        print("\nSon durum:")
        collection_stats()

        # --- AŞAMA 6: Repo bazli temizleme dogrulamasi ---
        delete_by_repo(test_repo)
        after_delete = search_by_embedding(
            mock_embedding,
            top_k=5,
            where_filter={"repo_url": test_repo},
        )
        remaining = len(after_delete.get("ids", [[]])[0])
        assert remaining == 0, f"delete_by_repo FAIL: kalan kayit sayisi {remaining}"
        print("delete_by_repo dogrulandi.")

        print("\nTUM TESTLER PASS: Veri katmani production icin hazir.")
        
    except AssertionError as e:
        print(f"\nTEST FAIL: {e}")
    except Exception as e:
        print(f"\nBEKLENMEYEN HATA (FAIL): {e}")

if __name__ == "__main__":
    run_production_test()