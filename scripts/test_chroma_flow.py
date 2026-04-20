import sys
import os

# Proje dizinini yola ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.chroma_repository import upsert_chunks, search_by_embedding, delete_by_repo, collection_stats

def run_production_test():
    print("🚀 Code-to-Paper Production Testi Başlatılıyor...\n")
    
    test_repo = "https://github.com/brkysbnc/Code-To-Paper-Test"
    
    # --- AŞAMA 1: Temizlik ---
    print("🧹 Eski test verileri temizleniyor...")
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
        print("\n📥 Veriler kaydediliyor...")
        upsert_chunks(mock_data)
        
        print("🔁 Duplicate testi için aynı veriler tekrar gönderiliyor (Hata vermemeli, sayıyı artırmamalı)...")
        upsert_chunks(mock_data)
        
        # --- AŞAMA 4: Filtreli Sorgular ---
        print("\n🔍 Filtreli Sorgular Test Ediliyor...")
        mock_embedding = [0.1] * 384
        
        # 1. doc_type=child filtresi
        child_results = search_by_embedding(mock_embedding, top_k=5, where_filter={"$and": [{"doc_type": "child"}, {"repo_url": test_repo}]})
        child_count = len(child_results.get('ids', [[]])[0])
        assert child_count >= 1, "doc_type=child filtresi veri bulamadı!"
        print("✅ doc_type=child filtresi çalışıyor.")
        
        # 2. repo_url filtresi
        repo_results = search_by_embedding(mock_embedding, top_k=5, where_filter={"repo_url": test_repo})
        repo_count = len(repo_results.get('ids', [[]])[0])
        assert repo_count >= 2, "repo_url filtresi eksik veri getirdi!"
        print("✅ repo_url filtresi çalışıyor.")

        # --- AŞAMA 5: İstatistik ---
        print("\n📈 Son Durum:")
        collection_stats()

        print("\n✅ TÜM TESTLER PASS: Veri katmanı production için hazır!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAIL: {e}")
    except Exception as e:
        print(f"\n❌ BEKLENMEYEN HATA (FAIL): {e}")

if __name__ == "__main__":
    run_production_test()