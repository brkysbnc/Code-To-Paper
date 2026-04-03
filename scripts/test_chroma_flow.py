import sys
import os
import uuid

# Proje kök dizinini Python yoluna ekle (Database klasörünü bulabilmesi için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.chroma_repository import upsert_chunks, search_by_embedding

def run_test():
    print("🚀 ChromaDB Testi Başlatılıyor...")

    # 1. Örnek Veri (Sanki GitHub'dan gelmiş gibi)
    sample_chunks = [
        {
            "text": "def calculate_area(radius): return 3.14 * radius ** 2",
            "source_repo": "https://github.com/user/math-lib",
            "file_path": "geometry.py",
            "content_hash": "hash123"
        }
    ]

    # 2. Kaydetme
    try:
        upsert_chunks(sample_chunks)
        
        # 3. Arama (OpenAI vektörü gibi 1536 boyutlu rastgele bir liste)
        print("🔍 Kaydedilen veri aranıyor...")
        mock_embedding = [0.1] * 384 
        results = search_by_embedding(mock_embedding, top_k=1)
        
        if results:
            print("✅ Test Başarılı! Veri bulundu.")
            print("Bulunan İçerik:", results['documents'][0][0])
    except Exception as e:
        print(f"❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    run_test()
    