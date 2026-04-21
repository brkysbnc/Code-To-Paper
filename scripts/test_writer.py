import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from agents.writer import AcademicWriter

def mock_invoke(prompt: str) -> str:
    context_part = prompt.split("CONTEXT:")[-1]
    
    if "def connect_db" not in context_part:
        return "Insufficient evidence in the repository to fully detail this section."
    
    return """
### Architecture Overview
This section explains the core design.

### Diagram
```mermaid
graph TD
    A[main.py] --> B[database.py]
```
(src/main.py:10-20)
"""

def run_tests():
    print("Writer modülü testleri başlatılıyor...\n")
    writer = AcademicWriter(llm_invoke_func=mock_invoke)
    
    print("TEST 1: Geçerli kod blokları gönderiliyor...")
    docs = [
        Document(
            page_content="def connect_db():\n    pass", 
            metadata={"file_path": "src/database.py", "start_line": 5, "end_line": 10}
        )
    ]
    res1 = writer.generate_section("Test Title", "Test Goal", docs)
    
    if "graph TD" in res1["text"]:
        print(" TEST 1 BAŞARILI: Mermaid diyagramı (graph TD) başarıyla üretildi.")
    else:
        print(" TEST 1 BAŞARISIZ: Mermaid diyagramı bulunamadı.")


    # --- TEST 2: Boş Context Gönderimi (Güvenlik Testi) ---
    print("\nTEST 2: Boş/alakasız kod blokları gönderiliyor...")
    res2 = writer.generate_section("Empty Title", "Empty Goal", [])
    
    if "Insufficient evidence" in res2["text"]:
        print(" TEST 2 BAŞARILI: Sistem boş context'te düzgün uyarı veriyor.")
    else:
        print(" TEST 2 BAŞARISIZ: Sistem boş context'i yakalayamadı.")
        
    print("\nTüm testler tamamlandı!")

if __name__ == "__main__":
    run_tests()