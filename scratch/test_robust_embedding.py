import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retriever import ThrottledGeminiEmbeddings
from unittest.mock import MagicMock

def test_robustness():
    print("Testing ThrottledGeminiEmbeddings robustness...")
    
    # Mock the inner GoogleGenerativeAIEmbeddings
    mock_inner = MagicMock()
    
    # Simulate Gemini returning None for some documents
    # and returning a valid vector for others.
    # gemini-embedding-001 dimension is 768.
    valid_vector = [0.1] * 768
    mock_inner.embed_documents.side_effect = lambda texts: [valid_vector if t.strip() else None for t in texts]
    
    throttled = ThrottledGeminiEmbeddings(mock_inner, min_interval_s=0.01, batch_size=2)
    
    test_texts = ["Hello world", "", "   ", "Valid text"]
    
    try:
        print(f"Input texts: {test_texts}")
        results = throttled.embed_documents(test_texts)
        
        assert len(results) == 4
        assert results[0] == valid_vector
        assert results[1] == [0.0] * 768 # Should have been replaced with zero vector
        assert results[2] == [0.0] * 768 # Should have been replaced with zero vector
        assert results[3] == valid_vector
        
        print("Success: ThrottledGeminiEmbeddings handled empty strings and None returns correctly.")
        
    except Exception as e:
        print(f"Failure: {e}")
        import traceback
        traceback.print_exc()

def test_empty_api_list():
    """Inner API bazen tum batch icin [] doner; Chroma patlamasin diye tam uzunluk uretilmeli."""
    mock_inner = MagicMock()
    mock_inner.embed_documents.return_value = []
    throttled = ThrottledGeminiEmbeddings(mock_inner, min_interval_s=0.01, batch_size=10)
    results = throttled.embed_documents(["a", "b", "c"])
    assert len(results) == 3
    assert all(len(v) == 768 for v in results)
    assert results[0] == [0.0] * 768
    print("Success: empty API list padded to batch length with zero vectors.")


if __name__ == "__main__":
    test_robustness()
    test_empty_api_list()
