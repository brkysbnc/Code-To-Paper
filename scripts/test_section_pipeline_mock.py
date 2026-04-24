import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os

# Python'un ana proje dizinini görmesini sağlayan sihirli satır:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Yeni yazdığımız pipeline modülünü import ediyoruz
from orchestration.section_pipeline import run_section_pipeline

class TestSectionPipelineAdaptiveRetrieval(unittest.TestCase):

    @patch("orchestration.section_pipeline.build_rag_stack_for_repo")
    @patch("orchestration.section_pipeline.index_repository_files")
    @patch("orchestration.section_pipeline._build_gemini_llm")
    @patch("orchestration.section_pipeline._get_cached_gemini_chat_model_name")
    @patch("orchestration.section_pipeline.generate_planner_queries")
    @patch("orchestration.section_pipeline.retrieve_parent_contexts_multi_query")
    @patch("orchestration.section_pipeline.AcademicWriter")
    def test_adaptive_retrieval_empty_after_retry(
        self, mock_writer, mock_retrieve, mock_planner, mock_get_model, 
        mock_build_llm, mock_index, mock_build_rag
    ):
        # 1. Mock Ayarları
        mock_index.return_value = {"total_files": 5, "indexed": 5}
        mock_planner.return_value = ["mock_query_1"]
        
        # EN ÖNEMLİ KISIM: Retriever'ın sürekli BOŞ dönmesini simüle ediyoruz
        mock_retrieve.return_value = []

        # Yazar modülünün sorunsuz geçmesini sağlıyoruz
        mock_writer_instance = MagicMock()
        mock_writer_instance.generate_section.return_value = {"text": "Yazı taslağı", "metadata": {}}
        mock_writer.return_value = mock_writer_instance

        # 2. Pipeline'ı Çalıştırma
        result = run_section_pipeline(
            repo_url="https://github.com/test/repo",
            commit_hash="abc1234",
            repo_root=Path("/tmp/fake_root"),
            paths_for_index=[Path("app.py")],
            section_title="Giriş",
            section_goal="Uygulamayı tanıt",
            max_index_files=10,
            similarity_threshold=0.20, # İlk eşik 0.20
            top_k_per_query=3
        )

        # 3. Doğrulamalar (Assertions)
        self.assertIsNone(result["error"])
        self.assertEqual(result["retrieval_status"], "empty_after_retry")
        
        # Retriever fonksiyonunun tam 2 kez çağrıldığını doğruluyoruz (Biri 0.20, diğeri düşürülmüş eşik)
        self.assertEqual(mock_retrieve.call_count, 2)
        
        # İkinci çağrının eşiğinin 0.1 düştüğünü (yani 0.10 olamayacağı için min 0.15 kuralına göre) doğruluyoruz
        args, kwargs = mock_retrieve.call_args_list[1]
        self.assertAlmostEqual(kwargs["similarity_threshold"], 0.15)
        
        print("✅ Başarılı: Boş retrieval yakalandı, eşik otomatik düşürülerek tekrar denendi.")

if __name__ == "__main__":
    unittest.main()