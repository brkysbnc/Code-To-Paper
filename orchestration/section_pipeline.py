import logging
from pathlib import Path
from typing import Any, Dict, List

# Projedeki mevcut modüllerden gerekli importlar (DRY Prensibi)
from retriever import (
    build_rag_stack_for_repo,
    index_repository_files,
    generate_planner_queries,
    retrieve_parent_contexts_multi_query
)
from agents.writer import AcademicWriter
from main import (
    _build_gemini_llm,
    _get_cached_gemini_chat_model_name,
    _invoke_gemini_chat_with_retry
)

logger = logging.getLogger(__name__)

def run_section_pipeline(
    *,
    repo_url: str,
    commit_hash: str,
    repo_root: Path,
    paths_for_index: List[Path],
    section_title: str,
    section_goal: str,
    max_index_files: int,
    similarity_threshold: float,
    top_k_per_query: int,
    max_planner_queries: int = 6
) -> Dict[str, Any]:
    
    # Teslimatta istenen sabit dönüş formatı
    result = {
        "planner_queries": [],
        "retrieved_parent_docs": [],
        "writer_text": "",
        "writer_metadata": {},
        "rag_totals": {},
        "retriever_ready": False,
        "error": None,
        "failed_step": None,
        "retrieval_status": "success" # Olası durumlar: success, success_after_retry, empty_after_retry
    }

    step = "init"
    try:
        # 1. Retriever + Chroma (İndeksleme)
        step = "indexing"
        vectorstore = build_rag_stack_for_repo(repo_url=repo_url, commit_hash=commit_hash)
        index_stats = index_repository_files(
            repo_root=repo_root,
            paths=paths_for_index,
            vectorstore=vectorstore,
            max_files=max_index_files
        )
        result["rag_totals"] = index_stats
        result["retriever_ready"] = True

        # 2. LLM Hazırlığı
        step = "llm_setup"
        model_name = _get_cached_gemini_chat_model_name()
        llm = _build_gemini_llm(model_name=model_name)

        # 3. Planner Sorguları
        step = "planner"
        queries = generate_planner_queries(
            llm=llm,
            section_title=section_title,
            section_goal=section_goal,
            max_queries=max_planner_queries
        )
        result["planner_queries"] = queries

        # 4. Retrieval + Kural Tabanlı Adaptif Esnetme
        step = "retrieval"
        docs = retrieve_parent_contexts_multi_query(
            vectorstore=vectorstore,
            queries=queries,
            similarity_threshold=similarity_threshold,
            top_k=top_k_per_query
        )

        # Adaptif Kural: Boş döndüyse ve threshold uygunsa, 1 kez eşiği düşür
        if not docs and similarity_threshold > 0.15:
            new_threshold = max(0.15, similarity_threshold - 0.1)
            logger.info(f"Retrieval boş döndü. Eşik {similarity_threshold}'dan {new_threshold}'a düşürülerek tekrar deneniyor.")
            
            docs = retrieve_parent_contexts_multi_query(
                vectorstore=vectorstore,
                queries=queries,
                similarity_threshold=new_threshold,
                top_k=top_k_per_query
            )
            
            if not docs:
                result["retrieval_status"] = "empty_after_retry"
            else:
                result["retrieval_status"] = "success_after_retry"

        # Dönen dokümanlar listesini LangChain formatında (UI'da serialize edilmek üzere) ekliyoruz
        result["retrieved_parent_docs"] = docs

        # 5. Writer Adımı
        step = "writer"
        writer = AcademicWriter(llm=llm)
        # safe_invoke benzeri callable kullanımı
        writer_output = writer.generate_section(
            section_title=section_title,
            section_goal=section_goal,
            docs=docs,
            invoke_fn=_invoke_gemini_chat_with_retry
        )
        
        # Eğer yazar sözlük döndürüyorsa ayırıyoruz
        if isinstance(writer_output, dict):
            result["writer_text"] = writer_output.get("text", str(writer_output))
            result["writer_metadata"] = writer_output.get("metadata", {})
        else:
            result["writer_text"] = str(writer_output)

    except Exception as e:
        logger.error(f"Boru hattı hatası '{step}' adımında gerçekleşti: {str(e)}")
        result["error"] = str(e)
        result["failed_step"] = step

    return result