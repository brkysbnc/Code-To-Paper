"""
Tek cagrida: indeks -> planner -> retrieval (adaptif esik) -> Writer.

Streamlit'ten bagimsiz calisabilmesi icin `main` import edilmez; Gemini kurulumu burada tekrarlanir.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.writer import AcademicWriter
from retriever import (
    build_rag_stack_for_repo,
    generate_planner_queries,
    index_repository_files,
    retrieve_parent_contexts_multi_query,
)

logger = logging.getLogger(__name__)

_DEFAULT_CHAT = "gemini-2.5-flash"


def _read_google_api_key() -> str:
    """Orchestration icin API anahtarini .env'den okur."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY bulunamadi. .env dosyasina ekleyin.")
    return api_key


def _resolve_chat_model_name() -> str:
    """Streamlit disi: GEMINI_CHAT_MODEL veya varsayilan flash."""
    name = os.getenv("GEMINI_CHAT_MODEL", "").strip()
    return name or _DEFAULT_CHAT


def _build_gemini_llm(model_name: str) -> ChatGoogleGenerativeAI:
    """Pipeline icin Gemini chat istemcisi."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=_read_google_api_key(),
        temperature=0.2,
    )


def _invoke_gemini_chat_with_retry(llm: BaseChatModel, prompt: str, *, max_attempts: int = 5) -> str:
    """503/429 vb. gecici hatalarda sinirli retry."""
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return str(llm.invoke(prompt).content).strip()
        except BaseException as exc:  # noqa: BLE001
            last_exc = exc
            err_s = str(exc).lower()
            retryable = any(
                x in err_s
                for x in (
                    "503",
                    "429",
                    "unavailable",
                    "resource_exhausted",
                    "quota",
                    "deadline",
                    "timeout",
                )
            )
            if not retryable or attempt == max_attempts - 1:
                raise
            sleep_s = 2.0 * (2**attempt)
            time.sleep(sleep_s)
    raise RuntimeError(str(last_exc)) from last_exc


def run_section_pipeline(
    *,
    repo_url: str,
    commit_hash: str,
    repo_root: Path,
    paths_for_index: list[Path],
    section_title: str,
    section_goal: str,
    max_index_files: int,
    similarity_threshold: float,
    top_k_per_query: int,
    max_planner_queries: int = 6,
) -> dict[str, Any]:
    """
    RAG + Writer zincirini uygular.

    paths_for_index: mutlak veya repo koku gore cozulebilir dosya yollari;
    max_index_files ile bas kesilir.
    """
    result: dict[str, Any] = {
        "planner_queries": [],
        "retrieved_parent_docs": [],
        "writer_text": "",
        "writer_metadata": {},
        "rag_totals": {},
        "retriever_ready": False,
        "error": None,
        "failed_step": None,
        "retrieval_status": "success",
    }

    step = "init"
    try:
        paths = list(paths_for_index)[: max(1, int(max_index_files))]

        step = "indexing"
        retriever, _store, _vs = build_rag_stack_for_repo(repo_url, commit_hash)
        totals = index_repository_files(
            retriever,
            paths,
            repo_root=repo_root,
            repo_url=repo_url,
            commit_hash=commit_hash,
        )
        result["rag_totals"] = totals
        result["retriever_ready"] = True

        step = "llm_setup"
        llm = _build_gemini_llm(_resolve_chat_model_name())

        step = "planner"
        queries = generate_planner_queries(
            llm,
            section_title=section_title,
            section_goal=section_goal,
            max_queries=max_planner_queries,
        )
        result["planner_queries"] = queries

        step = "retrieval"
        docs = retrieve_parent_contexts_multi_query(
            retriever,
            planner_queries=list(queries),
            top_k_per_query=int(top_k_per_query),
            similarity_threshold=float(similarity_threshold),
        )

        if docs:
            result["retrieval_status"] = "success"
        elif float(similarity_threshold) > 0.15:
            new_threshold = max(0.15, float(similarity_threshold) - 0.1)
            logger.info(
                "Retrieval bos; esik %.2f -> %.2f ile bir kez daha deneniyor.",
                similarity_threshold,
                new_threshold,
            )
            docs = retrieve_parent_contexts_multi_query(
                retriever,
                planner_queries=list(queries),
                top_k_per_query=int(top_k_per_query),
                similarity_threshold=new_threshold,
            )
            if docs:
                result["retrieval_status"] = "success_after_retry"
            else:
                result["retrieval_status"] = "empty_after_retry"
        else:
            result["retrieval_status"] = "empty_after_retry"

        result["retrieved_parent_docs"] = docs

        step = "writer"

        def _safe_invoke(prompt_text: str) -> str:
            return _invoke_gemini_chat_with_retry(llm, prompt_text)

        writer = AcademicWriter(llm_invoke_func=_safe_invoke)
        writer_out = writer.generate_section(
            section_title=section_title,
            section_goal=section_goal,
            parent_documents=list(docs),
            max_parents=10,
        )
        result["writer_text"] = str(writer_out.get("text", ""))
        result["writer_metadata"] = dict(writer_out.get("metadata") or {})

    except Exception as e:  # noqa: BLE001
        logger.error("Pipeline hata (adim=%s): %s", step, e)
        result["error"] = str(e)
        result["failed_step"] = step

    return result
