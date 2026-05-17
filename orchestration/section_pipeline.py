"""
Tek cagrida: indeks -> planner -> retrieval (adaptif esik) -> Writer.

Streamlit'ten bagimsiz calisabilmesi icin `main` import edilmez; Gemini kurulumu burada tekrarlanir.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.metadata_writer import MetadataWriter, extract_keywords_from_abstract
from agents.writer import AcademicWriter
from agents.diagram_planner import plan_diagrams
from orchestration.paper_blueprint import DEFAULT_PAPER_SECTIONS, combine_paper_markdown
from retriever import (
    build_rag_stack_for_repo,
    generate_planner_queries,
    heuristic_planner_queries,
    index_repository_files,
    retrieve_parent_contexts_multi_query,
)

logger = logging.getLogger(__name__)

_DEFAULT_CHAT = "gemini-3.1-flash-lite-preview"  # .env / env GEMINI_CHAT_MODEL ile ezilir


def _read_google_api_key() -> str:
    """Orchestration icin API anahtarini .env'den okur."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY bulunamadi. .env dosyasina ekleyin.")
    return api_key


def _resolve_chat_model_name() -> str:
    """GEMINI_CHAT_MODEL env degiskenini okur; yoksa varsayilan flash kullanilir."""
    load_dotenv()  # .env henuz yuklenmemisse (script modu / test) burada yukle
    name = os.getenv("GEMINI_CHAT_MODEL", "").strip()
    return name or _DEFAULT_CHAT


def _build_gemini_llm(model_name: str) -> ChatGoogleGenerativeAI:
    """Pipeline icin Gemini chat istemcisi."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=_read_google_api_key(),
        temperature=0.2,
    )


def _coerce_llm_content_to_text(content: Any) -> str:
    """
    Gemini SDK yeni surumlerinde .content alanini list[dict] (ornek: [{"type":"text","text":"..."}])
    olarak donderebiliyor; str() temsili Python list literal'i basar ve donanyadaki tum parser'lari
    (JSON, Markdown, regex) bozar. Bu helper liste girdisini parcalardaki text alanlarini birlestirip
    duz string'e cevirir; str ise oldugu gibi dondurur.
    """
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                value = chunk.get("text") or chunk.get("content") or ""
                parts.append(str(value))
            else:
                parts.append(str(chunk))
        return "".join(parts)
    return str(content)


def _invoke_gemini_chat_with_retry(llm: BaseChatModel, prompt: str, *, max_attempts: int = 5) -> str:
    """503/429 vb. gecici hatalarda sinirli retry."""
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return _coerce_llm_content_to_text(llm.invoke(prompt).content).strip()
        except BaseException as exc:  # noqa: BLE001
            last_exc = exc
            err_s = str(exc).lower()
            # Quota/HTTP hatalarinin yaninda Windows DNS (getaddrinfo) ve genel socket
            # baglanti kopukluklarini (connection reset/aborted) da retryable kabul ediyoruz.
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
                    "getaddrinfo",
                    "connection",
                    "temporary failure",
                )
            )
            if not retryable or attempt == max_attempts - 1:
                raise
            sleep_s = 2.0 * (2**attempt)
            time.sleep(sleep_s)
    raise RuntimeError(str(last_exc)) from last_exc


def _read_retriever_totals(retriever: Any) -> dict[str, int]:
    """
    Mevcut bir retriever'dan gercek Chroma child ve docstore parent sayisini okur.

    Yeni indeksleme yapilmadigi icin 'files' her zaman 0 gelir; parent ve child
    sayilari dogrudan Chroma koleksiyonu ve docstore'dan alinir.
    """
    n_children = 0
    n_parents = 0
    try:
        vs = retriever.vectorstore
        n_children = int(vs._collection.count())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        try:
            sample = retriever.vectorstore.get(limit=1)
            n_children = 1 if (sample and sample.get("ids")) else 0
        except Exception:  # noqa: BLE001
            pass
    try:
        for _ in retriever.docstore.yield_keys():
            n_parents += 1
    except Exception:  # noqa: BLE001
        pass
    return {"files": 0, "parents": n_parents, "children": n_children}


def _retrieve_parents_adaptive(
    retriever: Any,
    *,
    planner_queries: list[str],
    top_k_per_query: int,
    similarity_threshold: float,
) -> tuple[list[Document], str]:
    """
    Multi-query retrieval sonucunu dondurur; sonuc bos ve esik > 0.15 ise esigi dusurup bir kez daha dener.
    """
    docs = retrieve_parent_contexts_multi_query(
        retriever,
        planner_queries=list(planner_queries),
        top_k_per_query=int(top_k_per_query),
        similarity_threshold=float(similarity_threshold),
    )
    if docs:
        return list(docs), "success"
    if float(similarity_threshold) > 0.15:
        new_threshold = max(0.15, float(similarity_threshold) - 0.1)
        logger.info(
            "Retrieval bos; esik %.2f -> %.2f ile bir kez daha deneniyor.",
            similarity_threshold,
            new_threshold,
        )
        docs = retrieve_parent_contexts_multi_query(
            retriever,
            planner_queries=list(planner_queries),
            top_k_per_query=int(top_k_per_query),
            similarity_threshold=new_threshold,
        )
        if docs:
            return list(docs), "success_after_retry"
        return list(docs), "empty_after_retry"
    return list(docs), "empty_after_retry"


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
    writer_extra_rules: str = "",
    user_literature_block: str = "",
    existing_retriever: Any | None = None,
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
        if existing_retriever is not None:
            # UI'da daha once kurulmus retriever oturumunu yeniden kullan: 0 embed.
            retriever = existing_retriever
            totals = {"files": 0, "parents": 0, "children": 0, "reused": 1}
        else:
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
        docs, rstatus = _retrieve_parents_adaptive(
            retriever,
            planner_queries=list(queries),
            top_k_per_query=int(top_k_per_query),
            similarity_threshold=float(similarity_threshold),
        )
        result["retrieval_status"] = rstatus
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
            repository_url=repo_url,
            operator_addendum=writer_extra_rules,
            user_literature_block=user_literature_block,
        )
        result["writer_text"] = str(writer_out.get("text", ""))
        result["writer_metadata"] = dict(writer_out.get("metadata") or {})

    except Exception as e:  # noqa: BLE001
        logger.error("Pipeline hata (adim=%s): %s", step, e)
        result["error"] = str(e)
        result["failed_step"] = step

    return result


def run_paper_pipeline(
    *,
    repo_url: str,
    commit_hash: str,
    repo_root: Path,
    paths_for_index: list[Path],
    sections: list[tuple[str, str]] | None = None,
    max_index_files: int,
    similarity_threshold: float,
    top_k_per_query: int,
    max_planner_queries: int = 6,
    writer_extra_rules: str = "",
    user_literature_block: str = "",
    paper_title: str = "",
    abstract_text: str = "",
    keywords_text: str = "",
    existing_retriever: Any | None = None,
    diagram_mode: str = "none",
    manual_diagram_selection: list[str] | None = None,
) -> dict[str, Any]:
    """
    Tek indeksleme sonrasi ardışık bolumler: her biri icin planner -> retrieval -> writer.

    sections: (baslik, hedef) listesi; None ise DEFAULT_PAPER_SECTIONS kullanilir.
    """
    use_sections = list(sections) if sections else list(DEFAULT_PAPER_SECTIONS)
    # Literature Review sadece kullanici literatür metni saglamissa eklenir.
    _lit_provided = bool((user_literature_block or "").strip())
    use_sections = [
        (title, goal) for title, goal in use_sections
        if not (
            title.lower() in ("literature review", "related work")
            and not _lit_provided
        )
    ]
    # Kullanicidan gelenler bos kalabilir; bolum yazimindan sonra LLM tabanli metadata fallback devreye girer.
    repo_slug = repo_url.rstrip("/").split("/")[-1]
    user_title = paper_title.strip()
    user_abstract = abstract_text.strip()
    user_keywords = keywords_text.strip()
    combined_title = user_title if user_title else repo_slug
    combined_abstract = user_abstract
    combined_keywords = user_keywords
    out: dict[str, Any] = {
        "sections": [],
        "combined_markdown": "",
        "rag_totals": {},
        "retriever_ready": False,
        "error": None,
        "failed_step": None,
        "failed_section_index": None,
        "paper_title": combined_title,
        "abstract_text": combined_abstract,
        "keywords_text": combined_keywords,
    }
    section_blocks: list[dict[str, Any]] = []

    step = "init"
    try:
        paths = list(paths_for_index)[: max(1, int(max_index_files))]

        step = "indexing"
        if existing_retriever is not None:
            # UI'da hazir retriever varsa yeniden indekslemeden kullan (free-tier dostu).
            retriever = existing_retriever
            # Gercek Chroma sayisini oku; hard-coded 0 degil.
            totals = _read_retriever_totals(retriever)
            totals["reused"] = 1
        else:
            retriever, _store, _vs = build_rag_stack_for_repo(repo_url, commit_hash)
            totals = index_repository_files(
                retriever,
                paths,
                repo_root=repo_root,
                repo_url=repo_url,
                commit_hash=commit_hash,
            )
            # skip_if_indexed=True durumunda index_repository_files files=0 doner;
            # gercek sayiyi Chroma'dan tamamla.
            if totals.get("reused") == 1 and totals.get("files", 0) == 0:
                real = _read_retriever_totals(retriever)
                totals["parents"] = real["parents"]
                totals["children"] = real["children"]
        out["rag_totals"] = totals
        out["retriever_ready"] = True

        step = "llm_setup"
        llm = _build_gemini_llm(_resolve_chat_model_name())

        def _safe_invoke(prompt_text: str) -> str:
            return _invoke_gemini_chat_with_retry(llm, prompt_text)

        writer = AcademicWriter(llm_invoke_func=_safe_invoke)

        step = "diagram_planning"
        diagram_selections: list[str] = []
        if diagram_mode != "none":
            try:
                # LLM modunda gerçek repo context'i çek
                if diagram_mode == "llm":
                    planner_docs, _ = _retrieve_parents_adaptive(
                        retriever,
                        planner_queries=[
                            "system architecture main components",
                            "pipeline modules data flow",
                        ],
                        top_k_per_query=3,
                        similarity_threshold=0.3,
                    )
                    repo_context_for_planner = "\n\n".join(
                        d.page_content for d in planner_docs[:5]
                    )[:6000]
                else:
                    # all/manual modda context gerekmez, plan_diagrams zaten LLM çağırmaz
                    repo_context_for_planner = f"Repository URL: {repo_url}"

                diagram_selections = plan_diagrams(
                    repo_context=repo_context_for_planner,
                    llm_invoke_func=_safe_invoke,
                    mode=diagram_mode,
                    manual_selection=manual_diagram_selection,
                )
                logger.info("DiagramPlanner selected: %s", diagram_selections)
                out["diagram_selections"] = diagram_selections
            except Exception as _dex:  # noqa: BLE001
                logger.warning("DiagramPlanner failed (soft): %s", _dex)
                diagram_selections = []

        # Introduction abstract avoidance için ön-taslak
        # Kullanıcı abstract vermediyse repo URL'sinden basit bir placeholder oluştur
        abstract_for_intro_avoidance = combined_abstract
        if not abstract_for_intro_avoidance:
            abstract_for_intro_avoidance = (
                f"This paper presents an analysis of the "
                f"repository at {repo_url}. It covers the "
                f"system architecture, methodology, and "
                f"implementation details of the codebase."
            )

        for idx, (section_title, section_goal) in enumerate(use_sections):
            step = f"planner[{idx}]"
            queries = generate_planner_queries(
                llm,
                section_title=section_title,
                section_goal=section_goal,
                max_queries=max_planner_queries,
            )

            step = f"retrieval[{idx}]"
            docs, rstatus = _retrieve_parents_adaptive(
                retriever,
                planner_queries=list(queries),
                top_k_per_query=int(top_k_per_query),
                similarity_threshold=float(similarity_threshold),
            )

            step = f"writer[{idx}]"

            # Introduction için abstract'ı addendum olarak ekle
            extra_addendum = writer_extra_rules
            if "introduction" in section_title.lower() and abstract_for_intro_avoidance:
                intro_addendum = (
                    f"ABSTRACT AVOIDANCE — The following is the paper abstract. "
                    f"Do NOT copy, paraphrase, or mirror any sentence from it. "
                    f"Your Introduction must open from a completely different angle:\n\n"
                    f"ABSTRACT:\n{abstract_for_intro_avoidance}\n\n"
                    f"{writer_extra_rules}"
                ).strip()
                extra_addendum = intro_addendum

            writer_out = writer.generate_section(
                section_title=section_title,
                section_goal=section_goal,
                parent_documents=list(docs),
                max_parents=10,
                repository_url=repo_url,
                operator_addendum=extra_addendum,
                user_literature_block=user_literature_block,
            )
            meta = dict(writer_out.get("metadata") or {})
            # Oturum / markdown sismesin diye tam LLM cevabini coklu ciktilarda tutmayiz.
            meta.pop("full_response", None)

            block = {
                "section_index": idx,
                "section_title": section_title,
                "section_goal": section_goal,
                "planner_queries": list(queries),
                "retrieval_status": rstatus,
                "parents_retrieved": len(docs),
                "writer_text": str(writer_out.get("text", "")),
                "writer_metadata": meta,
                "faithfulness": None,
            }

            step = f"faithfulness_judge[{idx}]"
            try:
                from agents.faithfulness_judge import judge_section_faithfulness

                # writer.py _split_traceability'den gelen traceability tail'i ilk tercih.
                trace = meta.get("traceability", "").strip()
                writer_text_full = str(writer_out.get("text", ""))

                # Fallback: LLM 'TRACEABILITY:' marker'ini hic yazmamis veya farkli formatta yazmis olabilir.
                # Bu durumda writer cevabinda '| C1 |' ya da '| Claim ID |' kalibini arariz; bulursak
                # ham metni judge'a yine de geciririz cunku judge zaten regex ile claim satirlarini parse eder.
                if not trace:
                    has_claim_table = bool(
                        re.search(r"\|\s*C\d+\s*\|", writer_text_full)
                        or re.search(r"\|\s*Claim\s*ID\s*\|", writer_text_full, re.IGNORECASE)
                    )
                    if has_claim_table:
                        logger.info(
                            "Faithfulness judge [%s]: TRACEABILITY marker yok ama writer metninde "
                            "claim tablosu bulundu, raw text ile devam ediliyor.",
                            section_title,
                        )
                        trace = writer_text_full

                if trace:
                    block["faithfulness"] = judge_section_faithfulness(
                        writer_text=writer_text_full,
                        writer_traceability=trace,
                        parent_documents=list(docs),
                        llm_invoke=_safe_invoke,
                        user_literature_block=user_literature_block,
                    )
                    logger.info(
                        "Faithfulness judge [%s]: score=%.3f label=%s claims=%d",
                        section_title,
                        block["faithfulness"]["score"],
                        block["faithfulness"]["label"],
                        block["faithfulness"]["claim_count"],
                    )
                else:
                    if len(writer_text_full) > 500:
                        logger.info(
                            "Faithfulness judge [%s]: TRACEABILITY yok; uzun metin (%d karakter) "
                            "dogrudan judge ediliyor.",
                            section_title,
                            len(writer_text_full),
                        )
                        block["faithfulness"] = judge_section_faithfulness(
                            writer_text=writer_text_full,
                            writer_traceability=writer_text_full,
                            parent_documents=list(docs),
                            llm_invoke=_safe_invoke,
                            user_literature_block=user_literature_block,
                        )
                        logger.info(
                            "Faithfulness judge [%s]: score=%.3f label=%s claims=%d",
                            section_title,
                            block["faithfulness"]["score"],
                            block["faithfulness"]["label"],
                            block["faithfulness"]["claim_count"],
                        )
                    else:
                        logger.warning(
                            "Faithfulness judge skipped for '%s': TRACEABILITY yok ve metin kisa "
                            "(%d karakter). Fallback low-score.",
                            section_title,
                            len(writer_text_full),
                        )
                        block["faithfulness"] = {
                            "score": 0.0,
                            "label": "low",
                            "claim_count": 0,
                            "claims": [],
                            "raw_llm_response": "",
                            "judge_note": "traceability_missing",
                        }
            except Exception as _jex:  # noqa: BLE001
                logger.warning(
                    "Faithfulness judge failed (soft, section='%s'): %s",
                    section_title,
                    _jex,
                )
                # Soft fallback: beklenmeyen exception'larda da skor yapisini koru.
                block["faithfulness"] = {
                    "score": 0.0,
                    "label": "low",
                    "claim_count": 0,
                    "claims": [],
                    "raw_llm_response": "",
                    "judge_note": f"judge_exception:{type(_jex).__name__}",
                }

            step = f"post_judge[{idx}]"
            section_blocks.append(block)

        out["sections"] = section_blocks

        step = "metadata"
        # MetadataWriter ciktisi (md) yalnizca asagidaki try blogunda atanir; keyword satirinda md
        # okunurken atama olmayan dallarda UnboundLocalError olmasin diye baslangicta None tutulur.
        md = None
        # Tek LLM cagrisi: title + abstract + keywords (keywords pipeline sonunda onceliklenir).
        if (not user_title or not user_abstract) and section_blocks:
            try:
                combined_body_for_meta = "\n\n".join(
                    str(b.get("writer_text") or "") for b in section_blocks if b.get("writer_text")
                ).strip()
                if combined_body_for_meta:
                    abstract_queries = heuristic_planner_queries(
                        section_title="Abstract",
                        section_goal=(
                            "Identify core architectural components and technical contributions "
                            "for the paper abstract."
                        ),
                        max_queries=3,
                    )
                    abstract_docs, _ = _retrieve_parents_adaptive(
                        retriever,
                        planner_queries=abstract_queries,
                        top_k_per_query=int(top_k_per_query),
                        similarity_threshold=float(similarity_threshold),
                    )
                    md = MetadataWriter(llm_invoke_func=_safe_invoke).generate(
                        combined_body=combined_body_for_meta,
                        repo_url=repo_url,
                        rag_documents=abstract_docs,
                    )
                    if not user_title and md.get("title"):
                        combined_title = md["title"]
                    if not user_abstract and md.get("abstract"):
                        combined_abstract = md["abstract"]
            except Exception as meta_exc:  # noqa: BLE001
                logger.warning("MetadataWriter atlandi (yumusak hata): %s", meta_exc)

        # Keywords: kullanici girmediyse once MetadataWriter'in LLM keywords'u (tek cagri, ek kota yok),
        # yoksa veya bossa abstract uzerinden deterministic cikarim.
        if not user_keywords:
            if md and (md.get("keywords") or "").strip():
                combined_keywords = str(md["keywords"]).strip()
            elif combined_abstract:
                combined_keywords = extract_keywords_from_abstract(combined_abstract)

        out["paper_title"] = combined_title
        out["abstract_text"] = combined_abstract
        out["keywords_text"] = combined_keywords

        # Diagram planning moved up.

        step = "combine"
        out["combined_markdown"] = combine_paper_markdown(
            repo_url=repo_url,
            commit_hash=commit_hash,
            section_results=section_blocks,
            paper_title=combined_title,
            abstract_text=combined_abstract,
            keywords_text=combined_keywords,
            diagram_selections=diagram_selections,
            user_literature_block=user_literature_block,
        )

        # "all" modunda diyagramları üret ve kaydet
        if diagram_mode == "all" and diagram_selections:
            try:
                from export.diagram_renderer import generate_all_diagrams
                logger.info("Diyagramlar üretiliyor: %s", diagram_selections)
                # Ensure repo_context_for_planner is available; fallback to URL if not
                ctx = locals().get("repo_context_for_planner") or f"Repository URL: {repo_url}"
                diagram_paths = generate_all_diagrams(
                    repo_context=ctx,
                    llm_invoke_func=_safe_invoke,
                )
                out["diagram_paths"] = diagram_paths
                logger.info("Diyagramlar kaydedildi: %s", diagram_paths)
            except Exception as _dex:
                logger.warning("Diyagram üretimi başarısız (soft): %s", _dex)
                out["diagram_paths"] = {}

    except Exception as e:  # noqa: BLE001
        logger.error("Paper pipeline hata (adim=%s): %s", step, e)
        out["error"] = str(e)
        out["failed_step"] = step
        out["sections"] = section_blocks
        if section_blocks:
            out["combined_markdown"] = combine_paper_markdown(
                repo_url=repo_url,
                commit_hash=commit_hash,
                section_results=section_blocks,
                paper_title=combined_title,
                abstract_text=combined_abstract,
                keywords_text=combined_keywords,
                user_literature_block=user_literature_block,
            )
        m = re.match(r"^(planner|retrieval|writer|faithfulness_judge|post_judge)\[(\d+)\]$", str(step))
        if m:
            out["failed_section_index"] = int(m.group(2))

    return out
