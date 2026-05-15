"""
No-RAG tam makale uretim kosucusu.

Bu script mevcut `run_paper_pipeline()` akisina olabildigince sadik kalir;
tek fark retrieval/embedding adimini tamamen atlayip secili repo dosyalarini
tek bir duz CONTEXT listesi olarak AcademicWriter'a vermesidir.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from git import Repo
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.faithfulness_judge import judge_section_faithfulness
from agents.metadata_writer import MetadataWriter, extract_keywords_from_abstract
from agents.writer import AcademicWriter
from github_handler import iter_indexable_files
from orchestration.paper_blueprint import DEFAULT_PAPER_SECTIONS, combine_paper_markdown
from orchestration.section_pipeline import (
    _build_gemini_llm,
    _invoke_gemini_chat_with_retry,
    _resolve_chat_model_name,
)
from retriever import (
    _read_document,
    should_exclude_from_writer_context,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_PATH = Path("data/norag_results.json")
_MAX_INDEX_FILE_BYTES = 250_000
_INDEX_SUFFIX_PRIORITY = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".md",
    ".sql",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
)


def _configure_logging(level: int = logging.INFO) -> None:
    """Ayni process icinde tekrarli handler eklemeden okunabilir log ayari yapar."""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_sections(user_literature_block: str) -> list[tuple[str, str]]:
    """
    run_paper_pipeline ile ayni bolum secim mantigini uygular.

    Kullanici literatur saglamadiysa Literature Review / Related Work bolumlerini
    listeden cikarir ki iki mod da birebir ayni section set'iyle kosulsun.
    """
    lit_provided = bool((user_literature_block or "").strip())
    return [
        (title, goal)
        for title, goal in DEFAULT_PAPER_SECTIONS
        if not (
            title.lower() in ("literature review", "related work")
            and not lit_provided
        )
    ]


def _make_retry_invoke(llm) -> Any:
    """Writer ve judge icin ortak retry-sarmalli LLM invoke fonksiyonu dondurur."""

    def _safe_invoke(prompt_text: str) -> str:
        return _invoke_gemini_chat_with_retry(llm, prompt_text)

    return _safe_invoke


def _truncate_document_for_budget(document: Document, remaining_chars: int) -> Document:
    """
    No-RAG baglam limiti asildiginda son dokumani kalan karakter butcesine gore kirpar.

    Metadata'daki end_line'i de kirpilmis icerige gore guncelleyerek Writer/Judge
    tarafinda daha tutarli satir araligi gorulmesini saglar.
    """
    suffix = "\n...[CONTENT TRUNCATED FOR NO-RAG]"
    room_for_text = max(0, remaining_chars - len(suffix))
    truncated_text = (document.page_content or "")[:room_for_text]
    if len(document.page_content or "") > room_for_text:
        truncated_text += suffix

    new_metadata = dict(document.metadata or {})
    new_metadata["end_line"] = truncated_text.count("\n") + (1 if truncated_text else 0)
    new_metadata["norag_truncated"] = True
    return Document(page_content=truncated_text, metadata=new_metadata)


def build_norag_context_documents(
    file_paths: list[Path],
    repo_root: Path,
    repo_url: str,
    commit_hash: str,
    max_total_chars: int = 60000,
) -> list[Document]:
    """
    No-RAG mode icin secili repo dosyalarini dogrudan Writer CONTEXT'ine hazirlar.

    Davranis:
    - Dosyalari retriever._read_document ile ayni metadata formatiyla okur
    - Writer CONTEXT disi kalmasi gereken dosyalari ayni filtreyle atlar
    - Tum icerigi toplam max_total_chars limiti icinde tutar; son dosya gerekiyorsa kirpilir
    - AcademicWriter'in bekledigi Document listesi formatini korur
    """
    docs: list[Document] = []
    total_chars = 0
    files_used = 0
    repo_root_path = Path(repo_root).resolve()
    cap = max(1, int(max_total_chars))

    for raw_path in file_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            continue

        document = _read_document(
            path,
            repo_root=repo_root_path,
            repo_url=repo_url,
            commit_hash=commit_hash,
        )
        rel_path = str(document.metadata.get("file_path") or "")
        if should_exclude_from_writer_context(
            relative_path=rel_path,
            text=document.page_content or "",
        ):
            LOGGER.info("No-RAG context skip (writer exclude): %s", rel_path)
            continue

        remaining = cap - total_chars
        if remaining <= 0:
            break

        content_len = len(document.page_content or "")
        if content_len <= remaining:
            docs.append(document)
            total_chars += content_len
            files_used += 1
            continue

        docs.append(_truncate_document_for_budget(document, remaining))
        total_chars = cap
        files_used += 1
        break

    LOGGER.info(
        "No-RAG context hazirlandi: files=%s total_chars=%s limit=%s",
        files_used,
        total_chars,
        cap,
    )
    return docs


def _judge_writer_output_like_pipeline(
    *,
    section_title: str,
    writer_out: dict[str, Any],
    parent_documents: list[Document],
    llm_invoke,
    user_literature_block: str,
) -> dict[str, Any]:
    """
    section_pipeline.py ile ayni traceability fallback mantigiyla faithfulness hesaplar.

    Writer bazen `traceability` metadata'sini bos birakip tabloyu ham metnin icinde
    birakabildigi icin, pipeline'daki ayni regex fallback burada da kullanilir.
    """
    meta = dict(writer_out.get("metadata") or {})
    trace = str(meta.get("traceability") or "").strip()
    writer_text_full = str(writer_out.get("text") or "")

    if not trace:
        has_claim_table = bool(
            re.search(r"\|\s*C\d+\s*\|", writer_text_full)
            or re.search(r"\|\s*Claim\s*ID\s*\|", writer_text_full, re.IGNORECASE)
        )
        if has_claim_table:
            LOGGER.info(
                "No-RAG judge [%s]: TRACEABILITY marker yok ama claim tablosu bulundu.",
                section_title,
            )
            trace = writer_text_full

    if trace:
        return judge_section_faithfulness(
            writer_text=writer_text_full,
            writer_traceability=trace,
            parent_documents=list(parent_documents),
            llm_invoke=llm_invoke,
            user_literature_block=user_literature_block,
        )

    LOGGER.warning(
        "No-RAG judge skipped for '%s': traceability bulunamadi, fallback low-score uygulanacak.",
        section_title,
    )
    return {
        "score": 0.0,
        "label": "low",
        "claim_count": 0,
        "claims": [],
        "raw_llm_response": "",
        "judge_note": "traceability_missing",
    }


def _mean(values: list[float]) -> float:
    """Bos liste durumunda 0.0 donduren guvenli ortalama yardimcisi."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _save_results_json(results: dict[str, Any], output_path: Path) -> None:
    """Ablasyon sonucunu dizin olusturarak JSON dosyasina yazar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_norag_full_paper(
    *,
    repo_url: str,
    commit_hash: str,
    repo_root: Path,
    paths_for_index: list[Path],
    user_literature_block: str = "",
    writer_extra_rules: str = "",
    paper_title: str = "",
    abstract_text: str = "",
    keywords_text: str = "",
    max_index_files: int = 50,
    max_total_chars: int = 60000,
) -> dict[str, Any]:
    """
    Retrieval olmadan tam makale uretir; writer/judge/combine/metadata akislarini
    `run_paper_pipeline()` ile ayni yapida korur.
    """
    selected_paths = list(paths_for_index)[: max(1, int(max_index_files))]
    repo_root_path = Path(repo_root).resolve()
    sections = _resolve_sections(user_literature_block)
    repo_slug = repo_url.rstrip("/").split("/")[-1]
    user_title = paper_title.strip()
    user_abstract = abstract_text.strip()
    user_keywords = keywords_text.strip()
    combined_title = user_title if user_title else repo_slug
    combined_abstract = user_abstract
    combined_keywords = user_keywords
    model_name = _resolve_chat_model_name()
    out: dict[str, Any] = {
        "sections": [],
        "combined_markdown": "",
        "error": None,
        "failed_step": None,
        "failed_section_index": None,
        "paper_title": combined_title,
        "abstract_text": combined_abstract,
        "keywords_text": combined_keywords,
        "norag_mean_faithfulness": 0.0,
        "norag_generation_time_s": 0.0,
        "norag_selected_files": len(selected_paths),
        "norag_context_docs": 0,
        "norag_context_chars": 0,
        "mode": "norag",
    }

    section_blocks: list[dict[str, Any]] = []
    step = "init"
    llm = _build_gemini_llm(model_name)
    safe_invoke = _make_retry_invoke(llm)
    writer = AcademicWriter(llm_invoke_func=safe_invoke)

    try:
        step = "context_build"
        context_docs = build_norag_context_documents(
            selected_paths,
            repo_root=repo_root_path,
            repo_url=repo_url,
            commit_hash=commit_hash,
            max_total_chars=max_total_chars,
        )
        if not context_docs:
            raise RuntimeError("No-RAG context bos kaldi; secilen dosyalardan kullanilabilir belge cikmadi.")
        out["norag_context_docs"] = len(context_docs)
        out["norag_context_chars"] = sum(len(doc.page_content) for doc in context_docs)

        start_time = time.perf_counter()
        for idx, (section_title, section_goal) in enumerate(sections):
            LOGGER.info("No-RAG mode section=%s", section_title)

            step = f"writer[{idx}]"
            writer_out = writer.generate_section(
                section_title=section_title,
                section_goal=section_goal,
                parent_documents=list(context_docs),
                max_parents=20,
                repository_url=repo_url,
                operator_addendum=writer_extra_rules,
                user_literature_block=user_literature_block,
            )
            meta = dict(writer_out.get("metadata") or {})
            meta.pop("full_response", None)

            block = {
                "section_index": idx,
                "section_title": section_title,
                "section_goal": section_goal,
                "retrieval_status": "norag",
                "parents_retrieved": len(context_docs),
                "writer_text": str(writer_out.get("text", "")),
                "writer_metadata": meta,
                "faithfulness": None,
            }

            step = f"faithfulness_judge[{idx}]"
            try:
                block["faithfulness"] = _judge_writer_output_like_pipeline(
                    section_title=section_title,
                    writer_out=writer_out,
                    parent_documents=context_docs,
                    llm_invoke=safe_invoke,
                    user_literature_block=user_literature_block,
                )
                LOGGER.info(
                    "No-RAG judge [%s]: score=%.3f label=%s claims=%d",
                    section_title,
                    block["faithfulness"]["score"],
                    block["faithfulness"]["label"],
                    block["faithfulness"]["claim_count"],
                )
            except Exception as judge_exc:  # noqa: BLE001
                LOGGER.warning(
                    "No-RAG judge failed (soft, section='%s'): %s",
                    section_title,
                    judge_exc,
                )
                block["faithfulness"] = {
                    "score": 0.0,
                    "label": "low",
                    "claim_count": 0,
                    "claims": [],
                    "raw_llm_response": "",
                    "judge_note": f"judge_exception:{type(judge_exc).__name__}",
                }

            step = f"post_judge[{idx}]"
            section_blocks.append(block)
            time.sleep(3)

        out["sections"] = section_blocks
        out["norag_generation_time_s"] = time.perf_counter() - start_time
        out["norag_mean_faithfulness"] = _mean(
            [float((b.get("faithfulness") or {}).get("score") or 0.0) for b in section_blocks]
        )

        step = "metadata"
        md = None
        if (not user_title or not user_abstract) and section_blocks:
            try:
                combined_body_for_meta = "\n\n".join(
                    str(b.get("writer_text") or "") for b in section_blocks if b.get("writer_text")
                ).strip()
                if combined_body_for_meta:
                    md = MetadataWriter(llm_invoke_func=safe_invoke).generate(
                        combined_body=combined_body_for_meta,
                        repo_url=repo_url,
                        rag_documents=context_docs,
                    )
                    if not user_title and md.get("title"):
                        combined_title = str(md["title"])
                    if not user_abstract and md.get("abstract"):
                        combined_abstract = str(md["abstract"])
            except Exception as meta_exc:  # noqa: BLE001
                LOGGER.warning("MetadataWriter atlandi (yumusak hata): %s", meta_exc)

        if not user_keywords:
            if md and (md.get("keywords") or "").strip():
                combined_keywords = str(md["keywords"]).strip()
            elif combined_abstract:
                combined_keywords = extract_keywords_from_abstract(combined_abstract)

        out["paper_title"] = combined_title
        out["abstract_text"] = combined_abstract
        out["keywords_text"] = combined_keywords

        step = "combine"
        out["combined_markdown"] = combine_paper_markdown(
            repo_url=repo_url,
            commit_hash=commit_hash,
            section_results=section_blocks,
            paper_title=combined_title,
            abstract_text=combined_abstract,
            keywords_text=combined_keywords,
            user_literature_block=user_literature_block,
        )

    except Exception as exc:  # noqa: BLE001
        LOGGER.error("No-RAG full paper hata (adim=%s): %s", step, exc)
        out["error"] = str(exc)
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
        match = re.match(r"^(writer|faithfulness_judge|post_judge)\[(\d+)\]$", str(step))
        if match:
            out["failed_section_index"] = int(match.group(2))

    _save_results_json(out, DEFAULT_OUTPUT_PATH)
    LOGGER.info("No-RAG results saved: %s", DEFAULT_OUTPUT_PATH)
    return out


def print_summary_table(results: dict[str, Any]) -> None:
    """No-RAG tam makale kosusunun bolum bazli faithfulness ozetini stdout'a basar."""
    print("| Section | Faithfulness | Label |")
    print("|---------|--------------|-------|")
    for row in results.get("sections") or []:
        faith = row.get("faithfulness") or {}
        score = float(faith.get("score") or 0.0)
        label = str(faith.get("label") or "")
        print(
            f"| {row.get('section_title', '')} | {score:.2f} | {label} |"
        )

    mean_norag = float(results.get("norag_mean_faithfulness") or 0.0)
    print(f"| MEAN | **{mean_norag:.2f}** |  |")
    print()
    print(f"No-RAG generation time (s): {float(results.get('norag_generation_time_s') or 0.0):.2f}")


def _parse_args() -> argparse.Namespace:
    """CLI argumanlarini tanimlar."""
    parser = argparse.ArgumentParser(
        description="Run the full paper pipeline in standalone No-RAG mode."
    )
    parser.add_argument("--repo-url", required=True, help="HTTPS GitHub repo URL")
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Path to the already cloned repository root",
    )
    parser.add_argument(
        "--literature-file",
        default="",
        help="Optional path to a txt file containing USER_LITERATURE block",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        help="Maximum number of files from paths_for_index to use",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="JSON output path (default: data/norag_results.json)",
    )
    return parser.parse_args()


def _load_literature_block(path_str: str) -> str:
    """Opsiyonel user literature txt dosyasini okur; dosya yoksa bos string doner."""
    if not path_str:
        return ""
    path = Path(path_str).resolve()
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _rank_paths_for_indexing(paths: list[Path]) -> list[Path]:
    """
    Streamlit arayuzundeki `_pick_paths_for_indexing` mantigini script tarafinda tekrarlar.

    Boyut eleyici + uzanti onceligi + alfabetik yol sirasi ile deneye verilecek
    aday dosya listesini stabil hale getirir.
    """
    ranked: list[tuple[int, str, Path]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _MAX_INDEX_FILE_BYTES:
            continue
        suffix = path.suffix.lower()
        try:
            prio = _INDEX_SUFFIX_PRIORITY.index(suffix)
        except ValueError:
            prio = len(_INDEX_SUFFIX_PRIORITY)
        ranked.append((prio, path.as_posix().lower(), path))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked]


def _collect_paths_for_experiment(repo_root: Path) -> list[Path]:
    """
    Repo kokundan deneye verilecek tum aday dosya yol listesini toplar.

    Ablasyonun amaci ayni dosya listesiyle iki stratejiyi karsilastirmak oldugu icin,
    once mevcut sanitize/uzanti kurallariyla indexable dosyalari toplar, sonra
    arayuzdekiyle ayni boyut + uzanti onceligi siralamasini uygular.
    """
    repo_root_path = Path(repo_root).resolve()
    return _rank_paths_for_indexing(iter_indexable_files(repo_root_path))


def _resolve_repo_commit_hash(repo_root: Path) -> str:
    """
    Deneyin hangi kod anlik durumuna ait oldugunu kaydetmek icin repo commit hash'ini bulur.

    Git repo degilse kontrollu fallback ile `local-worktree` doner.
    """
    try:
        return Repo(str(Path(repo_root).resolve())).head.commit.hexsha
    except Exception:  # noqa: BLE001
        return "local-worktree"


if __name__ == "__main__":
    _configure_logging()
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY", "").strip():
        raise EnvironmentError("GOOGLE_API_KEY bulunamadi. Lutfen .env dosyasini kontrol edin.")

    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    literature_block = _load_literature_block(args.literature_file)
    paths = _collect_paths_for_experiment(repo_root)
    commit_hash = _resolve_repo_commit_hash(repo_root)

    results = run_norag_full_paper(
        repo_url=args.repo_url,
        commit_hash=commit_hash,
        repo_root=repo_root,
        paths_for_index=paths,
        user_literature_block=literature_block,
        max_index_files=args.max_files,
    )
    print_summary_table(results)

    output_path = Path(args.output).resolve()
    _save_results_json(results, output_path)
    LOGGER.info("No-RAG results written to %s", output_path)
