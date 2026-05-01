"""
Streamlit arayüzü: GitHub URL girilir, repo klonlanır ve süzülür; dosyalar listeden
seçilerek içerik önizlemesi gösterilir.

RAG (deneysel): Gemini embedding ile parent-child indeksleme, Planner sorgulari ve
multi-query retrieval bu ekrandan tetiklenebilir.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from agents.writer import AcademicWriter
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()  # .env'i Streamlit widget'larından ve LLM istemcilerinden önce yükle
from git.exc import GitCommandError
from langchain_google_genai import ChatGoogleGenerativeAI

from github_handler import clone_and_prepare
from agents.literature_filter import (
    extract_pdf_text_to_string,
    filter_literature_relevance,
    format_approved_for_writer,
    split_pasted_literature,
)
from agents.ieee_json_schema import normalize_ieee_paper_content, parse_ieee_paper_json
from agents.ieee_json_writer import generate_ieee_paper_json_raw
from export.ieee_document_from_json import build_ieee_document_bytes
from export.ieee_template_export import (
    markdown_to_ieee_template_docx_bytes,
    resolve_default_ieee_template,
)
from export.word_export import markdown_to_docx_bytes
from orchestration.paper_blueprint import DEFAULT_PAPER_SECTIONS
from orchestration.section_pipeline import run_paper_pipeline, run_section_pipeline
from retriever import (
    build_rag_stack_for_repo,
    configure_logging,
    generate_planner_queries,
    index_repository_files,
    retrieve_parent_contexts_multi_query,
)

# Önizlemede çok büyük dosyaları kesmek için üst sınır (karakter).
_MAX_PREVIEW_CHARS = 80_000
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
_DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
_GEMINI_FALLBACK_MODELS = (
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
)


def _read_google_api_key() -> str:
    """
    `.env` dosyasından GOOGLE_API_KEY değerini alır.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY bulunamadi. .env dosyasina ekleyin.")
    return api_key


def _build_gemini_llm(model_name: str = _DEFAULT_GEMINI_MODEL) -> ChatGoogleGenerativeAI:
    """
    Gemini istemcisini `.env` içindeki GOOGLE_API_KEY ile kurar.
    """
    api_key = _read_google_api_key()
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
    )


def _list_accessible_gemini_models() -> list[str]:
    """
    Key ile erişilebilen model isimlerini Google SDK üzerinden listeler.
    """
    client = genai.Client(api_key=_read_google_api_key())
    names: list[str] = []
    for model in client.models.list():
        name = getattr(model, "name", "")
        if isinstance(name, str) and name:
            names.append(name.replace("models/", ""))
    return names


def _resolve_working_gemini_model() -> tuple[str, str]:
    """
    Çalışabilir Gemini modelini test ederek seçer.

    Önce fallback sırasını dener; her model için kısa bir ping atar ve ilk başarılı
    modeli döndürür. Hiçbiri çalışmıyorsa son hata mesajını da taşır.
    """
    key = _read_google_api_key()
    available = set(_list_accessible_gemini_models())
    errors: list[str] = []

    for candidate in _GEMINI_FALLBACK_MODELS:
        if candidate not in available:
            errors.append(f"{candidate}: listelenmedi")
            continue
        try:
            llm = ChatGoogleGenerativeAI(model=candidate, google_api_key=key, temperature=0.2)
            response = llm.invoke("Reply with exactly: GEMINI_OK")
            text = str(response.content).strip()
            if "GEMINI_OK" in text:
                return candidate, text
            errors.append(f"{candidate}: beklenen cevap alinmadi -> {text[:80]}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("Gemini model secimi basarisiz -> " + " | ".join(errors))


def _gemini_connection_test() -> tuple[bool, str]:
    """
    Basit ping ile Gemini bağlantısını test eder.
    """
    try:
        model, response_text = _resolve_working_gemini_model()
        return True, f"[model={model}] {response_text}"
    except Exception as exc:  # noqa: BLE001
        return (False, str(exc))


def _gemini_retry_hint(exc: BaseException) -> str:
    """
    Gemini'den gelen yaygin gecici hatalar icin kullaniciya Turkce yol gosterir.
    """
    text = str(exc).lower()
    if "503" in text or "unavailable" in text:
        return "Gemini su an yogun (503). Bir kac saniye sonra ayni islemi tekrar deneyin."
    if "429" in text or "resource_exhausted" in text or "quota" in text:
        return (
            "Dakika basina istek kotasi asildi (429). Free tier limiti var; "
            "1 dakika bekleyip tekrar deneyin veya ard arda cok tiklamayin."
        )
    if "403" in text or "permission" in text:
        return "API anahtari veya proje erisiminde sorun var (403). AI Studio / proje ayarlarini kontrol edin."
    if "404" in text or "not found" in text:
        return "Model veya endpoint bulunamadi (404). Model adini ve API surumunu kontrol edin."
    return "Beklenmeyen Gemini hatasi. Mesaji kopyalayip ekibe iletin."


def _invoke_gemini_chat_with_retry(llm, prompt: str, *, max_attempts: int = 5) -> str:
    """
    Writer gibi tek seferlik chat cagrilarinda 503/429 icin retry uygular.
    """
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


def _get_cached_gemini_chat_model_name() -> str:
    """
    Streamlit yeniden calismalarinda model listesini tekrar tekrar denememek icin
    secilen chat model adini session_state uzerinde cache'ler.
    """
    if "gemini_chat_model_name" not in st.session_state:
        model_name, _ = _resolve_working_gemini_model()
        st.session_state["gemini_chat_model_name"] = model_name
    return str(st.session_state["gemini_chat_model_name"])


def _pick_paths_for_indexing(paths: list[Path], max_files: int) -> list[Path]:
    """
    Indekslemek icin dosya listesini boyut ve uzanti onceligiyle kirpar.

    Cok buyuk dosyalar embedding maliyetini patlatmasin diye atlanir.
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
    return [item[2] for item in ranked[: max(1, max_files)]]


def _sync_rag_session_if_repo_changed(repo_url: str, commit_hash: str) -> None:
    """
    Farkli repo veya commit cekildiginde eski retriever oturumunu temizler.
    """
    key = f"{repo_url}|{commit_hash}"
    if st.session_state.get("rag_session_key") != key:
        for k in (
            "rag_session_key",
            "rag_retriever",
            "rag_index_totals",
            "planner_queries",
            "retrieved_parent_docs",
            "writer_draft_en",
        ):
            st.session_state.pop(k, None)
        st.session_state["rag_session_key"] = key


def _render_rag_indexing_section(
    *,
    paths: list[Path],
    root: Path,
    repo_url: str,
    commit_hash: str,
) -> None:
    """
    Parent-child + Gemini embedding ile Chroma indekslemesini baslatan UI blogu.
    """
    _sync_rag_session_if_repo_changed(repo_url, commit_hash)
    with st.expander("RAG: indeksleme (Gemini embedding + Parent-Child)", expanded=False):
        st.caption(
            "Once repoyu cekin. Parent-child her dosyayi cok parcaya boler; Gemini free tier "
            "dakikada ~100 embedding istegi sinirlar. Indeksleme yavas olabilir (throttle aktif). "
            "Ilk denemede max dosya sayisini dusuk tutun (or. 5-10)."
        )
        max_files = st.number_input(
            "Indekslenecek max dosya sayisi",
            min_value=1,
            max_value=80,
            value=8,
            step=1,
        )
        if st.button("Indekslemeyi baslat", type="secondary", use_container_width=True):
            picked = _pick_paths_for_indexing(paths, int(max_files))
            if not picked:
                st.error("Indeks icin uygun dosya bulunamadi.")
                return
            with st.spinner("Retriever kuruluyor ve dosyalar indeksleniyor..."):
                try:
                    configure_logging()
                    retriever, _store, _vs = build_rag_stack_for_repo(repo_url, commit_hash)
                    totals = index_repository_files(
                        retriever,
                        picked,
                        repo_root=root,
                        repo_url=repo_url,
                        commit_hash=commit_hash,
                    )
                    st.session_state["rag_retriever"] = retriever
                    st.session_state["rag_index_totals"] = totals
                    st.session_state["rag_indexed_paths_count"] = len(picked)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Indeksleme hatasi: {exc}")
                    st.warning(_gemini_retry_hint(exc))
                    return
            st.success(
                f"Indeksleme tamam. {totals.get('files', 0)} dosya, "
                f"{totals.get('parents', 0)} parent, {totals.get('children', 0)} child."
            )

        if st.session_state.get("rag_retriever") is not None:
            st.info("Retriever hazir: asagidan Planner ve multi-query retrieval calistirabilirsiniz.")


def _inject_compact_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stSidebarUserContent"] .block-container { padding-top: 1rem; }
        .ctp-muted { color: #666; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _detect_language(suffix: str) -> str | None:
    """Dosya uzantısına göre st.code language parametresi (Streamlit / Pygments)."""
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".java": "java",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sql": "sql",
        ".md": "markdown",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".sh": "bash",
    }
    return mapping.get(suffix.lower())


def _friendly_git_error(exc: GitCommandError) -> str:
    err = str(exc).lower()
    if "could not read username" in err or "authentication" in err:
        return (
            "Depo özel (private) görünüyor veya kimlik doğrulama istiyor. "
            "Bu arayüz yalnızca herkese açık HTTPS adresleri için tasarlandı."
        )
    if "not found" in err or "repository not found" in err:
        return "Depo bulunamadı. URL’yi kontrol edin (ör. `https://github.com/kullanici/proje.git`)."
    if "network" in err or "connection" in err:
        return "Ağ hatası: İnternet bağlantınızı veya GitHub erişimini kontrol edin."
    return f"Git işlemi başarısız: {exc}"


def _render_sidebar() -> str:
    """Yan panel: proje özeti, klon klasörü, gelecek adımlar. Dönüş: hedef klasör metni."""
    st.sidebar.markdown("### Code-to-Paper")
    st.sidebar.markdown(
        '<p class="ctp-muted">Public GitHub kodunu çekip süzerek kaynak önizlemesi. '
        "Sonraki aşamada Vector RAG ile IEEE/ACM/Springer tarzı dokümantasyon üretilecek.</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    target_dir = st.sidebar.text_input(
        "Klon klasörü",
        value="data/source",
        help="Repo bu klasöre klonlanır; doluysa önce temizlenir.",
    )
    st.sidebar.divider()
    st.sidebar.markdown("##### Cloud LLM ayari")
    st.sidebar.caption(
        "Gemini otomatik model secimi: "
        + " -> ".join(_GEMINI_FALLBACK_MODELS)
    )
    if st.sidebar.button("Gemini baglantisini test et", use_container_width=True):
        ok, msg = _gemini_connection_test()
        if ok:
            st.sidebar.success("Gemini baglantisi basarili.")
        else:
            st.sidebar.error(f"Gemini baglanti hatasi: {msg}")
            st.sidebar.warning(_gemini_retry_hint(RuntimeError(msg)))
    st.sidebar.caption("Orchestration: LangChain · Vektor: ChromaDB · LLM: Gemini")
    return target_dir


def _build_docx_from_markdown(md: str) -> tuple[bytes, str]:
    """
    IEEE Word sablonunun uzerine yazar; sablon yoksa veya hata olursa duz .docx uretir.

    Sablon yolu: `ieee_template_path_input` oturumu, sonra IEEE_DOCX_TEMPLATE, sonra
    docs/templates/ConferenceTemplateIEEE.docx.
    """
    custom = str(st.session_state.get("ieee_template_path_input") or "").strip()
    path: Path | None = Path(custom) if custom else None
    if path is not None and not path.is_file():
        path = None
    path = path or resolve_default_ieee_template()
    if path is None:
        return markdown_to_docx_bytes(md), "IEEE sablonu bulunamadi; duz Word kullanildi."
    try:
        return markdown_to_ieee_template_docx_bytes(path, md), f"IEEE sablonu kullanildi: {path}"
    except Exception as exc:  # noqa: BLE001
        return markdown_to_docx_bytes(md), f"Sablon acilamadi ({exc}); duz Word kullanildi."


# ---------------------------------------------------------------------------
# Faithfulness badge + claim breakdown helpers
# ---------------------------------------------------------------------------

_FAITH_BADGE_CSS: dict[str, tuple[str, str]] = {
    # label -> (background colour, text colour)
    "high":   ("#1e7e34", "#ffffff"),
    "medium": ("#856404", "#ffffff"),
    "low":    ("#842029", "#ffffff"),
    "unavailable": ("#555555", "#ffffff"),
}

_VERDICT_EMOJI = {
    "supported":   "✓",
    "partial":     "~",
    "unsupported": "✗",
}


def _render_faithfulness_badge(faithfulness: dict | None, *, section_title: str = "") -> None:
    """Renders a coloured inline badge + per-claim breakdown expander."""
    if faithfulness is None:
        bg, fg = _FAITH_BADGE_CSS["unavailable"]
        st.markdown(
            f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:4px;'
            f'font-size:0.85rem;font-weight:600">⚫ Faithfulness: judge_unavailable</span>',
            unsafe_allow_html=True,
        )
        return

    score = faithfulness.get("score", 0.0)
    label = faithfulness.get("label", "low")
    count = faithfulness.get("claim_count", 0)
    emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(label, "⚫")
    bg, fg = _FAITH_BADGE_CSS.get(label, _FAITH_BADGE_CSS["unavailable"])
    st.markdown(
        f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:4px;'
        f'font-size:0.85rem;font-weight:600">{emoji} Faithfulness: {score:.2f} / {label} ({count} claims)</span>',
        unsafe_allow_html=True,
    )

    claims = faithfulness.get("claims") or []
    if claims:
        with st.expander("Claim breakdown", expanded=False):
            rows: list[str] = []
            rows.append("| Claim ID | Summary | Verdict | Evidence Quote |")
            rows.append("|----------|---------|---------|----------------|")
            for c in claims:
                verdict_icon = _VERDICT_EMOJI.get(c.get("verdict", ""), "?")
                cid = c.get("id", "")
                summary = (c.get("summary") or "")[:80]
                verdict = f"{verdict_icon} {c.get('verdict', '')}"
                quote = (c.get("evidence_quote") or "")[:120].replace("|", "\\|")
                rows.append(f"| {cid} | {summary} | {verdict} | {quote} |")
            st.markdown("\n".join(rows))


def _compute_paper_faithfulness(section_blocks: list[dict]) -> tuple[float | None, str | None, int]:
    """
    Computes the paper-wide faithfulness as a weighted mean (weighted by claim_count).

    Returns (score, label, total_claims) or (None, None, 0) if no scored sections.
    """
    total_weight = 0
    weighted_sum = 0.0
    for block in section_blocks:
        f = block.get("faithfulness")
        if not f:
            continue
        count = f.get("claim_count", 0)
        if count == 0:
            continue
        weighted_sum += f["score"] * count
        total_weight += count
    if total_weight == 0:
        return None, None, 0
    score = round(weighted_sum / total_weight, 3)
    label = "high" if score >= 0.8 else "medium" if score >= 0.6 else "low"
    return score, label, total_weight


def _render_agent_preview_panel() -> None:
    """
    Planner + (istege bagli) multi-query retrieval + Ingilizce writer taslagini test eder.

    Indeksleme yapilmadan retrieval adimi calismaz.
    """
    st.subheader("Is 4-5 onizleme: Planner, retrieval, writer (Gemini)")
    section_title = st.text_input("Section basligi", value="Security Architecture")
    section_goal = st.text_area(
        "Section amaci",
        value="Explain authentication, authorization and token validation flow from code evidence.",
    )

    with st.expander("Literatur ve ek yazim talimatlari (istege bagli)", expanded=False):
        st.caption(
            "Literatur: metin yapistirin ve/veya PDF yukleyin; 'Surzgec' mevcut bolum basligi/amacina gore "
            "alakasiz parcalari eler. Writer sadece onaylanan metni [2],[3],... olarak kullanir."
        )
        st.text_area(
            "Ek yazim talimatlari (OPERATOR_ADDENDUM)",
            height=72,
            key="writer_extra_rules_input",
            placeholder="Ornek: passive voice; avoid first person; max 800 words for PART 1.",
        )
        st.text_area(
            "Literatur metni (parcalari ayirmak icin satira sadece --- yazin; istege bagli ilk satir: Title: ...)",
            height=140,
            key="literature_paste_bundle",
            placeholder="Title: Related survey excerpt\n...\n---\nTitle: Second source\n...",
        )
        lit_pdfs = st.file_uploader(
            "PDF literatur (coklu)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("Literaturu LLM ile surzgecten gecir", type="secondary"):
            lit_items: list[tuple[str, str]] = list(
                split_pasted_literature(str(st.session_state.get("literature_paste_bundle") or ""))
            )
            if lit_pdfs:
                for up in lit_pdfs:
                    try:
                        try:
                            raw_b = up.getvalue()
                        except AttributeError:
                            raw_b = up.read()
                        lit_items.append((up.name, extract_pdf_text_to_string(raw_b)))
                    except Exception as pdf_exc:  # noqa: BLE001
                        st.error(f"PDF okunamadi ({up.name}): {pdf_exc}")
            if not lit_items:
                st.warning("Surzgec icin en az bir metin parcasi veya PDF ekleyin.")
            else:
                try:
                    model_name = _get_cached_gemini_chat_model_name()
                    llm_lit = _build_gemini_llm(model_name)

                    def _lit_invoke(prompt_text: str) -> str:
                        return _invoke_gemini_chat_with_retry(llm_lit, prompt_text)

                    approved, excluded = filter_literature_relevance(
                        _lit_invoke,
                        section_title=section_title,
                        section_goal=section_goal,
                        repository_hint=str(st.session_state.get("last_url") or ""),
                        items=lit_items,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Literatur surzgec hatasi: {exc}")
                    st.warning(_gemini_retry_hint(exc))
                else:
                    st.session_state["literature_writer_block"] = format_approved_for_writer(approved)
                    st.session_state["literature_filter_excluded"] = excluded
                    st.success(f"Surzgec tamam: {len(approved)} parca Writer'a verilecek, {len(excluded)} elendi.")
                    if excluded:
                        with st.expander("Elenen parcalar"):
                            for row in excluded:
                                st.caption(f"#{row.get('index')}: {row.get('reason')}")

    st.caption(
        "Bu adim embedding + LLM cagrisi yaptigi icin maliyet ve sure olusur. "
        "Free tier kullaniminda islemler daha yavas ilerleyebilir."
    )

    run_mode = st.radio(
        "Calisma modu",
        options=("Adim adim", "Tek akis", "Tam makale (coklu bolum)"),
        horizontal=True,
        help="Adim adim: adim adim. Tek akis: tek bolum. Tam makale: bir indeks, ardışık 3 bolum (daha fazla LLM cagrisi).",
    )
    if run_mode == "Tam makale (coklu bolum)":
        st.caption(
            "Tam makale: "
            + " · ".join(t for t, _ in DEFAULT_PAPER_SECTIONS)
            + " — her bolum icin planner + retrieval + writer calisir."
        )

    st.caption(
        "Benzerlik: Sonuc cok azsa esigi dusurun (or. 0.15). Cok gurultu varsa yukseltin (or. 0.45)."
    )
    similarity_threshold = st.slider(
        "Benzerlik esigi (relevance score)",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Dusuk skorlu child hitleri eler. Gerekirse dusurun.",
    )
    top_k_per_query = st.number_input(
        "Sorgu basina child aday sayisi (top-k)",
        min_value=2,
        max_value=20,
        value=6,
        step=1,
        help="Her planner sorgusu icin Chroma'dan cekilecek aday sayisi.",
    )
    max_index_files = st.number_input(
        "Tek akis icin indekslenecek max dosya sayisi",
        min_value=1,
        max_value=80,
        value=20,
        step=1,
        help="Indeksleme tarafindaki secim mantigi (_pick_paths_for_indexing) ile dosyalar kirpilir.",
    )
    max_planner_queries = st.number_input(
        "Maks planner sorgusu",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
    )

    _extra_rules = str(st.session_state.get("writer_extra_rules_input") or "").strip()
    _lit_block = str(st.session_state.get("literature_writer_block") or "").strip()

    if run_mode == "Tam makale (coklu bolum)":
        has_repo = all(
            st.session_state.get(k) is not None for k in ("paths", "root", "commit", "last_url")
        )
        if st.button(
            "Tam makale: indeks + 3 bolum (planner / retrieval / writer x3)",
            type="primary",
            use_container_width=True,
            disabled=not has_repo,
        ):
            if not has_repo:
                st.error("Once repoyu cekin.")
                return
            picked = _pick_paths_for_indexing(
                list(st.session_state.get("paths") or []),
                int(max_index_files),
            )
            if not picked:
                st.error("Indeks icin uygun dosya bulunamadi.")
                return
            status = st.empty()
            try:
                os.environ["GEMINI_CHAT_MODEL"] = _get_cached_gemini_chat_model_name()
                status.info("Tam makale pipeline basladi (uzun surebilir)...")
                with st.spinner("Indeks + 3 bolum yaziliyor..."):
                    paper_result = run_paper_pipeline(
                        repo_url=str(st.session_state["last_url"]),
                        commit_hash=str(st.session_state["commit"]),
                        repo_root=Path(st.session_state["root"]),
                        paths_for_index=picked,
                        sections=None,
                        max_index_files=int(max_index_files),
                        similarity_threshold=float(similarity_threshold),
                        top_k_per_query=int(top_k_per_query),
                        max_planner_queries=int(max_planner_queries),
                        writer_extra_rules=_extra_rules,
                        user_literature_block=_lit_block,
                        existing_retriever=st.session_state.get("rag_retriever"),
                    )
            except Exception as exc:  # noqa: BLE001
                if "GOOGLE_API_KEY" in str(exc):
                    st.error("GOOGLE_API_KEY bulunamadi. Proje kokundeki .env dosyasina ekleyin.")
                    st.info("Ornek: GOOGLE_API_KEY=AIza... (tek satir)")
                    return
                st.error(f"Tam makale hatasi: {exc}")
                st.warning(_gemini_retry_hint(exc))
                return
            finally:
                status.empty()

            if paper_result.get("error"):
                st.error(
                    f"Tam makale basarisiz (adim={paper_result.get('failed_step') or 'unknown'}): "
                    f"{paper_result.get('error')}"
                )
                if paper_result.get("failed_section_index") is not None:
                    st.caption(
                        f"Hata bolum indeksi: {paper_result.get('failed_section_index')} "
                        "(kismi cikti asagida olabilir)."
                    )
                st.warning(_gemini_retry_hint(RuntimeError(str(paper_result.get("error")))))
            else:
                totals = paper_result.get("rag_totals") or {}
                n_sec = len(paper_result.get("sections") or [])
                # Paper-wide faithfulness metric
                _pw_score, _pw_label, _pw_claims = _compute_paper_faithfulness(
                    list(paper_result.get("sections") or [])
                )
                _faith_col1, _faith_col2, _faith_col3, _faith_col4 = st.columns(4)
                _faith_col1.metric("Bölüm sayısı", n_sec)
                _faith_col2.metric("İndekslenen dosya", totals.get('files', 0))
                _faith_col3.metric("Toplam iddia (claim)", _pw_claims)
                if _pw_score is not None:
                    _faith_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(_pw_label, "⚫")
                    _faith_col4.metric(
                        "Makale geneli güvenilirlik",
                        f"{_faith_emoji} {_pw_score:.2f} ({_pw_label})",
                    )
                else:
                    _faith_col4.metric("Makale geneli güvenilirlik", "— (judge_unavailable)")
                st.success(
                    f"Tam makale tamamlandi (bolum sayisi={n_sec}). "
                    f"{totals.get('files', 0)} dosya indekslendi."
                )
            for sb in paper_result.get("sections") or []:
                rs = sb.get("retrieval_status")
                if rs in ("empty_after_retry",):
                    st.warning(
                        f"Bolum '{sb.get('section_title')}': retrieval sonuc bos olabilir "
                        f"({rs})."
                    )
            # --- Per-section faithfulness display ---
            # Streamlit nested expander DESTEKLEMEZ; outer expander yerine markdown
            # baslik + horizontal rule kullaniyoruz. Boylece icerideki "Claim breakdown"
            # (badge icindeki) ve "TRACEABILITY (ic kontrol)" expander'lari top-level kalir.
            _sections_for_display = list(paper_result.get("sections") or [])
            if _sections_for_display:
                st.markdown("**Bölüm önizlemeleri (güvenilirlik skorları)**")
                for _sb in _sections_for_display:
                    _sb_title = str(_sb.get("section_title") or "Section")
                    st.markdown(f"---\n#### 📄 {_sb_title}")
                    _render_faithfulness_badge(_sb.get("faithfulness"), section_title=_sb_title)
                    _sb_trace = str((_sb.get("writer_metadata") or {}).get("traceability") or "").strip()
                    if _sb_trace:
                        with st.expander("TRACEABILITY (iç kontrol)", expanded=False):
                            st.markdown(_sb_trace)
            combined = str(paper_result.get("combined_markdown") or "").strip()
            if combined:
                st.session_state["full_paper_combined_md"] = combined
            st.session_state["paper_pipeline_sections"] = list(
                paper_result.get("sections") or []
            )
            if paper_result.get("sections"):
                st.session_state["writer_draft_en"] = str(
                    paper_result["sections"][-1].get("writer_text") or ""
                )
                st.session_state["writer_metadata"] = dict(
                    paper_result["sections"][-1].get("writer_metadata") or {}
                )

    elif run_mode == "Tek akis":
        has_repo = all(
            st.session_state.get(k) is not None for k in ("paths", "root", "commit", "last_url")
        )
        if st.button(
            "Tek akis calistir (indeks -> planner -> retrieval -> writer)",
            type="primary",
            use_container_width=True,
            disabled=not has_repo,
        ):
            if not has_repo:
                st.error("Once repoyu cekin.")
                return
            picked = _pick_paths_for_indexing(
                list(st.session_state.get("paths") or []),
                int(max_index_files),
            )
            if not picked:
                st.error("Indeks icin uygun dosya bulunamadi.")
                return

            status = st.empty()
            try:
                os.environ["GEMINI_CHAT_MODEL"] = _get_cached_gemini_chat_model_name()
                status.info("Indeksleniyor...")
                status.info("Planner sorgulari uretiliyor...")
                status.info("Kaynaklar araniyor...")
                with st.spinner("Bolum yaziliyor..."):
                    result = run_section_pipeline(
                        repo_url=str(st.session_state["last_url"]),
                        commit_hash=str(st.session_state["commit"]),
                        repo_root=Path(st.session_state["root"]),
                        paths_for_index=picked,
                        section_title=section_title,
                        section_goal=section_goal,
                        max_index_files=int(max_index_files),
                        similarity_threshold=float(similarity_threshold),
                        top_k_per_query=int(top_k_per_query),
                        max_planner_queries=int(max_planner_queries),
                        writer_extra_rules=_extra_rules,
                        user_literature_block=_lit_block,
                        existing_retriever=st.session_state.get("rag_retriever"),
                    )
            except Exception as exc:  # noqa: BLE001
                if "GOOGLE_API_KEY" in str(exc):
                    st.error("GOOGLE_API_KEY bulunamadi. Proje kokundeki .env dosyasina ekleyin.")
                    st.info("Ornek: GOOGLE_API_KEY=AIza... (tek satir)")
                    return
                st.error(f"Tek akis hatasi: {exc}")
                st.warning(_gemini_retry_hint(exc))
                return
            finally:
                status.empty()

            if result.get("error"):
                st.error(
                    f"Tek akis basarisiz (adim={result.get('failed_step') or 'unknown'}): "
                    f"{result.get('error')}"
                )
                st.warning(_gemini_retry_hint(RuntimeError(str(result.get("error")))))
                return

            st.session_state["planner_queries"] = list(result.get("planner_queries") or [])
            st.session_state["retrieved_parent_docs"] = list(result.get("retrieved_parent_docs") or [])
            st.session_state["writer_draft_en"] = str(result.get("writer_text") or "")

            totals = result.get("rag_totals") or {}
            st.success(
                "Tek akis tamamlandi. "
                f"{totals.get('files', 0)} dosya, {totals.get('parents', 0)} parent, "
                f"{totals.get('children', 0)} child."
            )
            if result.get("retrieval_status") == "empty_after_retry":
                st.warning(
                    "Esik dusuruldu, yine sonuc yok; Planner sorgularini veya indeks dosya "
                    "sayisini gozden gecirin."
                )
    else:
        if st.button("Planner sorgularini uret", use_container_width=True):
            try:
                model_name = _get_cached_gemini_chat_model_name()
                llm = _build_gemini_llm(model_name)
                queries = generate_planner_queries(
                    llm,
                    section_title=section_title,
                    section_goal=section_goal,
                    max_queries=int(max_planner_queries),
                )
                st.session_state["planner_queries"] = queries
                st.success(f"{len(queries)} planner sorgusu uretildi. (model={model_name})")
            except Exception as e:  # noqa: BLE001
                st.error(f"Planner calisamadi: {e}")
                st.warning(_gemini_retry_hint(e))

    if "planner_queries" in st.session_state:
        st.markdown("**Planner ciktilari**")
        for q in st.session_state["planner_queries"]:
            st.code(q, language="text")

    retriever = st.session_state.get("rag_retriever")
    if retriever is None and run_mode == "Adim adim":
        st.warning("Multi-query retrieval icin once yukaridaki RAG indekslemesini calistirin.")
    elif retriever is not None and run_mode == "Adim adim":
        if st.button("Multi-query retrieval calistir", use_container_width=True):
            queries = st.session_state.get("planner_queries")
            if not queries:
                st.error("Once Planner sorgularini uretin.")
                return
            with st.spinner("Chroma uzerinde multi-query arama yapiliyor..."):
                try:
                    docs = retrieve_parent_contexts_multi_query(
                        retriever,
                        planner_queries=list(queries),
                        top_k_per_query=int(top_k_per_query),
                        similarity_threshold=float(similarity_threshold),
                    )
                    st.session_state["retrieved_parent_docs"] = docs
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Retrieval hatasi: {exc}")
                    return
            if not docs:
                st.warning(
                    "Hic parent baglam secilmedi. Esigi dusurun, top-k artirin veya "
                    "Planner sorgularini bolumle daha uyumlu hale getirip tekrar deneyin."
                )
            else:
                st.success(f"{len(docs)} parent baglam secildi (tekrar eden dosya/satir birlestirildi).")

    retrieved = st.session_state.get("retrieved_parent_docs")
    if retrieved:
        st.markdown("**Secilen kaynaklar (ozet)**")
        for doc in retrieved[:12]:
            fp = doc.metadata.get("file_path", "?")
            sl = doc.metadata.get("start_line", "?")
            el = doc.metadata.get("end_line", "?")
            st.caption(f"{fp}  (satir {sl}-{el})")

    st.markdown("---")
    st.markdown("##### Writer modulu (IEEE + Mermaid)")
    with st.expander("IEEE Word sablonu (.docx ciktisi)", expanded=False):
        st.caption(
            "IEEE dosyasi yalnizca **stil sablonu** olarak kullanilir: icindeki ornek/dummy metin "
            "silinir; yalnizca urettigimiz icerik, sablonun Word stilleriyle (paper title, Heading 1/2, "
            "Body Text) yazilir. Varsayilan: `docs/templates/ConferenceTemplateIEEE.docx`."
        )
        st.text_input(
            "Ozel sablon yolu (bos = varsayilan veya IEEE_DOCX_TEMPLATE ortam degiskeni)",
            key="ieee_template_path_input",
            placeholder=r"Ornek: C:\Users\...\ConferenceTemplateIEEE.docx",
        )

    with st.expander("IEEE tam makale: JSON + Word (sablon stilleri, onerilen)", expanded=False):
        st.caption(
            "Bu akista LLM **yalnizca JSON** uretir; Word tarafi sablon govdesini temizleyip "
            "`paper title`, `Author`, `Abstract`, `Keywords`, `Heading 1`–`Heading 3`, `Heading 5`, "
            "`Body Text`, `references` stillerini kullanir. Once **Multi-query retrieval** calistirin."
        )
        if st.button("IEEE tam makale: JSON uret ve Word hazirla", key="ieee_json_from_rag_btn"):
            docs_rag = st.session_state.get("retrieved_parent_docs")
            if not docs_rag:
                st.error("Once multi-query retrieval ile parent baglam secin.")
            else:
                tpl_path = str(st.session_state.get("ieee_template_path_input") or "").strip()
                tpl = Path(tpl_path) if tpl_path and Path(tpl_path).is_file() else resolve_default_ieee_template()
                if tpl is None:
                    st.error("IEEE sablon dosyasi bulunamadi.")
                else:
                    try:
                        model_name = _get_cached_gemini_chat_model_name()
                        llm_ij = _build_gemini_llm(model_name)

                        def _ij_invoke(p: str) -> str:
                            return _invoke_gemini_chat_with_retry(llm_ij, p)

                        raw_json = generate_ieee_paper_json_raw(
                            _ij_invoke,
                            parent_documents=list(docs_rag),
                            repository_url=str(st.session_state.get("last_url") or ""),
                            operator_addendum=_extra_rules,
                            user_literature_block=_lit_block,
                        )
                        data = parse_ieee_paper_json(raw_json)
                        data = normalize_ieee_paper_content(
                            data,
                            repository_url=str(st.session_state.get("last_url") or ""),
                        )
                        docx_b = build_ieee_document_bytes(tpl, data)
                        st.session_state["ieee_full_paper_docx"] = docx_b
                        st.session_state["ieee_full_paper_json"] = json.dumps(data, ensure_ascii=False, indent=2)
                        st.success("IEEE JSON + Word hazir. Asagidan indirin.")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"IEEE JSON/Word hatasi: {exc}")
                        st.warning(_gemini_retry_hint(exc))

        st.text_area(
            "Veya tam makale JSON'unu buraya yapistir (manuel Word)",
            height=220,
            key="ieee_json_paste_area",
            placeholder='{"title": "...", "authors": [...], ...}',
        )
        if st.button("Yapistirilan JSON'dan Word olustur", key="ieee_json_from_paste_btn"):
            tpl_path = str(st.session_state.get("ieee_template_path_input") or "").strip()
            tpl = Path(tpl_path) if tpl_path and Path(tpl_path).is_file() else resolve_default_ieee_template()
            if tpl is None:
                st.error("IEEE sablon dosyasi bulunamadi.")
            else:
                try:
                    pasted = str(st.session_state.get("ieee_json_paste_area") or "").strip()
                    data = parse_ieee_paper_json(pasted)
                    data = normalize_ieee_paper_content(
                        data,
                        repository_url=str(st.session_state.get("last_url") or ""),
                    )
                    st.session_state["ieee_full_paper_docx"] = build_ieee_document_bytes(tpl, data)
                    st.session_state["ieee_full_paper_json"] = json.dumps(data, ensure_ascii=False, indent=2)
                    st.success("Word hazir (yapistirilan JSON).")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"JSON/Word hatasi: {exc}")

        _ij_docx = st.session_state.get("ieee_full_paper_docx")
        if _ij_docx:
            st.download_button(
                "IEEE tam makaleyi indir (.docx — JSON sablonu)",
                data=_ij_docx,
                file_name="ieee-full-paper.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="download_ieee_json_docx",
            )
            _ij_json = st.session_state.get("ieee_full_paper_json")
            if _ij_json:
                with st.expander("Son uretilen JSON"):
                    st.code(_ij_json, language="json")

    if run_mode == "Adim adim" and st.button(
        "Writer: Ingilizce bolum (IEEE + Mermaid)",
        use_container_width=True,
    ):
        retrieved_docs = st.session_state.get("retrieved_parent_docs")
        if not retrieved_docs:
            st.error("Once multi-query retrieval calistirin.")
        else:
            model_name = _get_cached_gemini_chat_model_name()
            llm = _build_gemini_llm(model_name)

            def safe_invoke(prompt_text: str) -> str:
                """Bu tiklamaya ozel LLM oturumu ile invoke; retry str dondurur (.content yok)."""
                return _invoke_gemini_chat_with_retry(llm, prompt_text)

            writer = AcademicWriter(llm_invoke_func=safe_invoke)
            with st.spinner("Makale bolumu yaziliyor..."):
                result = writer.generate_section(
                    section_title=section_title,
                    section_goal=section_goal,
                    parent_documents=list(retrieved_docs),
                    max_parents=10,
                    repository_url=str(st.session_state.get("last_url") or ""),
                    operator_addendum=_extra_rules,
                    user_literature_block=_lit_block,
                )
            if result["metadata"]["status"] == "success":
                st.session_state["writer_draft_en"] = result["text"]
                st.session_state["writer_metadata"] = dict(result.get("metadata") or {})
                st.success(
                    f"Taslak olusturuldu. ({result['metadata']['parents_used']} kaynak parca kullanildi.)"
                )
            else:
                st.error(result["text"])

    full_combined = st.session_state.get("full_paper_combined_md")
    if full_combined:
        st.markdown("---")
        st.markdown("**Tam makale (birlestirilmis cikti)**")
        c_full_md, c_full_w = st.columns(2)
        with c_full_md:
            st.download_button(
                "Tam makaleyi indir (.md)",
                data=full_combined,
                file_name="full-paper-draft.md",
                mime="text/markdown",
                use_container_width=True,
                key="download_full_paper_md",
            )
        with c_full_w:
            _docx_full, _docx_note_full = _build_docx_from_markdown(full_combined)
            st.caption(_docx_note_full)
            st.download_button(
                "Tam makaleyi indir (.docx)",
                data=_docx_full,
                file_name="full-paper-draft.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="download_full_paper_docx",
            )
        with st.expander("Tam makale onizleme (ilk 12k karakter)"):
            prev = full_combined[:12000]
            st.text(prev + ("..." if len(full_combined) > 12000 else ""))

    draft = st.session_state.get("writer_draft_en")
    if draft:
        st.markdown("**Writer ciktisi (English) — son tek bolum veya tam makale son bolum**")
        st.markdown(draft)
        _wm = st.session_state.get("writer_metadata") or {}
        _tr = str(_wm.get("traceability") or "").strip()
        if _tr:
            with st.expander("TRACEABILITY (ic kontrol / hakem girdisi)"):
                st.markdown(_tr)
        _md_bundle = draft
        if _tr:
            _md_bundle = f"{draft}\n\n---\n\n## TRACEABILITY\n\n{_tr}"
        c_sec_md, c_sec_w = st.columns(2)
        with c_sec_md:
            st.download_button(
                "Taslagi indir (.md)",
                data=draft,
                file_name="section-draft.md",
                mime="text/markdown",
                use_container_width=True,
                key="download_section_md",
            )
        with c_sec_w:
            # Word: yalnizca Writer govdesi; TRACEABILITY tablosu .md indirmede kalir, sablona dusmez.
            _docx_sec, _docx_note_sec = _build_docx_from_markdown(draft)
            st.caption(_docx_note_sec)
            st.download_button(
                "Taslagi indir (.docx)",
                data=_docx_sec,
                file_name="section-draft.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="download_section_docx",
            )


def main() -> None:
    st.set_page_config(
        page_title="Code-to-Paper",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_compact_styles()

    target_dir = _render_sidebar()

    col_title, col_actions = st.columns([4, 1], vertical_alignment="bottom")
    with col_title:
        st.title("Code-to-Paper")
        st.caption(
            "Public GitHub deposunu indirin; gereksiz dosyalar süzülsün, kaynakları listeden seçip inceleyin."
        )

    default_url = "https://github.com/pallets/flask.git"
    github_url = st.text_input(
        "GitHub repo URL (HTTPS)",
        value=default_url,
        placeholder="https://github.com/kullanici/proje.git",
        help="Şimdilik yalnızca herkese açık (public) HTTPS adresleri desteklenir.",
    )

    with col_actions:
        st.write("")  # hizalama
        fetch_clicked = st.button("Repoyu çek ve süz", type="primary", use_container_width=True)

    if fetch_clicked:
        if not github_url.strip():
            st.error("Lütfen bir URL girin.")
            return
        with st.spinner("Klonlanıyor ve gereksiz dosyalar temizleniyor…"):
            try:
                commit, paths = clone_and_prepare(github_url.strip(), target_dir.strip())
            except GitCommandError as e:
                st.error(_friendly_git_error(e))
                return
            except OSError as e:
                st.error(f"Dosya sistemi hatası: {e}")
                return
            except Exception as e:
                st.error(f"Beklenmeyen hata: {e}")
                return
        st.session_state["commit"] = commit
        st.session_state["paths"] = paths
        st.session_state["root"] = Path(target_dir.strip()).resolve()
        st.session_state["stored_target_dir"] = target_dir.strip()
        st.session_state["last_url"] = github_url.strip()
        st.success(
            f"Tamamlandı. **{len(paths)}** dosya indeks için uygun. "
            f"Commit: `{commit[:12]}…`"
        )

    stored_dir = st.session_state.get("stored_target_dir")
    if stored_dir and stored_dir != target_dir.strip():
        st.warning(
            "Klon klasörü değiştirildi. Liste ve önizleme hâlâ önceki klasöre ait olabilir; "
            "güncel sonuç için aynı klasörle tekrar **Repoyu çek ve süz** düğmesine basın."
        )

    if "paths" not in st.session_state or not st.session_state["paths"]:
        st.info('Üstte HTTPS adresini girin ve **Repoyu çek ve süz** ile devam edin.')
        return

    paths: list[Path] = st.session_state["paths"]
    root: Path = st.session_state["root"]
    commit_full: str = st.session_state.get("commit", "")

    with st.expander("Son çekilen repo bilgisi", expanded=False):
        st.write(f"**Commit:** `{commit_full}`")
        st.write(f"**Kök:** `{root}`")
        if st.session_state.get("last_url"):
            st.write(f"**URL:** {st.session_state['last_url']}")

    rel_strings: list[str] = []
    for p in paths:
        try:
            rel_strings.append(str(p.relative_to(root)))
        except ValueError:
            rel_strings.append(str(p))

    m1, m2, m3 = st.columns(3)
    m1.metric("Dosya sayısı", len(paths))
    m2.metric("Önizleme limiti", f"{_MAX_PREVIEW_CHARS:,} kar.")
    m3.metric("Klon klasörü", target_dir.strip() or "—")

    last_url = str(st.session_state.get("last_url") or "").strip()
    if last_url and commit_full:
        _render_rag_indexing_section(
            paths=paths,
            root=root,
            repo_url=last_url,
            commit_hash=commit_full,
        )

    st.subheader("Dosya seçimi")
    search = st.text_input(
        "Dosya ara (yol veya dosya adı)",
        "",
        placeholder="örn. views, README, main.py",
        help="Liste göreli yollar üzerinde büyük/küçük harf duyarsız filtrelenir.",
    )
    q = search.strip().lower()
    if q:
        filtered_idx = [i for i, s in enumerate(rel_strings) if q in s.lower()]
    else:
        filtered_idx = list(range(len(paths)))

    if not filtered_idx:
        st.warning("Aramanızla eşleşen dosya yok; filtreyi temizleyin veya farklı bir metin deneyin.")
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        choice = st.selectbox(
            "Dosya",
            options=filtered_idx,
            format_func=lambda i: rel_strings[i],
            label_visibility="collapsed",
        )
    selected = paths[choice]

    with right:
        st.markdown(f"**Seçili:** `{rel_strings[choice]}`")

    try:
        text = selected.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        st.error(f"Dosya okunamadı: {e}")
        return

    truncated = len(text) > _MAX_PREVIEW_CHARS
    if truncated:
        text = text[:_MAX_PREVIEW_CHARS] + "\n\n… (önizleme kesildi)"
        st.warning(
            f"Bu dosya çok büyük; yalnızca ilk **{_MAX_PREVIEW_CHARS:,}** karakter gösteriliyor. "
            "Tam metin için editörde açın."
        )

    lang = _detect_language(selected.suffix)
    st.code(text, language=lang or "text")

    st.divider()
    st.caption(f"Tam yol: `{selected}`")
    st.divider()
    _render_agent_preview_panel()


if __name__ == "__main__":
    main()
