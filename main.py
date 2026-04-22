"""
Streamlit arayüzü: GitHub URL girilir, repo klonlanır ve süzülür; dosyalar listeden
seçilerek içerik önizlemesi gösterilir.

RAG (deneysel): Gemini embedding ile parent-child indeksleme, Planner sorgulari ve
multi-query retrieval bu ekrandan tetiklenebilir.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from agents.writer import AcademicWriter
import streamlit as st
from dotenv import load_dotenv
from google import genai
from git.exc import GitCommandError
from langchain_google_genai import ChatGoogleGenerativeAI

from github_handler import clone_and_prepare
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
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_GEMINI_FALLBACK_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
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
            "Once repoyu cekin. Indeksleme secilen dosyalarda embedding API cagrisi yapar; "
            "dosya sayisini dusuk tutmaniz onerilir."
        )
        max_files = st.number_input(
            "Indekslenecek max dosya sayisi",
            min_value=1,
            max_value=80,
            value=20,
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
    if st.button("Planner sorgularini uret", use_container_width=True):
        try:
            model_name = _get_cached_gemini_chat_model_name()
            llm = _build_gemini_llm(model_name)
            queries = generate_planner_queries(
                llm,
                section_title=section_title,
                section_goal=section_goal,
                max_queries=6,
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
    if retriever is None:
        st.warning("Multi-query retrieval icin once yukaridaki RAG indekslemesini calistirin.")
        return

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

    if st.button("Writer: Ingilizce bolum (IEEE + Mermaid)", use_container_width=True):
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
                )
            if result["metadata"]["status"] == "success":
                st.session_state["writer_draft_en"] = result["text"]
                st.success(
                    f"Taslak olusturuldu. ({result['metadata']['parents_used']} kaynak parca kullanildi.)"
                )
            else:
                st.error(result["text"])

    draft = st.session_state.get("writer_draft_en")
    if draft:
        st.markdown("**Writer ciktisi (English)**")
        st.markdown(draft)


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
