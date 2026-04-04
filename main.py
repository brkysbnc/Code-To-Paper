"""
Streamlit arayüzü: GitHub URL girilir, repo klonlanır ve süzülür; dosyalar listeden
seçilerek içerik önizlemesi gösterilir. RAG / embedding bu aşamada yoktur.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from git.exc import GitCommandError

from github_handler import clone_and_prepare

# Önizlemede çok büyük dosyaları kesmek için üst sınır (karakter).
_MAX_PREVIEW_CHARS = 80_000


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
    st.sidebar.markdown("##### İndeksleme (yakında)")
    st.sidebar.info(
        "Embedding ve ChromaDB adımı henüz bağlı değil. "
        "Bu ekran yalnızca süzülmüş dosya listesi ve kod önizlemesi sunar."
    )
    st.sidebar.caption("Orchestration: LangChain · Vektör: ChromaDB · LLM: OpenAI (plan)")
    return target_dir


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


if __name__ == "__main__":
    main()
