"""
Streamlit arayüzü: GitHub URL girilir, repo klonlanır ve süzülür; dosyalar listeden
seçilerek içerik önizlemesi gösterilir. RAG / embedding bu aşamada yoktur.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from github_handler import clone_and_prepare

# Önizlemede çok büyük dosyaları kesmek için üst sınır (karakter).
_MAX_PREVIEW_CHARS = 80_000


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


def main() -> None:
    st.set_page_config(page_title="Code-to-Paper — Kaynak önizleme", layout="wide")
    st.title("Code-to-Paper")
    st.caption("Public GitHub deposunu çekin; süzülmüş dosyaları burada inceleyin.")

    default_url = "https://github.com/pallets/flask.git"
    github_url = st.text_input("GitHub repo URL (HTTPS)", value=default_url)

    target_dir = st.sidebar.text_input("Klon klasörü", value="data/source")

    if st.button("Repoyu çek ve süz", type="primary"):
        if not github_url.strip():
            st.error("Lütfen bir URL girin.")
            return
        with st.spinner("Klonlanıyor ve gereksiz dosyalar temizleniyor..."):
            try:
                commit, paths = clone_and_prepare(github_url.strip(), target_dir.strip())
            except Exception as e:
                st.error(f"Hata: {e}")
                return
        st.session_state["commit"] = commit
        st.session_state["paths"] = paths
        st.session_state["root"] = Path(target_dir.strip()).resolve()
        st.success(f"Tamam. Commit: `{commit[:12]}…` — {len(paths)} dosya indeks için uygun.")

    if "paths" not in st.session_state or not st.session_state["paths"]:
        st.info('URL girip "**Repoyu çek ve süz**" butonuna basın.')
        return

    paths: list[Path] = st.session_state["paths"]
    root: Path = st.session_state["root"]

    st.subheader("Dosya listesi")
    # Selectbox için string listesi (göreli yol daha okunaklı)
    rel_strings = []
    for p in paths:
        try:
            rel_strings.append(str(p.relative_to(root)))
        except ValueError:
            rel_strings.append(str(p))

    choice = st.selectbox("Dosya seç", options=range(len(paths)), format_func=lambda i: rel_strings[i])
    selected = paths[choice]

    try:
        text = selected.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        st.error(f"Okunamadı: {e}")
        return

    truncated = False
    if len(text) > _MAX_PREVIEW_CHARS:
        text = text[:_MAX_PREVIEW_CHARS] + "\n\n… (önizleme kesildi)"
        truncated = True
    if truncated:
        st.warning(f"Dosya büyük; yalnızca ilk {_MAX_PREVIEW_CHARS:,} karakter gösteriliyor.")

    lang = _detect_language(selected.suffix)
    st.code(text, language=lang or "text")

    st.divider()
    st.caption(f"Tam yol: `{selected}`")


if __name__ == "__main__":
    # Çalıştırma: proje kökünde `streamlit run main.py`
    main()
