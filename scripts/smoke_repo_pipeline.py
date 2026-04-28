"""
Kucuk bir public repo klonlayip run_section_pipeline ile uctan uca duman testi.

Kullanim (proje kokunden):
  python scripts/smoke_repo_pipeline.py

GOOGLE_API_KEY yoksa: sadece klon + sablon .docx uretimi (LLM adimlari atlanir).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Proje kokunu path'e ekle
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Varsayilan: cok kucuk bir PyPA ornek deposu
DEFAULT_REPO = "https://github.com/pypa/sampleproject.git"


def _pick_py_paths(paths: list[Path], limit: int) -> list[Path]:
    """Indekslemek icin once .py dosyalarini secer."""
    py = [p for p in paths if p.suffix.lower() == ".py" and p.is_file()]
    if len(py) >= limit:
        return py[:limit]
    return list(paths)[: max(1, min(limit, len(paths)))]


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    tmp = Path(tempfile.mkdtemp(prefix="ctp_smoke_"))
    clone_root = tmp / "repo"
    url = os.getenv("SMOKE_REPO_URL", DEFAULT_REPO).strip()

    print(f"[1/4] Klon: {url} -> {clone_root}")
    try:
        from github_handler import clone_and_prepare

        commit, paths = clone_and_prepare(url, str(clone_root))
    except Exception as exc:  # noqa: BLE001
        print("Klon hatasi:", exc)
        shutil.rmtree(tmp, ignore_errors=True)
        return 1

    picked = _pick_py_paths(paths, 8)
    print(f"[2/4] Commit={commit[:8]}… indekslenecek dosya={len(picked)} (toplam aday={len(paths)})")

    # IEEE sablon + kisa metin (LLM olmadan)
    print("[3/4] IEEE sablonu uzerine kisa .docx…")
    try:
        from export.ieee_template_export import markdown_to_ieee_template_docx_bytes, resolve_default_ieee_template

        tpl = resolve_default_ieee_template()
        if tpl is None:
            print("  (sablon yok, atlandi)")
        else:
            docx = markdown_to_ieee_template_docx_bytes(
                tpl,
                "## Smoke test\n\nKlon ve sablon hattı çalışıyor.\n",
            )
            out_docx = tmp / "smoke_template.docx"
            out_docx.write_bytes(docx)
            print(f"  yazildi: {out_docx} ({len(docx)} bayt)")
    except Exception as exc:  # noqa: BLE001
        print("  IEEE docx hatasi:", exc)

    if not api_key:
        print("[4/4] GOOGLE_API_KEY yok — RAG+Writer atlandi (.env icine GOOGLE_API_KEY ekleyip tekrar calistirin).")
        shutil.rmtree(tmp, ignore_errors=True)
        return 0

    print("[4/4] run_section_pipeline (Gemini + Chroma, 1 bolum)…")
    try:
        from orchestration.section_pipeline import run_section_pipeline

        result = run_section_pipeline(
            repo_url=url,
            commit_hash=commit,
            repo_root=clone_root.resolve(),
            paths_for_index=picked,
            section_title="Overview",
            section_goal="Briefly describe the packaged Python sample layout from code only.",
            max_index_files=8,
            similarity_threshold=0.35,
            top_k_per_query=4,
            max_planner_queries=4,
            writer_extra_rules="Keep PART 1 under 400 words if possible.",
            user_literature_block="",
        )
    except Exception as exc:  # noqa: BLE001
        print("Pipeline hatasi:", exc)
        shutil.rmtree(tmp, ignore_errors=True)
        return 1

    if result.get("error"):
        print("Pipeline basarisiz:", result.get("failed_step"), result.get("error"))
        shutil.rmtree(tmp, ignore_errors=True)
        return 1

    wt = (result.get("writer_text") or "").strip()
    print("Writer cikti uzunlugu:", len(wt), "karakter")
    print("Ilk 400 karakter:\n", wt[:400], "\n…")
    out_md = tmp / "smoke_writer.md"
    out_md.write_text(wt, encoding="utf-8")
    print(f"Tam metin: {out_md}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("Gecici klasor silindi. Duman testi OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
