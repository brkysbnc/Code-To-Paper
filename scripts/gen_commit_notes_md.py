"""Baseline commit sonrasinda secilen commitlerin tam `git show` ciktisini UTF-8 markdown'a yazar."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Su anki kullanim: `79cde2e` uzerinde kalan son 2 commit (91f9f1e, cd545b4).
BASELINE_FULL = "79cde2e4c5dc25d0f94face8ee7ca868145f2b69"
BASELINE_SHORT = BASELINE_FULL[:7]

OUT = REPO_ROOT / "docs" / f"COMMITS_AFTER_BASELINE_{BASELINE_SHORT}.md"

COMMITS: list[tuple[str, str]] = [
    ("91f9f1e0dc04ed285a11a948fca81919b9b5d14e", "COMMIT 1/2"),
    ("cd545b459ae798efb1905cb9b770ad6fb9a39823", "COMMIT 2/2"),
]

HEADER = f"""# Geri donus notu: baseline `{BASELINE_SHORT}` sonrasi 2 commit

Bu dosya, **bu baseline'in hemen ustunde** kalan **tam 2 commit**in kelimesi kelimesine `git show --no-color` ciktisidir.

GitHub sirasi (yeniden eskiye): once `cd545b4` (Mermaid), onun altinda `91f9f1e` (SOURCE SEPARATION refactor).

## Baseline (reset / checkout hedefi)

- **Tam hash:** `{BASELINE_FULL}`
- **Ozeti:** fix(writer): literatur terminolojisinin repoya sizmasini engelle  
- **Bu committe degisen dosyalar:** `agents/writer.py`, `orchestration/paper_blueprint.py`

## Bu dosyada kronolojik sira (yeniden uygulama: eski commit once)

| Sira | Kisa hash | Ozet |
|------|-----------|------|
| 1 | `91f9f1e` | refactor(writer): literatur sizintisi icin ilke tabanli SOURCE SEPARATION |
| 2 | `cd545b4` | fix(export): Word'e Mermaid kod sizmasini engelle; writer'da govde icin Mermaid yasagi |

**Degisen dosyalar (commit bazinda):**

1. `91f9f1e`: `agents/writer.py`, `orchestration/paper_blueprint.py`
2. `cd545b4`: `agents/writer.py`, `export/ieee_template_export.py`

## Yeniden uygulama (baseline'a dondukten sonra)

```bash
git cherry-pick 91f9f1e0dc04ed285a11a948fca81919b9b5d14e
git cherry-pick cd545b459ae798efb1905cb9b770ad6fb9a39823
```

---

"""


def main() -> None:
    parts: list[str] = [HEADER]
    total = len(COMMITS)
    for full_hash, label in COMMITS:
        proc = subprocess.run(
            ["git", "show", full_hash, "--no-color"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        short = full_hash[:7]
        parts.append(f"\n\n## ===== {label}: `git show {short}` =====\n\n")
        parts.append("````text\n")
        parts.append(proc.stdout)
        if not proc.stdout.endswith("\n"):
            parts.append("\n")
        parts.append("````\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes), baseline={BASELINE_SHORT}, commits={total}")


if __name__ == "__main__":
    main()
