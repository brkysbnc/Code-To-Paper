"""
Kullanici literatur parcalarini (yapistirilan metin / PDF) bolum hedefine gore süzer.

LLM yalnizca JSON dondurur; uygun parcalar Writer promptunda [2], [3], ... olarak kullanilir.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, List, Sequence, Tuple

logger = logging.getLogger(__name__)


def extract_pdf_text_to_string(file_bytes: bytes, *, max_pages: int = 30) -> str:
    """
    PDF baytlarindan duz metin cikarir; sayfa sayisini sinirlayarak token sismesini azaltir.
    """
    try:
        from io import BytesIO

        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PDF okumak icin 'pypdf' paketi gerekli (requirements.txt).") from exc

    reader = PdfReader(BytesIO(file_bytes))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            parts.append(f"\n...[PDF truncated after {max_pages} pages]")
            break
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()


def split_pasted_literature(paste: str) -> List[Tuple[str, str]]:
    """
    Yapistirilan metni '---' ayiricilariyla parcalara bolur; ilk satir 'Title: ...' ise baslik olarak alir.
    """
    raw = (paste or "").strip()
    if not raw:
        return []
    blocks: list[tuple[str, str]] = []
    for i, part in enumerate(re.split(r"\n-{3,}\n", raw)):
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        title = f"Pasted source {i + 1}"
        body = part
        if lines and lines[0].strip().lower().startswith("title:"):
            title = lines[0].split(":", 1)[1].strip() or title
            body = "\n".join(lines[1:]).strip()
        blocks.append((title, body[:50_000]))
    return blocks


def _escape_curly_braces(s: str) -> str:
    """str.format ile kullanici metni birlestirirken suslu parantezleri korumak icin."""
    return (s or "").replace("{", "{{").replace("}", "}}")


def format_approved_for_writer(approved: Sequence[Tuple[str, str]], *, max_chars_per_item: int = 8000) -> str:
    """
    Onaylanmis (baslik, metin) listesini Writer'in USER_LITERATURE blogu icin metne cevirir.
    """
    lines: list[str] = []
    for i, (title, text) in enumerate(approved, start=2):
        snippet = (text or "").strip()[:max_chars_per_item]
        lines.append(f"[{i}] {title.strip() or f'Source {i}'}\n{snippet}")
    return "\n\n".join(lines).strip()


def _extract_json_object(raw: str) -> dict | None:
    """
    Defansif JSON cikarici: ham metni once oldugu gibi parse etmeyi dener;
    basarisiz olursa metnin icindeki en buyuk `{...}` blogunu regex ile yakalayip
    onu parse eder. Hicbiri tutmazsa None doner; cagri tarafi None gelince
    fail-soft policy uygular.
    """
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def filter_literature_relevance(
    llm_invoke: Callable[[str], str],
    *,
    section_title: str,
    section_goal: str,
    repository_hint: str,
    items: Sequence[Tuple[str, str]],
    max_prompt_chars_per_item: int = 4500,
) -> tuple[list[Tuple[str, str]], list[dict[str, Any]]]:
    """
    Her literatur parcasi icin LLM'den include/exclude JSON'u alir; dahil edilenleri dondurur.

    Donus: (onayli_baslik_metin_listesi, dislanan_kayitlari).

    Fail-soft: LLM cevabi parse edilemezse kullanici secimi kaybolmasin diye
    tum parcalar approved olarak donulur ve excluded_meta'ya tek satirlik bir
    'gate_bypassed' notu eklenir.
    """
    if not items:
        return [], []

    numbered: list[str] = []
    for idx, (title, body) in enumerate(items):
        t = _escape_curly_braces((title or f"item_{idx}").strip())
        b = _escape_curly_braces((body or "").strip()[:max_prompt_chars_per_item])
        numbered.append(f"### INDEX {idx}\nTITLE: {t}\nTEXT:\n{b}\n")

    bundle = "\n".join(numbered)
    prompt = (
        "You are a strict relevance gate for a technical academic paper section.\n"
        f"Section title: {section_title}\n"
        f"Section goal: {section_goal}\n"
        f"Repository context (short): {repository_hint or 'n/a'}\n\n"
        "Below are INDEXed excerpts the user wants to use as external literature. "
        "Some may be irrelevant.\n\n"
        "OUTPUT REQUIREMENTS — STRICT:\n"
        "- Respond with RAW JSON only. No markdown fences. No preamble. No postscript.\n"
        "- The first character of your reply MUST be `{` and the last MUST be `}`.\n"
        '- Exact shape: {"include":[<int>,...],"exclude":[{"index":<int>,"reason":"<short>"}]}\n'
        '- Example of a valid full reply: {"include":[0,2],"exclude":[{"index":1,"reason":"off-topic"}]}\n'
        "Rules:\n"
        "- include: integer indices that materially support the section goal.\n"
        "- exclude: all other indices with honest reasons.\n"
        "- If you are uncertain about an item, PREFER include over exclude.\n"
        "- If nothing is relevant, include may be an empty list.\n\n"
        f"LITERATURE CANDIDATES:\n{bundle}"
    )

    raw = llm_invoke(prompt).strip()
    data = _extract_json_object(raw)

    if data is None:
        logger.warning(
            "Literatur surzgec JSON parse edilemedi; fail-soft: %d parca otomatik onaylandi.",
            len(items),
        )
        return list(items), [{
            "index": -1,
            "reason": "gate_bypassed_due_to_json_parse_failure",
        }]

    # CLAUDE GUARD: LLM bazen "include": 0 (skaler) veya "include": "0" donderiyor;
    # liste haline ceviriyoruz ki asagidaki for x in raw_include patlamasin.
    raw_include = data.get("include") or []
    if not isinstance(raw_include, list):
        raw_include = [raw_include]

    include_set: set[int] = set()
    for x in raw_include:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= xi < len(items):
            include_set.add(xi)

    reason_by_idx: dict[int, str] = {}
    for e in data.get("exclude") or []:
        if not isinstance(e, dict):
            continue
        try:
            ix = int(e["index"])
        except (KeyError, TypeError, ValueError):
            continue
        reason_by_idx[ix] = str(e.get("reason", ""))

    approved: list[tuple[str, str]] = []
    excluded_meta: list[dict[str, Any]] = []
    for idx, (title, body) in enumerate(items):
        if idx in include_set:
            approved.append((title, body))
        else:
            excluded_meta.append({"index": idx, "reason": reason_by_idx.get(idx, "excluded_by_gate")})

    return approved, excluded_meta
