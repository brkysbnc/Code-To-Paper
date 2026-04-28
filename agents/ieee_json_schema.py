"""
IEEE tam makale JSON semasi: parse, dogrulama ve zorunlu bolum doldurma.

Writer ciktisi yalnizca JSON olmalidir; bu modul Word uretimine hazir dict uretir.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

# Sablon I–V ana bolum basliklari (Roman rakami Word stilinde otomatik; metne yazilmaz).
_DEFAULT_SECTIONS: List[Dict[str, Any]] = [
    {"heading": "Introduction", "body": "", "subsections": []},
    {"heading": "Related Work", "body": "", "subsections": []},
    {"heading": "System Architecture", "body": "", "subsections": []},
    {"heading": "Evaluation", "body": "", "subsections": []},
    {"heading": "Conclusion", "body": "", "subsections": []},
]

_INSUFF = "[Insufficient evidence — section requires more context]"


def parse_ieee_paper_json(raw: str) -> Dict[str, Any]:
    """
    LLM cevabindan JSON cikarir (```json ... ``` veya duz metin).
    """
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    return json.loads(text)


def _section_nonempty(sec: Dict[str, Any]) -> bool:
    """Bolumde anlamli metin veya alt baslik var mi kontrol eder."""
    if (sec.get("body") or "").strip():
        return True
    for sub in sec.get("subsections") or []:
        if not isinstance(sub, dict):
            continue
        if (sub.get("body") or "").strip():
            return True
        for ss in sub.get("subsubsections") or []:
            if isinstance(ss, dict) and (ss.get("body") or "").strip():
                return True
    return False


def normalize_ieee_paper_content(data: Dict[str, Any], *, repository_url: str) -> Dict[str, Any]:
    """
    Zorunlu alanlari doldurur; bes ana bolumu sablon sirasina oturtur; bos bolumlere INSUFF yazar.
    """
    out: Dict[str, Any] = {}
    out["title"] = str(data.get("title") or "Untitled Paper").strip()
    authors = data.get("authors")
    if isinstance(authors, list) and authors:
        out["authors"] = [str(a).strip() for a in authors if str(a).strip()]
    else:
        out["authors"] = [
            "Corresponding Author\nAffiliation\nCity, Country\nemail@example.com",
        ]
    out["abstract"] = str(data.get("abstract") or _INSUFF).strip()
    out["keywords"] = str(data.get("keywords") or "software engineering, documentation").strip()

    incoming = data.get("sections")
    if not isinstance(incoming, list):
        incoming = []

    merged: List[Dict[str, Any]] = []
    for i, template in enumerate(_DEFAULT_SECTIONS):
        base = dict(template)
        if i < len(incoming) and isinstance(incoming[i], dict):
            inc = incoming[i]
            if str(inc.get("heading") or "").strip():
                base["heading"] = str(inc["heading"]).strip()
            if "body" in inc:
                base["body"] = str(inc.get("body") or "")
            if isinstance(inc.get("subsections"), list):
                base["subsections"] = _normalize_subsections(inc["subsections"])
        if not _section_nonempty(base):
            base["body"] = _INSUFF
        merged.append(base)
    out["sections"] = merged

    out["acknowledgment"] = str(data.get("acknowledgment") or "").strip()
    refs = data.get("references")
    if isinstance(refs, list) and refs:
        out["references"] = [str(r).strip() for r in refs if str(r).strip()]
    else:
        web = repository_url[:-4] if repository_url.lower().endswith(".git") else repository_url
        out["references"] = [
            f"[1] Code-to-Paper repository, GitHub, 2026. [Online]. Available: {web or 'n/a'}",
        ]
    return out


def _normalize_subsections(items: List[Any]) -> List[Dict[str, Any]]:
    """Alt bolum listesini guvenli dict listesine cevirir."""
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        sub: Dict[str, Any] = {
            "heading": str(item.get("heading") or "Subsection").strip(),
            "body": str(item.get("body") or ""),
        }
        raw_ss = item.get("subsubsections")
        if isinstance(raw_ss, list):
            sub["subsubsections"] = [
                {
                    "heading": str(x.get("heading") or "").strip(),
                    "body": str(x.get("body") or ""),
                }
                for x in raw_ss
                if isinstance(x, dict)
            ]
        else:
            sub["subsubsections"] = []
        out.append(sub)
    return out
