"""
LLM-as-a-judge faithfulness verifier for Code-To-Paper.

Parses the Writer's TRACEABILITY table, locates matching parent-document evidence,
and asks Gemini (in a single batched call per section) whether each claim is
supported, partial, or unsupported.

Public API
----------
    judge_section_faithfulness(
        *,
        writer_text,
        writer_traceability,
        parent_documents,
        llm_invoke,
        max_claims=12,
    ) -> dict

Cache: data/faithfulness_cache/<sha256>.json  (already covered by data/ in .gitignore)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Callable


def _normalize_path_for_evidence_match(raw: str) -> str:
    """
    TRACEABILITY tablosundaki source_file ile metadata file_path'i karsilastirmak icin yolu duzeltir.
    Basit on ekler (./, /, \\) ve backslash'leri kaldirarak eslesmeyi guclendirir.
    """
    s = (raw or "").strip().lower().replace("\\", "/")
    # Remove markdown backticks or quotes that the LLM might hallucinate
    s = s.replace("`", "").replace('"', "").replace("'", "")
    while s.startswith("./"):
        s = s[2:]
    return s.lstrip("/")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CACHE_DIR = Path("data") / "faithfulness_cache"

_TRACE_ROW_RE = re.compile(
    r"^\s*\|\s*(C?\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$"
)

_JUDGE_PROMPT_HEADER = """\
You are a strict technical reviewer verifying claims against repository evidence.

For EACH claim below, decide whether the evidence text supports the claim:
- "supported": the evidence explicitly states or directly implements the claim.
- "partial": the evidence contains related context but does not fully support the claim.
- "unsupported": the evidence does not support the claim at all (or evidence is empty).

Output a SINGLE JSON object with this exact shape — no markdown, no preamble:

{"verdicts":[{"id":"C1","verdict":"supported","evidence_quote":"<10-30 word quote or empty>","judge_note":"<short phrase>"}, ...]}

CLAIMS AND EVIDENCE:
"""

_VERDICT_WEIGHT: dict[str, float] = {
    "supported": 1.0,
    "partial": 0.5,
    "unsupported": 0.0,
}


# ---------------------------------------------------------------------------
# Step 1 — Parse TRACEABILITY table
# ---------------------------------------------------------------------------

def _parse_traceability_rows(text: str) -> list[dict]:
    """Extracts claim rows from the TRACEABILITY markdown table."""
    rows: list[dict] = []
    for line in (text or "").splitlines():
        m = _TRACE_ROW_RE.match(line)
        if not m:
            continue
        cid, summary, src, lines, notes = m.groups()
        if cid.strip().lower() == "claim id":  # header row
            continue
        
        # Normalize ID to always start with "C"
        cid_clean = cid.strip().upper()
        if not cid_clean.startswith("C"):
            cid_clean = f"C{cid_clean}"

        rows.append(
            {
                "id": cid_clean,
                "summary": summary.strip(),
                "source_file": src.strip(),
                "lines": lines.strip(),
                "notes": notes.strip(),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Step 2 — Build evidence map
# ---------------------------------------------------------------------------

def _source_matches_any_parent_document(target_raw: str, parent_documents: list) -> bool:
    """
    TRACEABILITY'deki Source file herhangi bir parent dokuman file_path'i ile eslesiyor mu.
    Satir araligi kontrol edilmez; sadece dosya/yol eslemesi (Writer yanlis repo dosyasi yazdiginda
    ozet-literatur fallback'inin ne zaman devreye girecegini ayirmak icin).
    """
    if not (target_raw or "").strip():
        return False
    target = _normalize_path_for_evidence_match(target_raw)
    target_name = Path(target_raw.replace("\\", "/")).name.lower() if target_raw else ""
    for d in parent_documents:
        path_raw = str(d.metadata.get("file_path", "") or "")
        if not path_raw.strip():
            continue
        path = _normalize_path_for_evidence_match(path_raw)
        path_name = Path(path_raw.replace("\\", "/")).name.lower() if path_raw else ""
        if (
            bool(target and path and (target in path or path in target))
            or bool(target and path and (path.endswith(target) or target.endswith(path)))
            or bool(target_name and path_name and target_name == path_name)
        ):
            return True
    return False


def _find_evidence(
    claim: dict,
    parent_documents: list,
    user_literature_block: str = "",
) -> str:
    """Returns concatenated text of parent docs whose file_path and line range
    overlap with the claim's source. Empty string if none match."""
    target_raw = str(claim.get("source_file") or "").strip()
    if target_raw.lower() == "user literature":
        return (user_literature_block or "").strip()

    # Writer bazen USER_LITERATURE_APPROVED kaynaklari icin PDF/txt dosya adini ("makale1.pdf")
    # Source file sutununa yazar; repo path'i ile eslesmez -> kanit bos kalirdi.
    # Yuklenen dokuman uzantisina sahip ve hicbir parent dokumanla eslesmiyorsa tum literatur blogunu kanit say.
    if (user_literature_block or "").strip():
        ext = Path(target_raw).suffix.lower()
        if ext in (".pdf", ".txt", ".docx", ".md"):
            tn = Path(target_raw.replace("\\", "/")).name.lower()
            found_in_repo = False
            for d in parent_documents:
                fp = str(d.metadata.get("file_path", "") or "")
                if not fp.strip():
                    continue
                pn = Path(fp.replace("\\", "/")).name.lower()
                pnorm = _normalize_path_for_evidence_match(fp)
                tnorm = _normalize_path_for_evidence_match(target_raw)
                if tn == pn or (tnorm and pnorm and (tnorm in pnorm or pnorm in tnorm)):
                    found_in_repo = True
                    break
            if not found_in_repo:
                return (user_literature_block or "").strip()

    # Writer bazen literatur iddialarini repo dosyasina yaziyor; kaynak hic parent'ta yoksa
    # veya yanlis dosya yazilmissa, ozet metni USER_LITERATURE ile ortusuyorsa literaturu kanit say.
    if (user_literature_block or "").strip() and target_raw.lower() != "user literature":
        if not _source_matches_any_parent_document(target_raw, parent_documents):
            summary = str(claim.get("summary") or "").lower()
            summary_words = [w for w in summary.split() if len(w) > 4]
            lit_lower = user_literature_block.lower()
            matches = sum(1 for w in summary_words if w in lit_lower)
            if summary_words and matches / len(summary_words) > 0.4:
                return (user_literature_block or "").strip()

    target = _normalize_path_for_evidence_match(target_raw)
    target_name = Path(target_raw.replace("\\", "/")).name.lower() if target_raw else ""
    try:
        nums = re.findall(r"\d+", claim["lines"])[:2] or ["0", "0"]
        lo, hi = int(nums[0]), int(nums[1]) if len(nums) > 1 else int(nums[0])
    except (ValueError, IndexError):
        lo, hi = 0, 0

    parts: list[str] = []
    for d in parent_documents:
        path_raw = str(d.metadata.get("file_path", ""))
        path = _normalize_path_for_evidence_match(path_raw)
        path_name = Path(path_raw.replace("\\", "/")).name.lower() if path_raw else ""
        # Esnek yol: alt yol / ters slash / sadece dosya adi varyantlari
        path_matches = (
            bool(target and path and (target in path or path in target))
            or bool(target and path and (path.endswith(target) or target.endswith(path)))
            or bool(target_name and path_name and target_name == path_name)
        )
        if not path_matches:
            continue
        s = int(d.metadata.get("start_line", 0) or 0)
        e = int(d.metadata.get("end_line", 0) or 0)
        if hi == 0 or (s <= hi and e >= lo):
            parts.append(d.page_content or "")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 3 — Build batched judge prompt
# ---------------------------------------------------------------------------

def _build_judge_prompt(claims_with_evidence: list[dict], user_literature_block: str = "") -> str:
    """Constructs a single prompt covering all claims."""
    header = _JUDGE_PROMPT_HEADER
    if (user_literature_block or "").strip():
        header += (
            "\nUSER LITERATURE (external sources the writer was allowed to cite):\n"
            "---\n"
            f"{user_literature_block[:6000]}\n"
            "---\n"
            "Claims with Source file = 'User literature' must be verified against\n"
            "the USER LITERATURE block above, NOT against the codebase.\n"
            "A claim is SUPPORTED if its assertion is grounded in either the\n"
            "codebase evidence OR the user literature, depending on its source.\n\n"
        )

    sections: list[str] = []
    for c in claims_with_evidence:
        evidence = (c.get("evidence") or "")[:2000]
        sections.append(
            f"### CLAIM {c['id']}\n"
            f"SUMMARY: {c['summary']}\n"
            f"SOURCE FILE: {c['source_file']}\n"
            f"LINES: {c['lines']}\n"
            f"EVIDENCE TEXT (truncated to 2000 chars):\n---\n{evidence}\n---"
        )
    return header + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Step 4 — Defensive JSON extraction (same pattern as ieee_json_writer.py)
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> dict | None:
    """Finds the first { and last } in text and attempts json.loads on that slice."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Step 5 — Score aggregation
# ---------------------------------------------------------------------------

def _aggregate(verdicts: list[dict]) -> tuple[float, str]:
    """Weighted mean of verdicts → (score, label)."""
    if not verdicts:
        return 0.0, "low"
    total = sum(_VERDICT_WEIGHT.get(v.get("verdict", "unsupported"), 0.0) for v in verdicts)
    score = round(total / len(verdicts), 3)
    label = "high" if score >= 0.8 else "medium" if score >= 0.6 else "low"
    return score, label


# ---------------------------------------------------------------------------
# Step 6 — Caching
# ---------------------------------------------------------------------------

def _cache_key(writer_text: str, writer_traceability: str, parent_documents: list) -> str:
    """sha256 hash over writer text + traceability + sorted chunk metadata ids."""
    chunk_ids = sorted(
        str(d.metadata.get("file_path", "")) + ":" + str(d.metadata.get("start_line", ""))
        for d in parent_documents
    )
    # traceability uzunlugu da anahtara dahil: ayni writer_text ile farkli tablo/kolon varyasyonlarinda
    # eski 0.00 cache'e takilmayi azaltir.
    payload = (
        writer_text
        + "\x00"
        + writer_traceability
        + "\x00"
        + str(len(writer_traceability))
        + "\x00"
        + "|".join(chunk_ids)
    )
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _load_cache(key: str) -> dict | None:
    path = _CACHE_DIR / f"{key}.json"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            result = json.load(f)
        logger.info("faithfulness_judge: cache hit (%s)", key[:12])
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("faithfulness_judge: cache read error (%s): %s", key[:12], exc)
        return None


def _save_cache(key: str, result: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _CACHE_DIR / f"{key}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.debug("faithfulness_judge: cache saved (%s)", key[:12])
    except Exception as exc:  # noqa: BLE001
        logger.warning("faithfulness_judge: cache write error: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge_section_faithfulness(
    *,
    writer_text: str,
    writer_traceability: str,
    parent_documents: list,
    llm_invoke: Callable[[str], str],
    max_claims: int = 12,
    user_literature_block: str = "",
) -> dict:
    """
    Verifies each claim in the TRACEABILITY table against parent document evidence.

    Parameters
    ----------
    writer_text : str
        The PART 1 body text produced by AcademicWriter (used for caching).
    writer_traceability : str
        The raw TRACEABILITY block (everything from TRACEABILITY: onward).
    parent_documents : list
        LangChain Document objects retrieved for this section.
    llm_invoke : Callable[[str], str]
        A callable that takes a prompt string and returns the LLM response string.
        Should be the same retry-wrapped function used by the Writer.
    max_claims : int
        Maximum number of claims to judge (to stay within token limits).

    Returns
    -------
    dict with keys: score, label, claim_count, claims, raw_llm_response
    """
    # --- Cache check ---
    key = _cache_key(writer_text, writer_traceability, parent_documents)
    cached = _load_cache(key)
    if cached is not None:
        return cached

    # --- Parse claims ---
    all_claims = _parse_traceability_rows(writer_traceability)
    claims = all_claims[:max_claims]
    if not claims:
        result: dict = {
            "score": 0.0,
            "label": "low",
            "claim_count": 0,
            "code_claims_total": 0,
            "code_claims_supported": 0,
            "literature_claims_total": 0,
            "literature_claims_supported": 0,
            "claims": [],
            "raw_llm_response": "",
        }
        _save_cache(key, result)
        return result

    # --- Build evidence map (no LLM call for unmatched claims) ---
    claims_with_evidence: list[dict] = []
    no_evidence_ids: set[str] = set()
    for claim in claims:
        evidence = _find_evidence(claim, parent_documents, user_literature_block)
        claims_with_evidence.append({**claim, "evidence": evidence})
        if not evidence.strip():
            no_evidence_ids.add(claim["id"])

    logger.info(
        "faithfulness_judge: %d claims (%d with evidence, %d no-evidence→unsupported)",
        len(claims),
        len(claims) - len(no_evidence_ids),
        len(no_evidence_ids),
    )

    # --- Pre-fill verdicts for claims with no evidence (cost saving) ---
    pre_verdicts: dict[str, dict] = {}
    for cid in no_evidence_ids:
        pre_verdicts[cid] = {
            "id": cid,
            "verdict": "unsupported",
            "evidence_quote": "",
            "judge_note": "no matching evidence found in retrieved documents",
        }

    # Claims that need an LLM call
    llm_candidates = [c for c in claims_with_evidence if c["id"] not in no_evidence_ids]

    raw_llm_response = ""
    llm_verdicts: list[dict] = []

    if llm_candidates:
        prompt = _build_judge_prompt(llm_candidates, user_literature_block)
        try:
            raw_llm_response = llm_invoke(prompt)
            parsed = _extract_json_object(raw_llm_response)
            if parsed and isinstance(parsed.get("verdicts"), list):
                llm_verdicts = parsed["verdicts"]
            else:
                logger.warning("faithfulness_judge: LLM response could not be parsed as JSON")
        except Exception as exc:  # noqa: BLE001
            logger.warning("faithfulness_judge: LLM call failed: %s", exc)
            # Mark all remaining as unsupported on LLM failure
            for c in llm_candidates:
                pre_verdicts[c["id"]] = {
                    "id": c["id"],
                    "verdict": "unsupported",
                    "evidence_quote": "",
                    "judge_note": f"judge_llm_error: {str(exc)[:80]}",
                }

    # Index LLM verdicts by claim id (robustly map "1" to "C1")
    llm_verdict_by_id: dict[str, dict] = {}
    for v in llm_verdicts:
        if isinstance(v, dict):
            vid = str(v.get("id", "")).strip().upper()
            if not vid.startswith("C") and vid.isdigit():
                vid = f"C{vid}"
            llm_verdict_by_id[vid] = v

    # --- Assemble final claim list in original order ---
    final_claims: list[dict] = []
    all_verdicts_for_score: list[dict] = []

    for claim in claims:
        cid = claim["id"]
        if cid in pre_verdicts:
            verdict_rec = pre_verdicts[cid]
        elif cid in llm_verdict_by_id:
            verdict_rec = llm_verdict_by_id[cid]
        else:
            # LLM omitted this claim — treat as unsupported
            verdict_rec = {
                "id": cid,
                "verdict": "unsupported",
                "evidence_quote": "",
                "judge_note": "not returned by judge LLM",
            }

        claim_result = {
            "id": cid,
            "summary": claim["summary"],
            "source_file": claim["source_file"],
            "lines": claim["lines"],
            "verdict": str(verdict_rec.get("verdict", "unsupported")).lower(),
            "evidence_quote": str(verdict_rec.get("evidence_quote", "")),
            "judge_note": str(verdict_rec.get("judge_note", "")),
        }
        final_claims.append(claim_result)
        all_verdicts_for_score.append(claim_result)

    score, label = _aggregate(all_verdicts_for_score)

    code_claims_total = 0
    code_claims_supported = 0
    literature_claims_total = 0
    literature_claims_supported = 0

    for c in final_claims:
        src = c.get("source_file", "").strip().lower()
        is_lit = (src == "user literature")
        verdict = c.get("verdict", "")
        if is_lit:
            literature_claims_total += 1
            if verdict == "supported":
                literature_claims_supported += 1
        else:
            code_claims_total += 1
            if verdict == "supported":
                code_claims_supported += 1

    result = {
        "score": score,
        "label": label,
        "claim_count": len(final_claims),
        "code_claims_total": code_claims_total,
        "code_claims_supported": code_claims_supported,
        "literature_claims_total": literature_claims_total,
        "literature_claims_supported": literature_claims_supported,
        "claims": final_claims,
        "raw_llm_response": raw_llm_response,
    }

    _save_cache(key, result)
    logger.info(
        "faithfulness_judge: section score=%.3f (%s) over %d claims",
        score,
        label,
        len(final_claims),
    )
    return result
