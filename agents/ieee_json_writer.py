"""
IEEE tam makale icin Writer: yalnizca JSON uretir; Word export ayri modulde yapilir.
"""

from __future__ import annotations

import logging
from typing import Callable, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _brace_escape(s: str) -> str:
    """Prompt icindeki suslu parantezleri korumak icin."""
    return (s or "").replace("{", "{{").replace("}", "}}")

# Writer bu metne uyarak TEK bir JSON nesnesi dondurur (markdown yok).
IEEE_JSON_ONLY_PROMPT = """You are the IEEE conference paper Writer. Output ONLY one valid JSON object (no markdown fences, no commentary before or after).

REPOSITORY (cite as [1] in prose fields when appropriate): {repository_url}

CONTEXT — repository evidence (only source for implementation claims; NEVER put file paths, line numbers, or code pointers in any JSON string):
{context_blocks}

{operator_block}
{literature_block}

MANDATORY STRUCTURE — the "sections" array MUST contain exactly 5 objects in this order:
1. Introduction
2. Related Work
3. System Architecture (or methodology / system design — pick the best title for the evidence)
4. Evaluation (or results / discussion — pick the best title)
5. Conclusion

Each section object keys: "heading" (string), "body" (string), "subsections" (array).
Each subsection: "heading", "body", optional "subsubsections" array of {{"heading","body"}} objects.

If CONTEXT does not support a section or subsection, set its body to exactly:
[Insufficient evidence — section requires more context]

IEEE TEXT RULES (inside JSON string values):
- Citations: [1], [2], … only; sentence punctuation AFTER the bracket: "... shown in [2]."
- No "Ref. [3]" mid-sentence; use [3] alone.
- Abstract: ONE paragraph, plain English only — NO math, NO symbols, NO LaTeX, NO markdown.
- Paper title: NO math or special symbols.
- Heading fields: title text ONLY — do NOT prefix with Roman numerals (I., II.) or letters (A., B.); template styles add numbering.
- "data" is plural. "et al." has no period after "et". Use em dash meaning only where appropriate in prose; for Abstract/Keywords prefixes the Python layer adds "Abstract—" and "Keywords—".

TOP-LEVEL JSON KEYS (all required):
- "title": string
- "authors": array of strings (each string can use \\n for line breaks: name, affiliation, city, email)
- "abstract": string (single paragraph, plain text)
- "keywords": string (comma-separated keywords)
- "sections": array of exactly 5 section objects as above
- "acknowledgment": string
- "references": array of strings (each one full IEEE-style reference line, starting with [n])

Include at least [1] for the GitHub repository when URL is known.

OUTPUT: Raw JSON only."""


def _format_context_block(docs: List[Document], max_parents: int, max_chars: int) -> str:
    """Writer ile ayni baglam blogu formati."""
    parts: list[str] = []
    for doc in docs[:max_parents]:
        path = doc.metadata.get("file_path", "unknown")
        start = doc.metadata.get("start_line", "?")
        end = doc.metadata.get("end_line", "?")
        text = (doc.page_content or "")[:max_chars]
        parts.append(f"[source={path} lines={start}-{end}]\n{text}\n")
    return "\n".join(parts).strip()


def generate_ieee_paper_json_raw(
    llm_invoke: Callable[[str], str],
    *,
    parent_documents: List[Document],
    repository_url: str,
    operator_addendum: str = "",
    user_literature_block: str = "",
    max_parents: int = 10,
    max_chars_per_parent: int = 6000,
) -> str:
    """
    Retrieval parent parcalarindan IEEE tam makale JSON metni uretir (parse edilmemis ham cevap).
    """
    ctx = _format_context_block(parent_documents, max_parents, max_chars_per_parent)
    if not ctx.strip():
        raise ValueError("CONTEXT bos: once retrieval calistirin.")

    op = (operator_addendum or "").strip()
    op_block = f"OPERATOR NOTES:\n{op}\n" if op else "OPERATOR NOTES:\n(none)\n"
    lit = (user_literature_block or "").strip()
    lit_block = f"USER LITERATURE (optional):\n{lit}\n" if lit else "USER LITERATURE:\n(none)\n"

    prompt = IEEE_JSON_ONLY_PROMPT.format(
        repository_url=_brace_escape(repository_url or "n/a"),
        context_blocks=_brace_escape(ctx),
        operator_block=_brace_escape(op_block),
        literature_block=_brace_escape(lit_block),
    )
    logger.info("IEEE JSON Writer LLM cagrisi basliyor")
    return llm_invoke(prompt).strip()
