"""
DiagramPlanner: tek desteklenen cikti turu "context" (sistem baglam diyagrami).
Modlar: LLM kaynakta context eklenip eklenmeyecegine karar verir; class/ER kaldırıldı.

Public API
----------
    plan_diagrams(repo_context, llm_invoke_func, mode, manual_selection) -> list[str]
"""
from __future__ import annotations
import json
import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

# Yalnizca context diyagrami (Markdown [DIAGRAM:context] + PNG gomme).
AVAILABLE_DIAGRAMS = ("context",)

_PLANNER_PROMPT = """You are a software architecture expert reviewing a code repository.
We only support ONE optional diagram type: a high-level **context** diagram (main components
and external dependencies — flowchart style).

Repository context:
{repo_context}

Respond with ONLY a JSON array:
- ["context"] if a context diagram would help readers understand the system boundary and main parts.
- [] if the context is too thin or a diagram would not add value.

Rules:
- At most one entry; only the exact string "context" is allowed.
- No explanation, no markdown, just the JSON array.
"""

def _parse_planner_response(raw: str) -> list[str]:
    text = (raw or "").strip()
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            valid = [d for d in parsed if d in AVAILABLE_DIAGRAMS]
            return valid
    except (json.JSONDecodeError, TypeError):
        pass
    if "context" in text.lower():
        logger.warning("DiagramPlanner: fallback keyword extraction → ['context']")
        return ["context"]
    logger.warning("DiagramPlanner: could not parse response, defaulting to context diagram")
    return ["context"]

def plan_diagrams(
    repo_context: str,
    llm_invoke_func: Callable[[str], str],
    mode: str = "llm",
    manual_selection: list[str] | None = None,
) -> list[str]:
    """
    Decides which diagrams to generate.
    mode: "llm" | "manual" | "all" | "none"
    """
    if mode == "none":
        return []
    if mode == "all":
        return ["context"]
    if mode == "manual":
        # Kullanicinin Streamlit'te sectigi turler (yalnizca 'context'); bos = diyagram yok.
        return [d for d in (manual_selection or []) if d in AVAILABLE_DIAGRAMS]
    # mode == "llm"
    try:
        context_snippet = (repo_context or "")[:6000]
        prompt = _PLANNER_PROMPT.format(repo_context=context_snippet)
        raw = llm_invoke_func(prompt)
        result = _parse_planner_response(raw)
        logger.info("DiagramPlanner: LLM selected diagrams → %s", result)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("DiagramPlanner: LLM call failed: %s — defaulting to context", exc)
        return ["context"]
