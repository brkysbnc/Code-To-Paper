"""
DiagramPlanner: single LLM call that analyses the repository context
and decides which diagrams (context, class, er) are worth generating.
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

AVAILABLE_DIAGRAMS = ("context", "class", "er")

_PLANNER_PROMPT = """You are a software architecture expert reviewing a code repository.
Based on the repository context below, decide which of these three diagrams would be
most valuable to include in an IEEE academic paper about this project:

- "context": A high-level context diagram showing the system and its external dependencies.
  Include if: the system has multiple external integrations (APIs, databases, UIs).
- "class": A class diagram showing main classes and their relationships.
  Include if: the codebase has clear OOP structure with multiple interacting classes.
- "er": An entity-relationship diagram showing data entities and their relationships.
  Include if: the system has any of these: persistent storage, vector databases, JSON schemas defining data structure, configuration models, data pipeline stages with inputs/outputs, or multiple distinct data entities that interact with each other.

Repository context:
{repo_context}

Respond with ONLY a JSON array containing the diagram types to generate.
Example responses:
["context", "class", "er"]
["context", "er"]
["class", "er"]
["context"]

Rules:
- Include at least 1 diagram always.
- Include at most 3 diagrams.
- Only use these exact strings: "context", "class", "er"
- No explanation, no markdown, just the JSON array.
"""

def _parse_planner_response(raw: str) -> list[str]:
    text = (raw or "").strip()
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            valid = [d for d in parsed if d in AVAILABLE_DIAGRAMS]
            if valid:
                return valid
    except (json.JSONDecodeError, TypeError):
        pass
    found = [d for d in AVAILABLE_DIAGRAMS if d in text.lower()]
    if found:
        logger.warning("DiagramPlanner: fallback keyword extraction → %s", found)
        return found
    logger.warning("DiagramPlanner: could not parse response, defaulting to all diagrams")
    return list(AVAILABLE_DIAGRAMS)

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
        return list(AVAILABLE_DIAGRAMS)
    if mode == "manual":
        valid = [d for d in (manual_selection or []) if d in AVAILABLE_DIAGRAMS]
        return valid if valid else list(AVAILABLE_DIAGRAMS)
    # mode == "llm"
    try:
        context_snippet = (repo_context or "")[:6000]
        prompt = _PLANNER_PROMPT.format(repo_context=context_snippet)
        raw = llm_invoke_func(prompt)
        result = _parse_planner_response(raw)
        logger.info("DiagramPlanner: LLM selected diagrams → %s", result)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("DiagramPlanner: LLM call failed: %s — defaulting to all", exc)
        return list(AVAILABLE_DIAGRAMS)
