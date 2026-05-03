"""
Diagram renderer for Code-To-Paper.

Generates three Mermaid diagram types (context, class, er) by prompting Gemini,
then renders each to PNG and saves them under data/diagrams/.

Render pipeline:
  1. Try mermaid-py (requires `npm install -g @mermaid-js/mermaid-cli`)
  2. Fallback: GET https://mermaid.ink/img/<base64_encoded_mermaid>

Public API
----------
    generate_all_diagrams(repo_context, llm_invoke_func) -> dict[str, str]
    generate_diagram(diagram_type, repo_context, llm_invoke_func) -> Optional[str]

Internal helpers
----------------
    _generate_mermaid_code(diagram_type, repo_context, llm_invoke_func) -> str
    _render_mermaid_to_png(mermaid_code, output_path) -> bool
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIAGRAM_OUTPUT_DIR = Path("data") / "diagrams"

DIAGRAM_PATHS: dict[str, str] = {
    "context": str(DIAGRAM_OUTPUT_DIR / "diagram_context.png"),
    "class":   str(DIAGRAM_OUTPUT_DIR / "diagram_class.png"),
    "er":      str(DIAGRAM_OUTPUT_DIR / "diagram_er.png"),
}

_DIAGRAM_TYPES = ("context", "class", "er")

_PROMPTS: dict[str, str] = {
    "context": (
        "You are a software architecture expert. "
        "Given the following repository context, generate a Mermaid flowchart diagram "
        "that shows the main components and their relationships.\n\n"
        "STRICT Rules — violating any rule will cause a render error:\n"
        "- Output ONLY the raw Mermaid code. No markdown fences, no prose, no explanation.\n"
        "- The very first line MUST be exactly: graph TD\n"
        "- Node IDs must be plain alphanumeric strings with no spaces (e.g. Writer, RAGRetriever).\n"
        "- Node labels must use square brackets and contain only plain text: Writer[Academic Writer]\n"
        "- Do NOT use parentheses (), angle brackets <>, curly braces {{}}, or pipes | in labels.\n"
        "- Edge labels (if any) must use simple quoted strings: A -->|calls| B\n"
        "- Do NOT put special characters like colons, slashes, or dots inside node labels.\n"
        "- Keep the diagram focused: 6-10 nodes max.\n\n"
        "Example of correct syntax:\n"
        "graph TD\n"
        "    UI[Streamlit UI]\n"
        "    Fetcher[GitHub Fetcher]\n"
        "    Retriever[RAG Retriever]\n"
        "    Writer[Academic Writer]\n"
        "    Exporter[IEEE Exporter]\n"
        "    UI -->|clone repo| Fetcher\n"
        "    Fetcher --> Retriever\n"
        "    Retriever --> Writer\n"
        "    Writer --> Exporter\n\n"
        "Repository context:\n{repo_context}"
    ),
    "class": (
        "You are a software architecture expert. "
        "Given the following repository context, generate a Mermaid **classDiagram** "
        "that shows the main classes/modules and their relationships.\n\n"
        "Rules:\n"
        "- Output ONLY the raw Mermaid code, no markdown fences, no explanation.\n"
        "- The very first line MUST be exactly: classDiagram\n"
        "- Class names must be plain alphanumeric identifiers (no spaces).\n"
        "- Method signatures should be simple: +generate_section(prompt) str\n"
        "- Keep the diagram focused (4-10 classes max).\n\n"
        "Repository context:\n{repo_context}"
    ),
    "er": (
        "You are a software architecture expert. "
        "Given the following repository context, generate a Mermaid **erDiagram** "
        "that shows the main data entities and their relationships.\n\n"
        "Rules:\n"
        "- Output ONLY the raw Mermaid code, no markdown fences, no explanation.\n"
        "- The very first line MUST be exactly: erDiagram\n"
        "- Entity names must be plain uppercase alphanumeric identifiers (no spaces).\n"
        "- Attribute types must be simple: string, int, bool\n"
        "- Keep the diagram focused (3-8 entities max).\n\n"
        "Repository context:\n{repo_context}"
    ),
}

# Expected first tokens for each diagram type (used for validation and sanitization)
_EXPECTED_STARTS: dict[str, tuple[str, ...]] = {
    "context": ("graph td", "graph lr", "graph rl", "graph tb", "graph bt", "flowchart"),
    "class": ("classdiagram",),
    "er": ("erdiagram",),
}

# Canonical first line that must appear in the output for each type
_ANCHOR_LINES: dict[str, tuple[str, ...]] = {
    "context": ("graph TD", "graph LR", "graph RL", "graph TB", "graph BT", "flowchart TD", "flowchart LR"),
    "class": ("classDiagram",),
    "er": ("erDiagram",),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize_mermaid(diagram_type: str, text: str) -> str:
    """
    Robustly extract valid Mermaid code from raw LLM output.

    Steps:
    1. Strip all ```mermaid / ``` fences (handles nested or multiple fences).
    2. Find the first line that matches the expected diagram-type keyword
       (e.g. "graph TD", "classDiagram", "erDiagram") — discards any LLM
       preamble such as "Here is the diagram:" or explanation text.
    3. Return everything from that anchor line onward, stripped of trailing
       whitespace.

    Returns empty string if no anchor line is found.
    """
    if not text:
        return ""

    # --- Step 1: strip all markdown code fences ---
    # Replace ```mermaid ... ``` and ``` ... ``` blocks with their inner content.
    text = re.sub(
        r"```(?:mermaid)?[ \t]*\r?\n?",  # opening fence
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"```[ \t]*\r?\n?", "", text)  # closing fence
    text = text.strip()

    # --- Step 2: find the anchor line ---
    anchors = _ANCHOR_LINES.get(diagram_type, ())
    lines = text.splitlines()
    anchor_idx: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        for anchor in anchors:
            # Case-insensitive prefix match so "graph td" == "graph TD"
            if stripped.lower().startswith(anchor.lower()):
                # Normalise the anchor line to the canonical casing
                lines[i] = anchor + (stripped[len(anchor):] if len(stripped) > len(anchor) else "")
                anchor_idx = i
                break
        if anchor_idx is not None:
            break

    if anchor_idx is None:
        logger.debug(
            "diagram_renderer._sanitize_mermaid: no anchor found for %r in %r",
            diagram_type,
            text[:200],
        )
        return ""

    # --- Step 3: return from anchor line onward ---
    result = "\n".join(lines[anchor_idx:]).rstrip()
    return result


def _is_valid_mermaid(diagram_type: str, code: str) -> bool:
    """Basic validation: checks that the code starts with the expected keyword."""
    if not code or not code.strip():
        return False
    first_line = code.strip().splitlines()[0].strip().lower()
    expected = _EXPECTED_STARTS.get(diagram_type, ())
    return any(first_line.startswith(exp.lower()) for exp in expected)


def _generate_mermaid_code(
    diagram_type: str,
    repo_context: str,
    llm_invoke_func: Callable[[str], str],
) -> str:
    """
    Calls the LLM with a type-specific prompt and returns sanitized Mermaid code.

    Uses _sanitize_mermaid to strip fences and discard any LLM preamble/explanation.
    Returns empty string on failure or when no valid anchor is found.
    """
    prompt_template = _PROMPTS.get(diagram_type)
    if not prompt_template:
        logger.error("diagram_renderer: unknown diagram_type=%r", diagram_type)
        return ""

    # Truncate repo_context to stay within token limits
    context_snippet = (repo_context or "")[:8000]
    prompt = prompt_template.format(repo_context=context_snippet)

    try:
        raw = llm_invoke_func(prompt)
        code = _sanitize_mermaid(diagram_type, str(raw or ""))
        logger.debug(
            "diagram_renderer: sanitized %d chars for diagram_type=%r — preview: %r",
            len(code),
            diagram_type,
            code[:300],
        )
        return code
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "diagram_renderer: LLM call failed for diagram_type=%r: %s",
            diagram_type,
            exc,
        )
        return ""


def _render_with_mermaid_py(mermaid_code: str, output_path: Path) -> bool:
    """
    Attempts to render using the mermaid-py library (wraps mmdc CLI).
    Returns True on success, False on any failure.
    """
    try:
        import mermaid as md  # type: ignore[import]
        from mermaid.graph import Graph  # type: ignore[import]

        graph = Graph("diagram", mermaid_code)
        render = md.Mermaid(graph)
        # mermaid-py saves to file via to_png()
        render.to_png(str(output_path))
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("diagram_renderer: rendered via mermaid-py → %s", output_path)
            return True
        logger.warning(
            "diagram_renderer: mermaid-py produced empty/missing file at %s", output_path
        )
        return False
    except Exception as exc:  # noqa: BLE001
        logger.debug("diagram_renderer: mermaid-py unavailable or failed: %s", exc)
        return False


def _render_with_mermaid_ink(mermaid_code: str, output_path: Path) -> bool:
    """
    Fallback renderer: encodes Mermaid code as base64 and fetches PNG from mermaid.ink.
    Returns True on success, False on any failure.
    """
    try:
        import urllib.request  # stdlib — no extra dep
        import urllib.error

        encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("ascii")
        url = f"https://mermaid.ink/img/{encoded}"
        logger.debug("diagram_renderer: requesting mermaid.ink → %s", url[:80])

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CodeToPaper/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = resp.read()

        if not data or len(data) < 100:
            logger.warning(
                "diagram_renderer: mermaid.ink returned suspiciously small response (%d bytes)",
                len(data),
            )
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        logger.info(
            "diagram_renderer: rendered via mermaid.ink (%d bytes) → %s",
            len(data),
            output_path,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("diagram_renderer: mermaid.ink fallback failed: %s", exc)
        return False


def _render_mermaid_to_png(mermaid_code: str, output_path: Path) -> bool:
    """
    Tries mermaid-py first; falls back to mermaid.ink.
    Returns True if a PNG was successfully written, False otherwise.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Primary: mermaid-py (mmdc)
    if _render_with_mermaid_py(mermaid_code, output_path):
        return True

    # Fallback: mermaid.ink
    logger.info(
        "diagram_renderer: mermaid-py failed; using mermaid.ink fallback for %s",
        output_path.name,
    )
    return _render_with_mermaid_ink(mermaid_code, output_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_diagram(
    diagram_type: str,
    repo_context: str,
    llm_invoke_func: Callable[[str], str],
) -> Optional[str]:
    """
    Generates a single diagram of the given type.

    Parameters
    ----------
    diagram_type : str
        One of "context", "class", "er".
    repo_context : str
        Textual summary of the repository (e.g. file tree + key code snippets).
    llm_invoke_func : Callable[[str], str]
        A callable that accepts a prompt string and returns the LLM response string.

    Returns
    -------
    str
        Absolute path to the saved PNG file, or None on failure.
    """
    logger.info("diagram_renderer: generating diagram_type=%r", diagram_type)

    # Step 1 — Generate Mermaid code
    mermaid_code = _generate_mermaid_code(diagram_type, repo_context, llm_invoke_func)

    if not mermaid_code:
        logger.warning(
            "diagram_renderer: empty Mermaid code returned for diagram_type=%r; skipping",
            diagram_type,
        )
        return None

    # Step 2 — Validate syntax (basic check)
    if not _is_valid_mermaid(diagram_type, mermaid_code):
        logger.warning(
            "diagram_renderer: Gemini returned invalid Mermaid syntax for diagram_type=%r; "
            "first line: %r; returning None",
            diagram_type,
            mermaid_code.strip().splitlines()[0] if mermaid_code.strip() else "",
        )
        return None

    # Step 3 — Render to PNG
    output_path = Path(DIAGRAM_PATHS[diagram_type])
    success = _render_mermaid_to_png(mermaid_code, output_path)

    if not success:
        logger.error(
            "diagram_renderer: all render methods failed for diagram_type=%r", diagram_type
        )
        return None

    return str(output_path.resolve())


def generate_all_diagrams(
    repo_context: str,
    llm_invoke_func: Callable[[str], str],
) -> dict[str, str]:
    """
    Generates context, class, and er diagrams and saves them as PNG files.

    Parameters
    ----------
    repo_context : str
        Textual summary of the repository.
    llm_invoke_func : Callable[[str], str]
        LLM invocation callable (prompt → response string).

    Returns
    -------
    dict[str, str]
        Mapping of diagram_type → absolute PNG path for each successfully rendered diagram.
        Types that failed are omitted from the dict.
    """
    DIAGRAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    for diagram_type in _DIAGRAM_TYPES:
        try:
            path = generate_diagram(diagram_type, repo_context, llm_invoke_func)
            if path:
                results[diagram_type] = path
                logger.info(
                    "diagram_renderer: ✓ %s diagram saved → %s", diagram_type, path
                )
            else:
                logger.warning(
                    "diagram_renderer: ✗ %s diagram was not generated", diagram_type
                )
        except Exception as exc:  # noqa: BLE001 — pipeline must never crash
            logger.error(
                "diagram_renderer: unexpected error for diagram_type=%r: %s",
                diagram_type,
                exc,
                exc_info=True,
            )

    logger.info(
        "diagram_renderer: generate_all_diagrams complete — %d/%d diagrams produced",
        len(results),
        len(_DIAGRAM_TYPES),
    )
    return results
