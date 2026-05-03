"""
Integration test for export/diagram_renderer.py

Usage:
    python test_diagram_renderer.py

Requirements:
    - .env file in the project root with GOOGLE_API_KEY set
    - pip install langchain-google-genai python-dotenv

The test:
    1. Loads environment variables from .env
    2. Creates a ChatGoogleGenerativeAI instance (gemini-2.0-flash-lite)
    3. Calls generate_all_diagrams() with a representative repo context
    4. Asserts that 3 PNG files exist under data/diagrams/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — show INFO+ to stdout so progress is visible during the test run
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_diagram_renderer")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load .env
    # ------------------------------------------------------------------
    try:
        from dotenv import load_dotenv  # type: ignore[import]
    except ImportError:
        logger.error("python-dotenv is not installed. Run: pip install python-dotenv")
        sys.exit(1)

    env_path = Path(__file__).parent / ".env"
    if not env_path.is_file():
        logger.error(".env file not found at %s", env_path)
        sys.exit(1)

    loaded = load_dotenv(dotenv_path=env_path, override=True)
    logger.info(".env loaded: %s (path=%s)", loaded, env_path)

    # ------------------------------------------------------------------
    # 2. Build LLM + llm_invoke_func
    # ------------------------------------------------------------------
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]
    except ImportError:
        logger.error(
            "langchain-google-genai is not installed. Run: pip install langchain-google-genai"
        )
        sys.exit(1)

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.2,
            # GOOGLE_API_KEY is read automatically from the environment
        )
        logger.info("LLM created: %s", llm.model)
    except Exception as exc:
        logger.error("Failed to create ChatGoogleGenerativeAI: %s", exc)
        sys.exit(1)

    def llm_invoke_func(prompt: str) -> str:
        """Thin wrapper: sends prompt to Gemini and returns the text content."""
        response = llm.invoke(prompt)
        # LangChain AIMessage → string
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    # ------------------------------------------------------------------
    # 3. Build a representative repo context (Code-To-Paper itself)
    # ------------------------------------------------------------------
    repo_context = """\
Repository: Code-To-Paper
Language: Python

Directory structure (top-level):
  agents/          — LLM-based writing, metadata, faithfulness agents
  export/          — Word/IEEE export helpers
  orchestration/   — Pipeline orchestration and section planning
  data/            — Chroma vector store, faithfulness cache, source files
  retriever.py     — RAG retrieval (ChromaDB + LangChain)
  main.py          — Streamlit web UI + pipeline entry point
  fetcher.py       — GitHub repository cloner
  github_handler.py — GitHub API integration

Key modules:
  agents/writer.py           — AcademicWriter: generates IEEE-style paper sections
  agents/metadata_writer.py  — Generates IEEE metadata (title, abstract, keywords, authors)
  agents/faithfulness_judge.py — LLM-as-a-judge: verifies claims against source code
  orchestration/paper_blueprint.py — Plans paper sections from repo analysis
  orchestration/section_pipeline.py — Runs writer + judge per section
  retriever.py               — Indexes repo files into ChromaDB; performs RAG retrieval
  export/ieee_template_export.py — Renders Markdown to IEEE .docx template
  export/word_export.py      — Generic Word export helper

Data flow:
  GitHub URL → fetcher → retriever (index) → blueprint → section_pipeline (write + judge)
  → ieee_template_export → .docx output

Key classes / callables:
  AcademicWriter (agents/writer.py)
  MetadataWriter (agents/metadata_writer.py)
  FaithfulnessJudge / judge_section_faithfulness (agents/faithfulness_judge.py)
  PaperBlueprint (orchestration/paper_blueprint.py)
  run_section_pipeline (orchestration/section_pipeline.py)
  CodeRetriever (retriever.py)
"""

    # ------------------------------------------------------------------
    # 4. Call generate_all_diagrams()
    # ------------------------------------------------------------------
    try:
        from export.diagram_renderer import generate_all_diagrams  # type: ignore[import]
    except ImportError as exc:
        logger.error("Could not import diagram_renderer: %s", exc)
        sys.exit(1)

    logger.info("Calling generate_all_diagrams() …")
    results = generate_all_diagrams(repo_context, llm_invoke_func)

    # ------------------------------------------------------------------
    # 5. Verify 3 PNG files exist in data/diagrams/
    # ------------------------------------------------------------------
    diagrams_dir = Path("data") / "diagrams"
    expected_files = [
        diagrams_dir / "context_diagram.png",
        diagrams_dir / "class_diagram.png",
        diagrams_dir / "er_diagram.png",
    ]

    passed = 0
    failed = 0
    for expected in expected_files:
        if expected.exists() and expected.stat().st_size > 0:
            logger.info("  ✓ FOUND  %s (%d bytes)", expected, expected.stat().st_size)
            passed += 1
        else:
            logger.error("  ✗ MISSING %s", expected)
            failed += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("Test results: %d passed, %d failed (out of 3)", passed, failed)
    logger.info("generate_all_diagrams returned keys: %s", list(results.keys()))
    logger.info("=" * 60)

    if failed > 0:
        logger.error("FAIL: %d PNG file(s) were not produced.", failed)
        sys.exit(1)
    else:
        logger.info("PASS: all 3 PNG files successfully created in %s", diagrams_dir)


if __name__ == "__main__":
    main()
