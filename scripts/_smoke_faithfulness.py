"""
Smoke test for agents/faithfulness_judge.py.

Usage:
    python scripts/_smoke_faithfulness.py

Verifies that:
1. _parse_traceability_rows correctly extracts claim rows.
2. _find_evidence matches file_path + line overlap.
3. judge_section_faithfulness returns the correct dict shape.
4. Cache hit produces zero LLM calls on second invocation.

Does NOT make real LLM calls — uses a mock llm_invoke.
Delete this script after maintainer review.
"""

from __future__ import annotations

import json
import sys
import tempfile
import shutil
from pathlib import Path

# Ensure repo root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

# ------------------------------------------------------------------
# Patch cache dir to a temp directory so tests don't pollute data/
# ------------------------------------------------------------------
import agents.faithfulness_judge as fj  # noqa: E402

_tmp_cache = tempfile.mkdtemp(prefix="fj_smoke_cache_")
fj._CACHE_DIR = Path(_tmp_cache)  # type: ignore[attr-defined]

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

SAMPLE_TRACEABILITY = """\
TRACEABILITY:
| Claim ID | Claim summary                              | Source file        | Lines   | Notes                    |
| -------- | ------------------------------------------ | ------------------ | ------- | ------------------------ |
| C1       | Pipeline indexes parent and child chunks   | retriever.py       | 10-50   | implementation evidenced |
| C2       | Adaptive threshold reduces empty results   | section_pipeline.py| 113-148 | retry on empty docs      |
"""

DOC_RETRIEVER = Document(
    page_content="def index_repository_files(retriever, paths, *, repo_root, repo_url, commit_hash): ...",
    metadata={"file_path": "retriever.py", "start_line": 10, "end_line": 60},
)
DOC_SECTION_PIPELINE = Document(
    page_content="if float(similarity_threshold) > 0.15: new_threshold = max(0.15, float(similarity_threshold) - 0.1)",
    metadata={"file_path": "section_pipeline.py", "start_line": 113, "end_line": 148},
)
PARENT_DOCS = [DOC_RETRIEVER, DOC_SECTION_PIPELINE]

# ------------------------------------------------------------------
# Mock LLM
# ------------------------------------------------------------------
_call_count = 0


def _mock_llm_invoke(prompt: str) -> str:
    global _call_count
    _call_count += 1
    return json.dumps(
        {
            "verdicts": [
                {
                    "id": "C1",
                    "verdict": "supported",
                    "evidence_quote": "def index_repository_files",
                    "judge_note": "verbatim match",
                },
                {
                    "id": "C2",
                    "verdict": "partial",
                    "evidence_quote": "new_threshold = max(0.15",
                    "judge_note": "paraphrase only",
                },
            ]
        }
    )


# ------------------------------------------------------------------
# Test 1: _parse_traceability_rows
# ------------------------------------------------------------------
def test_parse_rows() -> None:
    rows = fj._parse_traceability_rows(SAMPLE_TRACEABILITY)
    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
    assert rows[0]["id"] == "C1"
    assert rows[1]["id"] == "C2"
    assert "retriever.py" in rows[0]["source_file"]
    print("  [PASS] _parse_traceability_rows")


# ------------------------------------------------------------------
# Test 2: _find_evidence
# ------------------------------------------------------------------
def test_find_evidence() -> None:
    rows = fj._parse_traceability_rows(SAMPLE_TRACEABILITY)
    ev1 = fj._find_evidence(rows[0], PARENT_DOCS)
    assert "index_repository_files" in ev1, f"Expected retriever content, got: {ev1[:200]}"
    ev2 = fj._find_evidence(rows[1], PARENT_DOCS)
    assert "similarity_threshold" in ev2, f"Expected pipeline content, got: {ev2[:200]}"
    print("  [PASS] _find_evidence")


# ------------------------------------------------------------------
# Test 3: judge_section_faithfulness — result shape
# ------------------------------------------------------------------
def test_judge_shape() -> None:
    global _call_count
    _call_count = 0

    result = fj.judge_section_faithfulness(
        writer_text="sample writer text for test",
        writer_traceability=SAMPLE_TRACEABILITY,
        parent_documents=PARENT_DOCS,
        llm_invoke=_mock_llm_invoke,
    )

    assert isinstance(result["score"], float), "score must be float"
    assert result["label"] in ("high", "medium", "low"), "label must be high/medium/low"
    assert isinstance(result["claim_count"], int)
    assert isinstance(result["claims"], list)
    assert len(result["claims"]) == 2

    for claim in result["claims"]:
        for key in ("id", "summary", "source_file", "lines", "verdict", "evidence_quote", "judge_note"):
            assert key in claim, f"Missing key: {key}"
        assert claim["verdict"] in ("supported", "partial", "unsupported")

    assert _call_count == 1, f"Expected 1 LLM call, got {_call_count}"
    print(f"  [PASS] judge shape — score={result['score']}, label={result['label']}, claims={result['claim_count']}")
    return result


# ------------------------------------------------------------------
# Test 4: Cache hit — zero LLM calls on second invocation
# ------------------------------------------------------------------
def test_cache_hit(first_result: dict) -> None:
    global _call_count
    _call_count = 0

    result2 = fj.judge_section_faithfulness(
        writer_text="sample writer text for test",
        writer_traceability=SAMPLE_TRACEABILITY,
        parent_documents=PARENT_DOCS,
        llm_invoke=_mock_llm_invoke,
    )

    assert _call_count == 0, f"Expected 0 LLM calls on cache hit, got {_call_count}"
    assert result2["score"] == first_result["score"], "Cached score must match original"
    print("  [PASS] Cache hit — zero LLM calls on second invocation")


# ------------------------------------------------------------------
# Test 5: Empty traceability returns zero claims gracefully
# ------------------------------------------------------------------
def test_empty_traceability() -> None:
    result = fj.judge_section_faithfulness(
        writer_text="some body text",
        writer_traceability="",
        parent_documents=PARENT_DOCS,
        llm_invoke=_mock_llm_invoke,
    )
    assert result["claim_count"] == 0
    assert result["score"] == 0.0
    print("  [PASS] Empty traceability → 0 claims, score=0.0")


# ------------------------------------------------------------------
# Run all tests
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Running faithfulness_judge smoke tests...\n")
    try:
        test_parse_rows()
        test_find_evidence()
        first = test_judge_shape()
        test_cache_hit(first)
        test_empty_traceability()
        print("\n✅ All smoke tests PASSED")
    finally:
        shutil.rmtree(_tmp_cache, ignore_errors=True)
