"""Hizli sagduyu testi: metadata writer JSON parse + ieee_template_export Roman/temizleme.

CI degil, lokal smoke. Harici API cagirmaz; LLM yerine deterministik bir lambda enjekte edilir.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.metadata_writer import MetadataWriter  # noqa: E402
from export import ieee_template_export as ie  # noqa: E402
from orchestration import paper_blueprint  # noqa: E402
from orchestration import section_pipeline  # noqa: E402


def main() -> int:
    """Roma rakami, MetadataWriter parse, blueprint birlestirme akislarinda hizli check."""
    assert ie._to_roman_numeral(1) == "I"
    assert ie._to_roman_numeral(4) == "IV"
    assert ie._to_roman_numeral(9) == "IX"
    assert ie._to_roman_numeral(40) == "XL"
    assert ie._to_roman_numeral(0) == ""
    print("roman ok")

    fake_llm_response = '```json\n{"title": "Code-to-Paper RAG", "abstract": "An IEEE pipeline.", "keywords": "RAG, IEEE, OOXML"}\n```'
    md = MetadataWriter(lambda _p: fake_llm_response).generate(combined_body="hello world", repo_url="x")
    assert md["title"] == "Code-to-Paper RAG"
    assert md["abstract"] == "An IEEE pipeline."
    assert md["keywords"] == "RAG, IEEE, OOXML"
    print("metadata parse ok")

    bad = MetadataWriter(lambda _p: "garbage").generate(combined_body="hi", repo_url="x")
    assert bad == {"title": "", "abstract": "", "keywords": ""}
    print("metadata bad ok")

    out = paper_blueprint.combine_paper_markdown(
        repo_url="https://github.com/x/y.git",
        commit_hash="abc123",
        section_results=[
            {"section_title": "Introduction", "writer_text": "Body of intro.", "writer_metadata": {}},
        ],
        paper_title="My Title",
        abstract_text="A short abstract.",
        keywords_text="rag, ieee",
    )
    assert "Repository" not in out, "Repository satiri hala blueprint'te!"
    assert "Commit" not in out, "Commit satiri hala blueprint'te!"
    assert "# My Title" in out
    assert "Abstract\u2014A short abstract." in out
    assert "Keywords\u2014rag, ieee" in out
    print("blueprint clean ok")

    print("ALL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
