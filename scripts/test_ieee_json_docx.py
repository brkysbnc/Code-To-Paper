"""IEEE JSON -> Word hizli test."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.ieee_json_schema import normalize_ieee_paper_content, parse_ieee_paper_json
from export.ieee_document_from_json import build_ieee_document_bytes
from export.ieee_template_export import resolve_default_ieee_template

sample = {
    "title": "Test Paper Title",
    "authors": ["Author One\nUniversity\nCity\na@b.com"],
    "abstract": "This is a plain abstract without math.",
    "keywords": "rag, documentation, llm",
    "sections": [
        {"heading": "Introduction", "body": "Intro text with [1].", "subsections": []},
        {"heading": "Related Work", "body": "", "subsections": [{"heading": "Prior art", "body": "Related [2].", "subsubsections": []}]},
        {"heading": "System Architecture", "body": "", "subsections": []},
        {"heading": "Evaluation", "body": "", "subsections": []},
        {"heading": "Conclusion", "body": "Done.", "subsections": []},
    ],
    "acknowledgment": "Thanks.",
    "references": ['[1] Repo, "GitHub," 2026. [Online]. Available: https://github.com/x/y'],
}

def main() -> None:
    raw = json.dumps(sample)
    data = parse_ieee_paper_json(raw)
    data = normalize_ieee_paper_content(data, repository_url="https://github.com/x/y.git")
    tpl = resolve_default_ieee_template()
    assert tpl is not None
    b = build_ieee_document_bytes(tpl, data)
    out = ROOT / "_test_ieee_json_out.docx"
    out.write_bytes(b)
    print("OK", out, len(b))


if __name__ == "__main__":
    main()
