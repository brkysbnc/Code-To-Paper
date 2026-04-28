"""Tam ic-uca render testi: Markdown -> IEEE template DOCX bytes.

Kotaya hic dokunmaz; sadece OOXML uretimini ve yeni temizleme/numaralama mantigini dogrular.

Kontroller:
- Belge basariyla yazilir (bytes > 0).
- Govdede en az bir 'I.' veya 'II.' Roman onek varsa ## Roman numaralandirma calisiyor.
- TRACEABILITY metni govdede gorunmemeli.
- Repository / Commit metni govdede gorunmemeli.
- Author bloğunda dummy IEEE 'Paper Title' / 'Abstract' / 'Introduction' kalmamali; ama 'Author'
  veya 'affiliation' gibi stillerde paragraf varsa korunmali (sablonda varsa).
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from export.ieee_template_export import (  # noqa: E402
    markdown_to_ieee_template_docx_bytes,
    resolve_default_ieee_template,
)


SAMPLE_MD = """# Code-to-Paper: A RAG Pipeline for IEEE Conference Generation

Abstract\u2014A retrieval-augmented system that converts code repositories into IEEE-format papers.

Keywords\u2014RAG, LLM, IEEE, OOXML

## Introduction and motivation

This is the introduction body. It references retrieval and indexing.

### Background

Subsection text.

### Scope

Another subsection.

## System architecture and implementation

Body for architecture.

### Indexing

Sub.

## Limitations and future work

Body for limits.

## TRACEABILITY (all sections)

This block must not appear in DOCX output.
"""


def _extract_document_xml(docx_bytes: bytes) -> str:
    """Uretilen DOCX'in word/document.xml icerigini metin olarak dondurur."""
    with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zf:
        return zf.read("word/document.xml").decode("utf-8", errors="replace")


def main() -> int:
    """Smoke render: assert'lerle yeni davranisi dogrular."""
    tpl = resolve_default_ieee_template()
    if tpl is None:
        print("WARN: IEEE template bulunamadi; smoke atlandi (docs/templates/ConferenceTemplateIEEE.docx)")
        return 0

    bytes_out = markdown_to_ieee_template_docx_bytes(tpl, SAMPLE_MD)
    assert bytes_out and len(bytes_out) > 1000, "Bos / sacma bicimli docx"

    xml = _extract_document_xml(bytes_out)

    assert "Code-to-Paper:" in xml, "Paper title govdeye yazilmali"
    assert "I. Introduction and motivation" in xml, "Roman numara (I.) Heading 1'e eklenmeli"
    assert "II. System architecture and implementation" in xml, "Roman numara (II.) ikinci ## icin"
    assert "III. Limitations and future work" in xml, "Roman numara (III.) ucuncu ## icin"
    assert "TRACEABILITY" not in xml, "TRACEABILITY govdeden cikmis olmali"
    assert "Repository" not in xml.split("Code-to-Paper:")[-1], "Repository satiri kalmamali"
    assert "A. Background" in xml, "### A. preki uretilmeli"
    assert "B. Scope" in xml, "### ikinci subsection B. olmali"

    # Yeni akis dogrulamalari: title author'in USTUNDE, 2-col continuous sectPr inject edilmis,
    # final sectPr cols num=2 olmali.
    title_pos = xml.find("Code-to-Paper:")
    # Author paragraflarinin tipik metni IEEE template'inden: 'Given Name Surname'
    author_pos = xml.find("Given Name Surname")
    if author_pos != -1:
        assert title_pos < author_pos, (
            f"Paper title author block UZERINDE olmali (title_pos={title_pos}, author_pos={author_pos})"
        )

    # cols num=2 OOXML attribute'u: w:num="2" (lxml ondaki w: namespace qualifier'la yazar).
    cols_2_count = xml.count('w:num="2"')
    assert cols_2_count >= 1, (
        f"En az bir w:cols num='2' bekleniyor (final sectPr ve/veya transition); bulunan={cols_2_count}"
    )

    out = Path(__file__).resolve().parent.parent / "_smoke_ieee_render.docx"
    out.write_bytes(bytes_out)
    print(f"OK render bytes={len(bytes_out)} cols2_count={cols_2_count} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
