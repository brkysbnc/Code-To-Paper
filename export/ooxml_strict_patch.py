"""
Bazi IEEE sablonlari ISO/IEC 'Strict OOXML' (purl.oclc.org) ad alanlari kullanir;
python-docx paketi Office acilis URI'lerini bekler. Tum paket icinde bilinen eslemeleri donusturur.
"""

from __future__ import annotations

import zipfile
from io import BytesIO

# Uzun eslemeler once (prefix carpismasini azaltmak icin)
PURL_TO_MS: tuple[tuple[str, str], ...] = (
    (
        "http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing",
        "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    ),
    (
        "http://purl.oclc.org/ooxml/wordprocessingml/main",
        "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    ),
    (
        "http://purl.oclc.org/ooxml/drawingml/main",
        "http://schemas.openxmlformats.org/drawingml/2006/main",
    ),
    (
        "http://purl.oclc.org/ooxml/officeDocument/math",
        "http://schemas.openxmlformats.org/officeDocument/2006/math",
    ),
    (
        "http://purl.oclc.org/ooxml/officeDocument/relationships/",
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/",
    ),
)


def patch_strict_ooxml_to_opc(docx_bytes: bytes) -> bytes:
    """
    .docx zip icindeki xml/rels dosyalarinda purl tabanli URI'leri OPC esdegerine cevirir.
    """
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(docx_bytes), "r") as zin:
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                raw = zin.read(info.filename)
                if info.filename.endswith((".xml", ".rels")):
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        zout.writestr(info, raw)
                        continue
                    for purl, ms in PURL_TO_MS:
                        text = text.replace(purl, ms)
                    zout.writestr(info, text.encode("utf-8"))
                else:
                    zout.writestr(info, raw)
    return out.getvalue()
