"""DOCX body sıralamasını ve sectPr/cols durumunu hızlıca analiz eder."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def main(path: str) -> int:
    """Verilen .docx dosyasinin word/document.xml body sirasini ozetler."""
    p = Path(path)
    with zipfile.ZipFile(p, "r") as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="replace")

    # XML'in tum namespace bildirimlerini koruyabilmek icin document parse edip body'i tarayacagiz.
    root_doc = ET.fromstring(xml)
    root = root_doc.find(W_NS + "body")
    if root is None:
        raise SystemExit("body bulunamadi")

    print(f"=== {p.name} body ozet ===")
    for idx, child in enumerate(list(root)):
        tag = child.tag.replace(W_NS, "w:")
        info = ""

        if tag == "w:p":
            ppr = child.find(W_NS + "pPr")
            style = ""
            sect_info = ""
            if ppr is not None:
                ps = ppr.find(W_NS + "pStyle")
                if ps is not None:
                    style = ps.get(W_NS + "val") or ""
                sp = ppr.find(W_NS + "sectPr")
                if sp is not None:
                    cols = sp.find(W_NS + "cols")
                    n = cols.get(W_NS + "num") if cols is not None else None
                    s = cols.get(W_NS + "space") if cols is not None else None
                    type_el = sp.find(W_NS + "type")
                    type_v = type_el.get(W_NS + "val") if type_el is not None else None
                    sect_info = f" [INNER SECTPR type={type_v} cols.num={n} cols.space={s}]"
            text_parts = [t.text or "" for t in child.iter(W_NS + "t")]
            text = "".join(text_parts)
            info = f"style='{style}' text='{text[:90]}'{sect_info}"
        elif tag == "w:sectPr":
            cols = child.find(W_NS + "cols")
            type_el = child.find(W_NS + "type")
            n = cols.get(W_NS + "num") if cols is not None else None
            s = cols.get(W_NS + "space") if cols is not None else None
            type_v = type_el.get(W_NS + "val") if type_el is not None else None
            info = f"FINAL SECTPR type={type_v} cols.num={n} cols.space={s}"
        elif tag == "w:tbl":
            info = "(table)"
        else:
            info = f"(other: {tag})"

        print(f"  [{idx:02d}] {tag:10s} {info}")

    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "outputs/full-paper-draft.docx"
    sys.exit(main(target))
