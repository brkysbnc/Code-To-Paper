"""
RAG destekli akademik baslik / abstract uretimi ve abstract metninden deterministik keyword cikarimi.

Akis (tek LLM cagrisi):
1) MetadataWriter.generate combined_body (yazilmis bolumler) + rag_documents (ek repo kaniti) + repo_url alir.
2) Body 18000, RAG context 12000 char ile kesilir; LLM'e str.replace ile enjekte edilir
   (str.format kullanilmaz — bolum metinlerindeki literal { } kod fence'lerini bozmamak icin).
3) LLM yalnizca {"title", "abstract"} JSON dondurur; abstract 250 kelimede sertce kesilir.
4) extract_keywords_from_abstract regex/akronim/bigram zincirini calistirip keywords ⊆ abstract
   garantisini Python tarafinda saglar (LLM keyword uretmez).

Free-tier: pipeline basina 1 LLM cagrisi (mevcut MetadataWriter ile ayni); +3 embedContent cagrisi
abstract icin heuristic planner sorgulariyla cagri tarafinda yapilir (bu dosya disinda).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# LLM prompt sablonu: placeholder'lar replace ile doldurulur (format() degil; { } kac zorunlu degil).
_PROMPT_TEMPLATE = """You are an expert academic technical writer for IEEE conference papers.

You are given:
- REPO URL: {repo_url}
- BODY: combined draft of the paper's body sections (already RAG-grounded).
- CONTEXT: additional raw evidence retrieved directly from the repository.

Produce TWO pieces of metadata in JSON format:

1) TITLE: A concise, descriptive academic title (8-15 words).
   - Reflect the paper's actual technical contribution.
   - Use specific technical terms found in BODY or CONTEXT.

2) ABSTRACT: A single paragraph in IEEE style.
   - STRICT MAXIMUM 250 WORDS.
   - Use ONLY facts present in BODY or CONTEXT. Do NOT invent numbers or benchmarks.
   - Include domain-specific technical terms (e.g. RAG, ChromaDB, OOXML).

REPO URL:
{repo_url}

REPOSITORY CONTEXT:
---
{rag_context}
---

BODY:
---
{combined_body}
---

OUTPUT FORMAT — return EXACTLY a JSON object with this shape, nothing before or after:
{"title": "<string>", "abstract": "<string>"}
"""


# Acronym pass'inde keyword'e DUSURMEMEK istedigimiz cok yaygin 2-5 harfli buyuk-harf belirteçleri.
# Filtre kucuk-harf karsilastirmasi yapar; entry'leri lowercase tutuyoruz.
_ACRONYM_BLACKLIST: frozenset[str] = frozenset({"the", "and", "for", "but", "not", "all", "any"})

# Tum bigram filtreleri icin yaygın akademik dolgu sozcukleri seti.
_BIGRAM_STOP_WORDS: frozenset[str] = frozenset(
    {
        "this", "that", "with", "from", "such", "also", "based", "using", "into", "both",
        "more", "than", "then", "when", "where", "which", "there", "these", "those", "being",
        "upon", "each", "other", "paper", "work", "study", "approach", "method", "result",
        "results", "provide", "provides", "shown", "show", "present", "presented", "have",
        "while", "thus", "their", "they", "where", "given", "data", "high", "well",
    }
)

# Repo-spesifik teknik terimler: case-insensitive eslesme, IEEE Title-Case sabit cikti.
# Cakismayi (ChromaDB vs Chroma) tek pattern altinda birlestirdik: opsiyonel "db" eki.
_KNOWN_TERMS: list[tuple[str, str]] = [
    ("RAG", r"\brag\b"),
    ("LLM", r"\bllm\b"),
    ("OOXML", r"\booxml\b"),
    ("IEEE", r"\bieee\b"),
    ("Chroma", r"\bchroma(?:db)?\b"),
    ("LangChain", r"\blangchain\b"),
    ("Streamlit", r"\bstreamlit\b"),
    ("Gemini", r"\bgemini\b"),
    ("Mermaid", r"\bmermaid\b"),
    ("Parent-Child Retrieval", r"\bparent[\- ]child retrieval\b"),
    ("Hierarchical Indexing", r"\bhierarchical indexing\b"),
    ("Similarity Search", r"\bsimilarity search\b"),
    ("Multi-Query Retrieval", r"\bmulti[\- ]query retrieval\b"),
    ("Vector Store", r"\bvector (?:store|database|db)\b"),
    ("Embedding", r"\bembeddings?\b"),
    ("Retrieval-Augmented Generation", r"\bretrieval[\- ]augmented generation\b"),
]


def extract_keywords_from_abstract(abstract: str, max_keywords: int = 6) -> str:
    """
    Verilen abstract'ten deterministik IEEE keyword listesi cikarir; sonuc abstract metninin alt kumesidir.

    3 asamali: (1) buyuk-harf akronim regex (orijinal case korunur), (2) repo-spesifik bilinen
    terimlerin Title-Case formatinda eslesmesi, (3) bigram fallback (stop-word'le elenir).
    Cikti "Term1, Term2, ..." virgullu string; bos ise "" doner (DEFAULT placeholder fallback'i tetiklenir).
    """
    if not abstract:
        return ""

    found: list[str] = []
    seen_lower: set[str] = set()

    # 1) Akronim pass: orijinal case 2-5 harfli buyuk harf bloklarini yakala.
    for acr in re.findall(r"\b([A-Z]{2,5})\b", abstract):
        key = acr.lower()
        if key in seen_lower or key in _ACRONYM_BLACKLIST:
            continue
        found.append(acr)
        seen_lower.add(key)

    # 2) Bilinen teknik terimler: case-insensitive arama, sabit Title-Case cikti.
    abstract_lower = abstract.lower()
    for display, pattern in _KNOWN_TERMS:
        if re.search(pattern, abstract_lower) and display.lower() not in seen_lower:
            found.append(display)
            seen_lower.add(display.lower())

    # 3) Bigram fallback: yeterli sayida term yoksa abstract icindeki anlamli ikilikleri ekle.
    if len(found) < max_keywords:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", abstract)
        for i in range(len(words) - 1):
            w1_low = words[i].lower()
            w2_low = words[i + 1].lower()
            if w1_low in _BIGRAM_STOP_WORDS or w2_low in _BIGRAM_STOP_WORDS:
                continue
            bigram_display = f"{words[i].capitalize()} {words[i + 1].capitalize()}"
            key = bigram_display.lower()
            if key in seen_lower:
                continue
            found.append(bigram_display)
            seen_lower.add(key)
            if len(found) >= max_keywords:
                break

    return ", ".join(found[:max_keywords])


class MetadataWriter:
    """
    RAG destekli paper title/abstract uretici; keywords abstract uzerinden deterministik turetilir.

    LLM yalnizca title + abstract dondurur; keyword JSON'a girmez. generate() bos JSON / parse hatasi /
    LLM exception durumlarinda bos string'lerle dondurerek cagri tarafinda DEFAULT placeholder'in
    devreye girmesine izin verir.
    """

    def __init__(self, llm_invoke_func: Callable[[str], str]):
        """LLM cagri fonksiyonunu (prompt -> str) saklar; retry/throttle dis tarafta yonetilir."""
        self.llm_invoke = llm_invoke_func

    def generate(
        self,
        *,
        combined_body: str,
        repo_url: str = "",
        rag_documents: Optional[List[Document]] = None,
        max_body_chars: int = 18000,
        max_context_chars: int = 12000,
    ) -> Dict[str, str]:
        """
        Body + RAG context'ten title/abstract uretir, abstract'i 250 kelimede sertce keser ve keywords cikarir.

        Donus: {"title": str, "abstract": str, "keywords": str}. Hatada uc alan da "" doner.
        """
        body_text = combined_body or ""
        if len(body_text) > max_body_chars:
            body_text = body_text[:max_body_chars] + "\n...[BODY TRUNCATED]"

        ctx_text = ""
        if rag_documents:
            ctx_parts = [d.page_content for d in rag_documents if getattr(d, "page_content", "")]
            ctx_text = "\n\n".join(ctx_parts)
            if len(ctx_text) > max_context_chars:
                ctx_text = ctx_text[:max_context_chars] + "\n...[CONTEXT TRUNCATED]"

        # str.format yerine .replace zinciri: bolum metinlerindeki literal { } (Mermaid, Python, JSON
        # ornekleri) format() ile KeyError/IndexError firlatirdi.
        prompt = (
            _PROMPT_TEMPLATE
            .replace("{repo_url}", (repo_url or "").strip() or "(not provided)")
            .replace("{rag_context}", ctx_text or "(no additional context retrieved)")
            .replace("{combined_body}", body_text)
        )

        try:
            raw = self.llm_invoke(prompt)
            return self._parse_response(raw)
        except Exception as exc:  # noqa: BLE001
            logger.error("MetadataWriter LLM hatasi: %s", exc)
            return {"title": "", "abstract": "", "keywords": ""}

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, str]:
        """LLM ham ciktisini ```json fence/dis JSON karmasi durumlarinda da defansif olarak parse eder."""
        text = (raw or "").strip()
        # ``` fence'lerini soyup icindeki JSON'a in.
        if "```json" in text:
            text = text.split("```json", 1)[-1].split("```", 1)[0].strip()
        elif text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

        data: Dict[str, object] = {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = {}

        title = str(data.get("title") or "").strip()
        abstract_raw = str(data.get("abstract") or "").strip()
        abstract = MetadataWriter._enforce_word_cap(abstract_raw, 250)
        return {
            "title": title,
            "abstract": abstract,
            "keywords": extract_keywords_from_abstract(abstract),
        }

    @staticmethod
    def _enforce_word_cap(text: str, max_words: int) -> str:
        """LLM cikti uzunlugunu kelime sayisiyla sertce keser; nokta ile kapatir."""
        if not text:
            return ""
        tokens = text.split()
        if len(tokens) <= max_words:
            return text
        return " ".join(tokens[:max_words]).rstrip(",;:.- ") + "."
