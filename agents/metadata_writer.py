"""
RAG destekli akademik baslik / abstract uretimi ve abstract metninden deterministik keyword cikarimi.

Akis (tek LLM cagrisi):
1) MetadataWriter.generate combined_body (yazilmis bolumler) + rag_documents (ek repo kaniti) + repo_url alir.
2) Body 18000, RAG context 12000 char ile kesilir; LLM'e str.replace ile enjekte edilir
   (str.format kullanilmaz — bolum metinlerindeki literal { } kod fence'lerini bozmamak icin).
3) LLM yalnizca {"title", "abstract", "keywords"} JSON dondurur; abstract 250 kelimede sertce kesilir.
4) Keywords oncelikle LLM'den alinir; abstract'ta teyit edilemeyenler (ek API yok) silinir,
   listeyi deterministic extractor ile tamamlar. Parse edilemezse yalnizca regex fallback.
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

Produce THREE pieces of metadata in JSON format:

1) TITLE: A concise, descriptive academic title (8-15 words).
   - Reflect the paper's actual technical contribution.
   - Use specific technical terms found in BODY or CONTEXT.

2) ABSTRACT: A single paragraph in IEEE style.
   - LENGTH & COMPLETENESS: The abstract must be thorough and complete — aim for 200-250 words. Do NOT artificially pad with filler sentences. Instead, ensure each of the four required questions is answered with sufficient technical detail: the problem should include a concrete example or context, the method should name at least two specific technical components from the system, the results should include at least one concrete finding or trade-off, and the significance should explain who benefits and why. If you finish all four points and have fewer than 200 words, expand the technical depth of your weakest point — do not repeat yourself.
   - REQUIRED CONTENT: It must answer four questions: (1) What problem does this work solve? (2) How is it solved? (3) What are the key results or findings? (4) Why does it matter?
   - QUALITY & TONE: Write the abstract for a human reader first, not a machine. Avoid overly technical jargon in the opening sentence. Start with the problem being solved in plain language. Progress from problem → method → result → significance. Include at least one concrete finding or metric if available from the retrieved context. The tone should be accessible to a conference attendee who is not yet familiar with the system.
   - Use ONLY facts present in BODY or CONTEXT. Do NOT invent numbers.
   - Include domain-specific technical terms (e.g. RAG, ChromaDB, OOXML).
   - FORBIDDEN OPENINGS: Do NOT start with any of these patterns:
     'The rapid', 'This paper presents', 'In recent years',
     'The growing', 'With the advent', 'As technology evolves'.
     Start instead with the SYSTEM NAME or the CORE CONTRIBUTION.
   - Do NOT copy or paraphrase sentences from the Introduction section
     in BODY. The abstract must be independently written.

3) KEYWORDS: A list of 4-6 IEEE-style keywords.
   - Extract ONLY from the abstract text you just wrote.
   - Do NOT invent terms not present in the abstract.
   - Prefer acronyms over full forms when both appear (e.g. "RAG" not "Retrieval-Augmented Generation").
   - Use standard IEEE keyword format: Title Case for multi-word terms, uppercase for acronyms.
   - Do NOT include generic words like "system", "paper", "approach", "method", "proposed".

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
{"title": "<string>", "abstract": "<string>", "keywords": ["<string>", ...]}
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


# Kısaltma → açılım eşleşmeleri (her ikisi de keyword listesine girerse
# sadece kısaltma kalır). Küçük harf anahtarlar.
_ABBREVIATION_EXPANSIONS: dict[str, str] = {
    "rag": "retrieval-augmented generation",
    "llm": "large language model",
    "ieee": "institute of electrical and electronics engineers",
    "json": "javascript object notation",
    "api": "application programming interface",
    "nlp": "natural language processing",
    "mas": "multi-agent system",
}


def deduplicate_keywords(keywords: list[str]) -> list[str]:
    """
    Keyword listesinden kısaltma+açılım tekrarlarını temizler.
    İki strateji:
    1) Explicit eşleşme tablosu (_ABBREVIATION_EXPANSIONS): bilinen çiftlerde
       kısaltmayı tutar, açılımı atar.
    2) Substring fallback: biri diğerinin içinde geçiyorsa kısa olanı tutar.
    """
    result = []
    lower_map = {kw.lower(): kw for kw in keywords}
    removed: set[str] = set()

    # Strateji 1: Explicit eşleşme tablosu
    for abbr_lower, expansion_lower in _ABBREVIATION_EXPANSIONS.items():
        if abbr_lower in lower_map and expansion_lower in lower_map:
            # İkisi de varsa açılımı kaldır, kısaltmayı tut
            removed.add(expansion_lower)

    # Strateji 2: Substring fallback (explicit tabloda olmayan çiftler için)
    kw_list = [kw for kw in keywords if kw.lower() not in removed]
    for i, kw in enumerate(kw_list):
        for j, other in enumerate(kw_list):
            if i == j or other.lower() in removed:
                continue
            if kw.lower() in other.lower() and len(kw) < len(other):
                # kw, other'ın içinde geçiyor → other açılım, kw kısaltma
                removed.add(other.lower())

    for kw in keywords:
        if kw.lower() not in removed:
            result.append(kw)

    return result


def _keywords_grounded_in_abstract(keywords: List[str], abstract: str) -> List[str]:
    """
    LLM keyword adaylarini abstract metnine baglar: yalnizca abstract icinde (buyuk/kucuk harf
    duyarsiz alt-dize olarak) gecen ifadeler kalir. Ek model cagrisi yok; halisunasyonu keser.
    """
    hay = (abstract or "").lower()
    out: List[str] = []
    seen: set[str] = set()
    for k in keywords:
        t = str(k).strip()
        if not t:
            continue
        if t.lower() in hay and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def _merge_keywords_llm_then_deterministic(
    grounded_llm: List[str],
    abstract: str,
    *,
    max_keywords: int = 6,
) -> str:
    """
    Once LLM'den gecen abstract-ici keyword'leri kullanir; 6'ya tamamlamak icin
    extract_keywords_from_abstract ile doldurur (tekrarsiz). LLM listesi bossa tamamen deterministic.
    """
    cap = max(1, int(max_keywords))
    out: List[str] = []
    seen: set[str] = set()
    for k in grounded_llm:
        tl = k.lower()
        if tl in seen:
            continue
        seen.add(tl)
        out.append(k)
        if len(out) >= cap:
            return ", ".join(out)
    det = extract_keywords_from_abstract(abstract, max_keywords=cap)
    for part in [p.strip() for p in det.split(",") if p.strip()]:
        pl = part.lower()
        if pl in seen:
            continue
        seen.add(pl)
        out.append(part)
        if len(out) >= cap:
            break
    return ", ".join(out)


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

    # 3) Bigram fallback: SON CARE; yalnizca aşama 1+2'den 3'ten az anlamli term cikti ise
    # tetiklenir. Boylece "Rapid Evolution" / "Evolution Technical" gibi cumle ortasi
    # dolgu bigramlari rutin olarak keyword listesine sizmaz.
    if len(found) < 3:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", abstract)
        for i in range(len(words) - 1):
            w1_low = words[i].lower()
            w2_low = words[i + 1].lower()
            if w1_low in _BIGRAM_STOP_WORDS or w2_low in _BIGRAM_STOP_WORDS:
                continue
            # Cumle basi / proper-noun bigramlarini ele: ilk harfi buyuk olan kelime
            # iceren bigram atlanir. Acronymler aşama 1'de zaten yakalandi (ALL CAPS)
            # icin bu filtre yalnizca lowercase content kelimelerini bigram'a sokar.
            if words[i][0].isupper() or words[i + 1][0].isupper():
                continue
            # Min uzunluk filtresi: stop-word listesinden kacan kisa yaygın kelimeler
            # ("the", "its", "use", "key" vb.) bigram olusturmasin diye iki tarafta
            # da en az 5 karakter zorunlu.
            if len(w1_low) < 5 or len(w2_low) < 5:
                continue
            bigram_display = f"{words[i].capitalize()} {words[i + 1].capitalize()}"
            key = bigram_display.lower()
            if key in seen_lower:
                continue
            found.append(bigram_display)
            seen_lower.add(key)
            if len(found) >= max_keywords:
                break

    found = deduplicate_keywords(found)
    return ", ".join(found[:max_keywords])


class MetadataWriter:
    """
    RAG destekli title/abstract + tek JSON ciktisinda LLM keywords.

    generate() ciktisinda keywords: LLM listesi once alinir, abstract icinde gecmeyen terimler silinir,
    4-6 slot deterministic extractor ile doldurulabilir. Parse/LLM hatasinda alanlar "".
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
        
        # Keywords: LLM listesi / string -> abstract'ta substring dogrulama -> gerekirse deterministic tamamlama.
        keywords_raw = data.get("keywords")
        kw_list: List[str] = []
        if isinstance(keywords_raw, list):
            kw_list = [str(k).strip() for k in keywords_raw if str(k).strip()]
        elif isinstance(keywords_raw, str) and keywords_raw.strip():
            kw_list = [p.strip() for p in re.split(r"[,;]", keywords_raw) if p.strip()]

        if kw_list:
            kw_list = deduplicate_keywords(kw_list)
            grounded = _keywords_grounded_in_abstract(kw_list, abstract)
            # Az sayida gecerli terim kaldiysa tamamen deterministic daha guvenilir.
            if len(grounded) < 2:
                keywords = extract_keywords_from_abstract(abstract)
            else:
                keywords = _merge_keywords_llm_then_deterministic(grounded, abstract, max_keywords=6)
        else:
            keywords = extract_keywords_from_abstract(abstract)

        if not MetadataWriter._check_minimum_words(abstract, 150):
            logger.warning(
                "Abstract minimum kelime sayısını karşılamıyor: %d kelime (min 150). "
                "Abstract kısa kalmış olabilir.",
                len(abstract.split()) if abstract else 0,
            )
        return {
            "title": title,
            "abstract": abstract,
            "keywords": keywords,
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

    @staticmethod
    def _check_minimum_words(text: str, min_words: int) -> bool:
        """Abstract'in minimum kelime sayısını karşılayıp karşılamadığını kontrol eder."""
        if not text:
            return False
        return len(text.split()) >= min_words
