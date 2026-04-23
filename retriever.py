"""
Parent-Child Document Retrieval katmanı.

Bu modül, repository içindeki kaynak kod dosyalarını dil farkında (code-aware)
olarak parçalar ve LangChain ParentDocumentRetriever mimarisiyle indeksler.

Temel akış:
1. Dosya uzantısına göre uygun language splitter seçilir.
2. Parent splitter büyük bağlam bloklarını üretir.
3. Child splitter arama isabeti yüksek küçük blokları üretir.
4. Child dokümanlar Chroma'ya yazılır.
5. Parent dokümanlar InMemoryStore içinde tutulur.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

try:
    # LangChain 0.x uyumluluğu
    from langchain.retrievers import ParentDocumentRetriever
except ImportError:  # pragma: no cover - LangChain 1.x uyumluluğu
    from langchain_classic.retrievers import ParentDocumentRetriever

try:
    # Yeni sürümlerde önerilen import yolu.
    from langchain.storage import InMemoryStore
except ImportError:  # pragma: no cover - geriye dönük uyumluluk
    try:
        from langchain_core.stores import InMemoryStore
    except ImportError:  # pragma: no cover - LangChain 1.x
        from langchain_classic.storage import InMemoryStore

try:
    # Ayrı paket kuruluysa bunu kullan.
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - mevcut requirements ile uyumluluk
    from langchain_community.vectorstores import Chroma

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

LOGGER = logging.getLogger(__name__)

# Free tier: gemini-embedding dakikada ~100 istek; parent-child cok child uretir.
_DEFAULT_EMBED_MIN_INTERVAL_S = 0.62


class ThrottledGeminiEmbeddings(Embeddings):
    """
    Gemini embedding cagrilarini siralar ve istekler arasinda bekler.

    Parent-child indekslemede tek dosya yuzlerce child uretebilir; Google
    free tier 429 verir. Bu sarmalayici embed_documents icinde metinleri tek
    tek (veya kisa aralikla) gondererek dakika kotasinin altinda kalir.
    """

    def __init__(
        self,
        inner: GoogleGenerativeAIEmbeddings,
        *,
        min_interval_s: float = _DEFAULT_EMBED_MIN_INTERVAL_S,
    ) -> None:
        self._inner = inner
        self._min_interval_s = max(0.05, float(min_interval_s))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Her metin icin ayri embedding + istekler arasi bekleme ve 429 retry."""
        if not texts:
            return []
        out: list[list[float]] = []
        for i, text in enumerate(texts):
            if i > 0:
                time.sleep(self._min_interval_s)
            out.append(self._embed_one_with_retry(text))
        return out

    def embed_query(self, text: str) -> list[float]:
        """Sorgu embeddingi; ayni kota icin kisa throttle uygulanir."""
        time.sleep(self._min_interval_s)
        return self._embed_one_with_retry(text)

    def _embed_one_with_retry(self, text: str) -> list[float]:
        """429/resource_exhausted durumunda ussel geri cekilme ile tekrar dener."""
        delay_s = 2.0
        last_exc: BaseException | None = None
        for attempt in range(10):
            try:
                vec = self._inner.embed_documents([text])
                return list(vec[0])
            except BaseException as exc:  # noqa: BLE001
                last_exc = exc
                err = str(exc).lower()
                retryable = any(
                    x in err
                    for x in (
                        "429",
                        "resource_exhausted",
                        "quota",
                        "503",
                        "unavailable",
                    )
                )
                if not retryable or attempt == 9:
                    raise
                LOGGER.warning(
                    "Embedding gecici hata (deneme %s/10), %.1fs bekleniyor: %s",
                    attempt + 1,
                    delay_s,
                    exc,
                )
                time.sleep(delay_s)
                delay_s = min(delay_s * 1.8, 90.0)
        raise RuntimeError(str(last_exc)) from last_exc


DEFAULT_PARENT_CHUNK_SIZE = 1500
DEFAULT_PARENT_CHUNK_OVERLAP = 200
DEFAULT_CHILD_CHUNK_SIZE = 300
DEFAULT_CHILD_CHUNK_OVERLAP = 50
DEFAULT_COLLECTION_NAME = "code_parent_child_chunks"
PARENT_ID_KEY = "parent_id"
DEFAULT_CONTEXT_TOKEN_BUDGET = 10_000


def _estimate_tokens(text: str) -> int:
    """
    Yaklaşık token hesabı yapar.

    API bağımsız hızlı limit kontrolü için 4 karakter ~= 1 token yaklaşımı
    kullanılır. Hassas hesap gerektiren üretimde modelin resmi tokenizer'ı
    ile değiştirilmelidir.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _format_context_block(document: Document) -> str:
    """
    Parent dokümanı writer/judge için standart bağlam metnine çevirir.
    """
    file_path = str(document.metadata.get("file_path", "unknown"))
    start_line = document.metadata.get("start_line", -1)
    end_line = document.metadata.get("end_line", -1)
    return (
        f"[source={file_path} lines={start_line}-{end_line}]\n"
        f"{document.page_content}"
    )


def build_gemini_embeddings(
    *,
    model: str = "gemini-embedding-001",
) -> GoogleGenerativeAIEmbeddings:
    """
    Gemini tabanlı embedding nesnesini `.env` üzerinden kurar.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY bulunamadi. Lutfen .env dosyasini kontrol edin.")
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)


def build_rag_stack_for_repo(
    repo_url: str,
    commit_hash: str,
    *,
    embedding_model: str = "gemini-embedding-001",
) -> tuple:
    """
    Belirli repo+commit icin izole Chroma dizini ve retriever stack'i kurar.

    Ayni repoyu tekrar indekslerken farkli koleksiyon/dizin carpismasin diye
    URL ve commit hash'ten kisa bir token uretilir.
    """
    token = hashlib.sha256(f"{repo_url}|{commit_hash}".encode("utf-8")).hexdigest()[:16]
    persist_dir = Path("data/chroma_rag") / token
    persist_dir.mkdir(parents=True, exist_ok=True)
    base_embeddings = build_gemini_embeddings(model=embedding_model)
    # Free tier: cok sayida child = cok embed; istekleri seyrelt
    if os.getenv("GEMINI_EMBED_NO_THROTTLE", "").strip().lower() in {"1", "true", "yes"}:
        embeddings = base_embeddings
    else:
        raw_iv = os.getenv("GEMINI_EMBED_MIN_INTERVAL_S", str(_DEFAULT_EMBED_MIN_INTERVAL_S)).strip()
        try:
            min_iv = float(raw_iv) if raw_iv else _DEFAULT_EMBED_MIN_INTERVAL_S
        except ValueError:
            min_iv = _DEFAULT_EMBED_MIN_INTERVAL_S
        embeddings = ThrottledGeminiEmbeddings(base_embeddings, min_interval_s=min_iv)
    return build_parent_child_retriever(
        embeddings,
        collection_name=f"code_pc_{token}",
        persist_directory=str(persist_dir),
    )


def configure_logging(level: int = logging.INFO) -> None:
    """
    Retriever loglarını konsolda görünür hale getirir.

    Aynı process içinde birden fazla kez çağrılsa da duplicate handler
    oluşturmamaya dikkat eder.
    """
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _normalize_repo_relative_path(path: Path, repo_root: Path) -> str:
    """Dosya yolunu repo köküne göre normalize edip `/` ayırıcısına çevirir."""
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _infer_content_type(path: Path) -> str:
    """Dosya uzantısından içerik kategorisi üretir."""
    suffix = path.suffix.lower()
    if suffix in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".c", ".cpp", ".h", ".hpp", ".cs", ".sql"}:
        return "code"
    if suffix == ".md" or path.name.lower().startswith("readme"):
        return "markdown"
    if suffix in {".json", ".yaml", ".yml", ".toml", ".xml"}:
        return "config"
    return "text"


def supported_language_for_path(path: str | Path) -> str:
    """
    Dosya uzantısını ortak language string alanına çevirir.

    Bu değer metadata içinde tutulur ve ileride filtreleme / raporlama için
    kullanılır.
    """
    suffix = Path(path).suffix.lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".sql": "sql",
        ".md": "markdown",
        ".json": "json",
        ".xml": "xml",
        ".html": "html",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
    }
    return mapping.get(suffix, "unknown")


def _resolve_langchain_language(path: str | Path) -> Language | None:
    """
    Dosya uzantısına göre LangChain `Language` enum değeri döndürür.

    Desteklenmeyen uzantılarda `None` döner; bu durumda kontrollü fallback
    splitter kullanılacaktır.
    """
    suffix = Path(path).suffix.lower()
    mapping = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".jsx": Language.JS,
        ".ts": Language.TS,
        ".tsx": Language.TS,
        ".java": Language.JAVA,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".rb": Language.RUBY,
        ".php": Language.PHP,
        ".swift": Language.SWIFT,
        ".kt": Language.KOTLIN,
        ".scala": Language.SCALA,
        ".c": Language.CPP,
        ".cpp": Language.CPP,
        ".h": Language.CPP,
        ".hpp": Language.CPP,
        ".cs": Language.CSHARP,
        ".md": Language.MARKDOWN,
        ".html": Language.HTML,
    }
    return mapping.get(suffix)


def build_language_splitter(
    path: str | Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """
    Dosya türüne göre code-aware splitter oluşturur.

    Öncelik:
    1. LangChain `from_language(...)` ile dil farkında ayırma
    2. Config / text dosyalarında yapısal ayraçlarla kontrollü fallback
    """
    language = _resolve_langchain_language(path)
    if language is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    suffix = Path(path).suffix.lower()
    if suffix in {".json", ".toml", ".yaml", ".yml", ".xml"}:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ",", " ", ""],
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def build_parent_child_retriever(
    embeddings,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = "data/chroma_parent_child",
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_chunk_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
):
    """
    Chroma + InMemoryStore + ParentDocumentRetriever yapısını kurar.

    Returns:
        tuple[ParentDocumentRetriever, InMemoryStore, Chroma]
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=["\nclass ", "\ndef ", "\nasync def ", "\n\n", "\n", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\nclass ", "\ndef ", "\nasync def ", "\n\n", "\n", " ", ""],
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        id_key=PARENT_ID_KEY,
        search_kwargs={"k": 8},
    )
    return retriever, store, vectorstore


def _read_document(
    file_path: Path,
    *,
    repo_root: Path,
    repo_url: str,
    commit_hash: str,
) -> Document:
    """
    Tek bir dosyayı LangChain `Document` nesnesine dönüştürür.

    Metadata alanları retrieval sonrası açıklanabilirlik ve faithfulness
    değerlendirmesi için özellikle geniş tutulur.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    relative_path = _normalize_repo_relative_path(file_path, repo_root)
    line_count = text.count("\n") + (1 if text else 0)

    metadata = {
        "repo_url": repo_url,
        "commit_hash": commit_hash,
        "file_path": relative_path,
        "language": supported_language_for_path(file_path),
        "content_type": _infer_content_type(file_path),
        "start_line": 1,
        "end_line": line_count,
        "source": relative_path,
    }
    return Document(page_content=text, metadata=metadata)


def _estimate_parent_child_counts(
    document: Document,
    *,
    file_path: Path,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
) -> tuple[int, int]:
    """
    Dosya için tahmini parent/child parça sayılarını hesaplar.

    Bu hesap loglama amaçlıdır; gerçek indeksleme yine retriever tarafından
    yapılır. Aynı splitter parametreleri kullanıldığı için sayıların pratikte
    retriever ile uyumlu olması beklenir.
    """
    parent_splitter = build_language_splitter(
        file_path,
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
    )
    parent_docs = parent_splitter.split_documents([document])

    child_count = 0
    for parent_doc in parent_docs:
        child_splitter = build_language_splitter(
            file_path,
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
        )
        child_docs = child_splitter.split_documents([parent_doc])
        child_count += len(child_docs)

    return len(parent_docs), child_count


def index_repository_files(
    retriever: ParentDocumentRetriever,
    file_paths: Iterable[str | Path],
    *,
    repo_root: str | Path,
    repo_url: str,
    commit_hash: str,
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_chunk_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
) -> dict[str, int]:
    """
    Verilen repository dosyalarını parent-child retrieval mimarisiyle indeksler.

    Logging çıktısı sayesinde her dosyanın kaç parent ve child parçaya ayrıldığı
    konsolda görülebilir.
    """
    repo_root_path = Path(repo_root).resolve()
    totals = {"files": 0, "parents": 0, "children": 0}

    for raw_path in file_paths:
        file_path = Path(raw_path).resolve()
        if not file_path.is_file():
            LOGGER.warning("Atlandi: dosya bulunamadi -> %s", file_path)
            continue

        document = _read_document(
            file_path,
            repo_root=repo_root_path,
            repo_url=repo_url,
            commit_hash=commit_hash,
        )
        parent_count, child_count = _estimate_parent_child_counts(
            document,
            file_path=file_path,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
        )

        # ParentDocumentRetriever tek bir splitter nesnesi tuttuğu için, her dosya
        # öncesinde splitter'ları o dosyanın diline göre güncelliyoruz.
        retriever.parent_splitter = build_language_splitter(
            file_path,
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )
        retriever.child_splitter = build_language_splitter(
            file_path,
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
        )

        # ParentDocumentRetriever parent kimliklerini kendi içinde yönetir.
        retriever.add_documents([document])

        relative_path = _normalize_repo_relative_path(file_path, repo_root_path)
        LOGGER.info(
            "%s dosyasi %s parent, %s child parcaya bolundu.",
            relative_path,
            parent_count,
            child_count,
        )

        totals["files"] += 1
        totals["parents"] += parent_count
        totals["children"] += child_count

    LOGGER.info(
        "Indeksleme tamamlandi. %s dosya, %s parent, %s child parca hazirlandi.",
        totals["files"],
        totals["parents"],
        totals["children"],
    )
    return totals


def query_context(
    retriever: ParentDocumentRetriever,
    query: str,
) -> list[Document]:
    """
    Sorgu için parent bağlam dokümanlarını döndürür.

    Retrieval sonucu parent blokları içerdiği için writer agent doğrudan bu
    çıktı üzerinden akademik metin üretebilir.
    """
    LOGGER.info("Retrieval basladi: %s", query)
    documents = retriever.invoke(query)
    LOGGER.info("Retrieval tamamlandi. %s parent dokuman dondu.", len(documents))
    return documents


def generate_planner_queries(
    llm: BaseChatModel,
    *,
    section_title: str,
    section_goal: str,
    max_queries: int = 6,
    max_llm_attempts: int = 5,
    base_sleep_seconds: float = 2.0,
) -> list[str]:
    """
    Planner ajanı için section bazlı çoklu teknik sorgu üretir.

    Gemini tarafinda 503/429 gibi gecici hatalar icin sinirli retry + exponential
    backoff uygulanir (Is 4b: API dayanikliligi).
    """
    prompt = (
        "You are a retrieval planner for software architecture analysis.\n"
        f"Section title: {section_title}\n"
        f"Section goal: {section_goal}\n"
        f"Generate {max_queries} concise search queries for code retrieval.\n"
        "Rules:\n"
        "- Focus on code-level terms and architecture clues.\n"
        "- Include synonyms and implementation-oriented variants.\n"
        "- Return only one query per line, no numbering."
    )
    last_error: Exception | None = None
    response_text = ""
    for attempt in range(max_llm_attempts):
        try:
            response_text = str(llm.invoke(prompt).content)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            err_s = str(exc).lower()
            retryable = any(
                token in err_s
                for token in (
                    "503",
                    "429",
                    "unavailable",
                    "resource_exhausted",
                    "quota",
                    "deadline",
                    "timeout",
                )
            )
            if not retryable or attempt == max_llm_attempts - 1:
                raise
            sleep_s = base_sleep_seconds * (2**attempt)
            LOGGER.warning(
                "Planner LLM denemesi %s basarisiz, %.1f sn sonra tekrar: %s",
                attempt + 1,
                sleep_s,
                exc,
            )
            time.sleep(sleep_s)
    if not response_text and last_error:
        raise last_error
    lines = [line.strip(" -\t") for line in str(response_text).splitlines() if line.strip()]
    unique: list[str] = []
    for line in lines:
        if line not in unique:
            unique.append(line)
    if not unique:
        unique = [section_title]
    return unique[:max_queries]


def dedupe_parent_documents_by_location(docs: list[Document]) -> list[Document]:
    """
    Ayni dosya ve satir araligina denk gelen parent bloklari tekrar etmez.

    Farkli parent_id ile ayni bolge iki kez gelebildiginden Writer baglaminda
    gereksiz tekrar ve token israfini azaltir (Is 4b).
    """
    seen: set[tuple[str, int, int]] = set()
    out: list[Document] = []
    for doc in docs:
        fp = str(doc.metadata.get("file_path", "")).strip()
        try:
            sl = int(doc.metadata.get("start_line", -1))
            el = int(doc.metadata.get("end_line", -1))
        except (TypeError, ValueError):
            sl, el = -1, -1
        key = (fp, sl, el)
        if fp and key in seen:
            continue
        if fp:
            seen.add(key)
        out.append(doc)
    return out


def retrieve_parent_contexts_multi_query(
    retriever: ParentDocumentRetriever,
    *,
    planner_queries: list[str],
    top_k_per_query: int = 6,
    similarity_threshold: float = 0.6,
    max_context_tokens: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
) -> list[Document]:
    """
    Multi-query sonuçlarını child seviyesinde toplayıp parent bağlama yükseltir.

    - Düşük benzerlik skorlarını eleyerek çöp veriyi azaltır.
    - Parent blokları token bütçesine göre paketler.
    """
    if not planner_queries:
        return []

    parent_hits: dict[str, tuple[float, int]] = {}
    for query in planner_queries:
        child_hits = retriever.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=top_k_per_query,
        )
        for child_doc, score in child_hits:
            if score < similarity_threshold:
                continue
            parent_id = str(child_doc.metadata.get(PARENT_ID_KEY, "")).strip()
            if not parent_id:
                continue
            best_score, seen_count = parent_hits.get(parent_id, (0.0, 0))
            parent_hits[parent_id] = (max(best_score, score), seen_count + 1)

    if not parent_hits:
        return []

    ranked_parent_ids = sorted(
        parent_hits.items(),
        key=lambda item: (item[1][1], item[1][0]),
        reverse=True,
    )
    parent_ids = [pid for pid, _ in ranked_parent_ids]

    docstore = retriever.docstore
    parent_docs = docstore.mget(parent_ids)

    packed_docs: list[Document] = []
    used_tokens = 0
    for doc in parent_docs:
        if not isinstance(doc, Document):
            continue
        block = _format_context_block(doc)
        block_tokens = _estimate_tokens(block)
        if used_tokens + block_tokens > max_context_tokens:
            continue
        packed_docs.append(doc)
        used_tokens += block_tokens

    packed_docs = dedupe_parent_documents_by_location(packed_docs)

    LOGGER.info(
        "Multi-query retrieval tamamlandi. %s parent secildi, yaklasik %s token paketlendi.",
        len(packed_docs),
        used_tokens,
    )
    return packed_docs
