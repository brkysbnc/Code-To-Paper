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

import logging
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

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

LOGGER = logging.getLogger(__name__)

DEFAULT_PARENT_CHUNK_SIZE = 1500
DEFAULT_PARENT_CHUNK_OVERLAP = 200
DEFAULT_CHILD_CHUNK_SIZE = 300
DEFAULT_CHILD_CHUNK_OVERLAP = 50
DEFAULT_COLLECTION_NAME = "code_parent_child_chunks"
PARENT_ID_KEY = "parent_id"


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
