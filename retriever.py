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
6. Writer CONTEXT'ine girmemesi gereken dosyalar indekslenirken metadata ile isaretlenir;
   multi-query akisi cocuk isabetlerinde ve parent paketlemede `document_blocked_for_writer_context`
   ile filtrelenir (eski indekslerde bile path/icerik sezgisi devreye girer).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
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

# Parent dokumanlari diske yazip oturumlar arasi kalici tutmak icin LocalFileStore.
# LangChain >=1.x'te `langchain.storage` paketi kaldirildi; classic paketi kullaniyoruz.
try:
    from langchain_classic.storage import LocalFileStore
    from langchain_classic.storage._lc_store import create_kv_docstore
except ImportError:  # pragma: no cover - eski versiyonlar
    try:
        from langchain.storage import LocalFileStore
        from langchain.storage._lc_store import create_kv_docstore
    except ImportError:  # pragma: no cover
        LocalFileStore = None  # type: ignore[assignment]
        create_kv_docstore = None  # type: ignore[assignment]

try:
    # Ayrı paket kuruluysa bunu kullan.
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - mevcut requirements ile uyumluluk
    from langchain_community.vectorstores import Chroma

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

LOGGER = logging.getLogger(__name__)

# Free tier: gemini-embedding dakikada ~100 istek (per-request).
# Onemli: batchEmbedContents tek istek sayilir; o yuzden kucuk batch'lerle gonderirsek
# kota cok daha geç dolar. Tek tek gondermek free-tier'i hizla yakar.
_DEFAULT_EMBED_MIN_INTERVAL_S = 0.62
_DEFAULT_EMBED_BATCH_SIZE = 100

# gemini-embedding-001 cikis boyutu (API cevabi bos geldiginde pad icin referans).
_GEMINI_EMBEDDING_001_DIM = 768


def _infer_embedding_dim_from_vectors(vecs: list | None, *, default: int = _GEMINI_EMBEDDING_001_DIM) -> int:
    """
    Ilk gecerli (None olmayan, icinde en az bir eleman olan) vektorun uzunlugunu dondurur.

    Tum liste None veya bos vektorlerden olusuyorsa `default` kullanilir; boylece
    Chroma upsert icin tutarli boyutta sifir vektor uretilebilir.
    """
    if not vecs:
        return default
    for v in vecs:
        if v is None:
            continue
        try:
            seq = list(v)
        except TypeError:
            continue
        if seq:
            return len(seq)
    return default


def _coerce_gemini_batch_embeddings(batch_len: int, vecs: list | None) -> list[list[float]]:
    """
    Inner `embed_documents` ciktisini her zaman `batch_len` uzunlugunda float vektor listesine cevirir.

    Chroma `upsert` bos embedding listesi (`[]`) ile patlar; Gemini bazen guvenlik veya
    servis tarafinda `None`, bos liste veya girdi sayisindan kisa/uzun liste donebilir.
    Eksik indeksler sifir vektor, fazla indeksler yok sayilir (uyari loglanir).
    """
    if batch_len <= 0:
        return []
    dim = _infer_embedding_dim_from_vectors(vecs, default=_GEMINI_EMBEDDING_001_DIM)
    if vecs is None:
        LOGGER.warning(
            "Embedding API None dondurdu (batch_len=%s); tum parcalar sifir vektor (%s boy) ile dolduruldu.",
            batch_len,
            dim,
        )
        return [[0.0] * dim for _ in range(batch_len)]
    if len(vecs) == 0:
        LOGGER.warning(
            "Embedding API bos vektor listesi dondurdu (batch_len=%s); tum parcalar sifir vektor ile dolduruldu.",
            batch_len,
        )
        return [[0.0] * dim for _ in range(batch_len)]
    if len(vecs) != batch_len:
        LOGGER.warning(
            "Embedding vektor sayisi uyumsuz: beklenen=%s, gelen=%s. Eksikler sifir vektor, fazlalik yok sayilacak.",
            batch_len,
            len(vecs),
        )
    out: list[list[float]] = []
    for i in range(batch_len):
        v = vecs[i] if i < len(vecs) else None
        if v is None:
            LOGGER.warning("Batch indeks %s icin embedding yok (None / kisa liste), sifir vektor kullaniliyor.", i)
            out.append([0.0] * dim)
            continue
        try:
            seq = list(v)
        except TypeError:
            LOGGER.warning("Batch indeks %s embedding listeye cevrilemedi, sifir vektor.", i)
            out.append([0.0] * dim)
            continue
        if not seq:
            LOGGER.warning("Batch indeks %s icin embedding bos dizi, sifir vektor.", i)
            out.append([0.0] * dim)
            continue
        out.append(seq)
    return out


class ThrottledGeminiEmbeddings(Embeddings):
    """
    Gemini embedding cagrilarini batch'leyip istekler arasinda bekler.

    Onceki surum: her metin icin ayri embed_documents([text]) cagriyordu;
    free-tier'da N child = N HTTP istegi -> kota hizli dolar. Yeni surum:
    metinleri batch_size'lik gruplara bolup tek istekle gonderir; her batch
    arasinda min_interval_s kadar bekler. Toplam istek sayisi ~ N / batch_size.
    """

    def __init__(
        self,
        inner: GoogleGenerativeAIEmbeddings,
        *,
        min_interval_s: float = _DEFAULT_EMBED_MIN_INTERVAL_S,
        batch_size: int = _DEFAULT_EMBED_BATCH_SIZE,
    ) -> None:
        self._inner = inner
        self._min_interval_s = max(0.05, float(min_interval_s))
        self._batch_size = max(1, int(batch_size))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Metinleri batch_size'lik gruplara bolup tek istekle yollar; gruplar arasi throttle uygulanir."""
        if not texts:
            return []

        out: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self._batch_size):
            if start > 0:
                time.sleep(self._min_interval_s)
            batch = list(texts[start : start + self._batch_size])
            
            # Batch icindeki metinleri on-isleme: cok kisa veya bos olanlari temizle
            # Gemini bazen 0-length stringlerde 400 error veriyor.
            safe_batch = [t if (t and t.strip()) else " " for t in batch]
            
            try:
                embeddings = self._embed_batch_with_retry(safe_batch)
                out.extend(embeddings)
            except Exception as exc:
                LOGGER.error("Batch embedding basarisiz (start=%s, size=%s): %s", start, len(batch), exc)
                raise

        return out

    def embed_query(self, text: str) -> list[float]:
        """Sorgu embeddingi; tek metin -> tek istek (kucuk throttle ile)."""
        time.sleep(self._min_interval_s)
        safe_text = text if (text and text.strip()) else " "
        return self._embed_batch_with_retry([safe_text])[0]

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """Bir batch icin embed_documents cagrisini 429/503'e karsi sinirli retry ile yapar."""
        if not batch:
            return []
        delay_s = 2.0
        last_exc: BaseException | None = None
        for attempt in range(10):
            try:
                vecs = self._inner.embed_documents(list(batch))
                return _coerce_gemini_batch_embeddings(len(batch), vecs)
            except BaseException as exc:  # noqa: BLE001
                last_exc = exc
                err = str(exc).lower()
                # Quota/HTTP geri itmesi + Windows DNS/socket hicotsu (getaddrinfo) ve
                # genel baglanti kopukluklarini (connection reset/aborted) retryable kabul ediyoruz.
                retryable = any(
                    x in err
                    for x in (
                        "429",
                        "resource_exhausted",
                        "quota",
                        "503",
                        "unavailable",
                        "deadline",
                        "timeout",
                        "getaddrinfo",
                        "connection",
                        "temporary failure",
                    )
                )
                if not retryable or attempt == 9:
                    raise
                LOGGER.warning(
                    "Embedding batch gecici hata (deneme %s/10, n=%s), %.1fs bekleniyor: %s",
                    attempt + 1,
                    len(batch),
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

# Writer CONTEXT'ine alinmayacak kaynaklar bu metadata ile isaretlenir (yalniz True iken etkili).
_WRITER_CONTEXT_EXCLUDE_KEY = "exclude_from_writer_context"

# Proje yapisi: JSON-only Writer prompt dosyasi AcademicWriter CONTEXT'ine girmemeli (indirect injection).
_WRITER_CONTEXT_BLOCKED_PATH_SUFFIXES: tuple[str, ...] = (
    "agents/ieee_json_writer.py",
)


def _normalize_repo_relative_path_str(relative_path: str) -> str:
    """Windows/posix ayraclarini tek forma indirger (path karsilastirmasi icin)."""
    return (relative_path or "").replace("\\", "/").strip().lower()


def _metadata_exclude_writer_context_truthy(metadata: dict | None) -> bool:
    """
    Chroma/LangChain metadata degerleri bazen bool yerine string tutar; True kabul edilen tum varyantlari yakalar.
    """
    if not metadata:
        return False
    v = metadata.get(_WRITER_CONTEXT_EXCLUDE_KEY)
    if v is True:
        return True
    if isinstance(v, str) and v.strip().lower() == "true":
        return True
    if v == 1:
        return True
    return False


def should_exclude_from_writer_context(*, relative_path: str, text: str) -> bool:
    """
    AcademicWriter CONTEXT'ine konmamasi gereken dosya parcasini path + icerik ile tespit eder.

    - Oncelik: bilinen kod tabani yollari (`_WRITER_CONTEXT_BLOCKED_PATH_SUFFIXES`).
    - Ek: parca metninde JSON-only prompt imzalari (IEEE_JSON_ONLY_PROMPT, Output ONLY … JSON, vb.).
    Eski indekslerde metadata bayragi olmasa bile path/icerik yakalarsa parent paketleme elenir.
    """
    norm = _normalize_repo_relative_path_str(relative_path)
    for suffix in _WRITER_CONTEXT_BLOCKED_PATH_SUFFIXES:
        if norm == suffix or norm.endswith("/" + suffix):
            return True
    if not text:
        return False
    if "IEEE_JSON_ONLY_PROMPT" in text:
        return True
    low = text.lower()
    if re.search(r"output\s+only[^\n]{0,120}json", low):
        return True
    if "valid json object" in low:
        return True
    return False


def document_blocked_for_writer_context(doc: Document) -> bool:
    """
    Tek bir LangChain Document icin Writer baglam filtresi (cocuk veya parent).

    Sira: (1) indekslemede yazilan exclude metadata, (2) file_path/source + page_content heuristikleri.
    ParentDocumentRetriever cocuklari parent metadata'dan cogaltir; metadata yoksa yine path ile yakalanir.
    """
    md = doc.metadata or {}
    if _metadata_exclude_writer_context_truthy(md):
        return True
    rel = str(md.get("file_path") or md.get("source") or "").strip()
    return should_exclude_from_writer_context(relative_path=rel, text=doc.page_content or "")


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


def _resolve_repo_persist_dir(repo_url: str, commit_hash: str) -> tuple[Path, str]:
    """
    repo_url + commit_hash'ten deterministik bir token uretip Chroma kalici dizinini dondurur.

    Ayni repo + commit icin ayni dizin; commit degisirse yeni dizin (otomatik invalidation).
    """
    token = hashlib.sha256(f"{repo_url}|{commit_hash}".encode("utf-8")).hexdigest()[:16]
    persist_dir = Path("data/chroma_rag").resolve() / token
    persist_dir.mkdir(parents=True, exist_ok=True)
    return persist_dir, token


def is_repo_already_indexed(persist_dir: Path | str) -> bool:
    """
    Verilen dizinde Chroma + parent docstore birlikte hazir mi kontrol eder.

    Yeniden embeddinge girmemek icin: hem children koleksiyonu hem parent diskstore dolu olmali.
    Aksi halde retrieval'da parent_id var ama mget bos doner; bu yari-bos durum kullanilamaz.
    """
    p = Path(persist_dir)
    chroma_marker = p / "chroma.sqlite3"
    docstore_dir = p / "docstore"
    if not chroma_marker.is_file():
        return False
    if not docstore_dir.is_dir():
        return False
    try:
        return any(docstore_dir.iterdir())
    except OSError:
        return False


def build_rag_stack_for_repo(
    repo_url: str,
    commit_hash: str,
    *,
    embedding_model: str = "gemini-embedding-001",
) -> tuple:
    """
    Belirli repo+commit icin izole Chroma dizini ve retriever stack'i kurar.

    Ayni repoyu tekrar indekslerken farkli koleksiyon/dizin carpismasin diye
    URL ve commit hash'ten kisa bir token uretilir. Parent dokumanlari LocalFileStore
    icinde diske yazilir; sonraki oturum/cagriya hazir kalir (free-tier embed kotasini korur).
    """
    persist_dir, token = _resolve_repo_persist_dir(repo_url, commit_hash)
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
        raw_bs = os.getenv("GEMINI_EMBED_BATCH_SIZE", str(_DEFAULT_EMBED_BATCH_SIZE)).strip()
        try:
            bs = int(raw_bs) if raw_bs else _DEFAULT_EMBED_BATCH_SIZE
        except ValueError:
            bs = _DEFAULT_EMBED_BATCH_SIZE
        embeddings = ThrottledGeminiEmbeddings(
            base_embeddings,
            min_interval_s=min_iv,
            batch_size=bs,
        )
    docstore_dir = persist_dir / "docstore"
    docstore_dir.mkdir(parents=True, exist_ok=True)
    return build_parent_child_retriever(
        embeddings,
        collection_name=f"code_pc_{token}",
        persist_directory=str(persist_dir),
        docstore_dir=str(docstore_dir),
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
    docstore_dir: str | None = None,
):
    """
    Chroma + parent docstore + ParentDocumentRetriever yapısını kurar.

    docstore_dir verilirse kalici LocalFileStore kullanilir (langchain-classic zorunlu).
    docstore_dir None ise gelistirme/test icin InMemoryStore kullanilir.

    Returns:
        tuple[ParentDocumentRetriever, store, Chroma]
    """
    # Windows/Streamlit uyumu icin persist_directory'i absolute yap
    persist_abs = str(Path(persist_directory).resolve())
    if docstore_dir:
        docstore_abs = str(Path(docstore_dir).resolve())
    else:
        docstore_abs = None

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
        persist_directory=persist_abs,
    )
    # Kalici docstore zorunlu: Chroma diskte kalip InMemory docstore kullanilirsa
    # vektorler parent metnine map edilemez (yargi/writer'da 'kanit yok' zinciri).
    if docstore_abs:
        if LocalFileStore is None or create_kv_docstore is None:
            raise RuntimeError(
                "Kalici parent docstore icin langchain-classic gerekli. "
                "Komut: pip install langchain-classic"
            )
        store = create_kv_docstore(LocalFileStore(docstore_abs))
    else:
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
    if should_exclude_from_writer_context(relative_path=relative_path, text=text):
        metadata[_WRITER_CONTEXT_EXCLUDE_KEY] = True
        LOGGER.info(
            "Writer CONTEXT disinda birakildi (prompt/json-imza): %s",
            relative_path,
        )
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


def _vectorstore_has_documents(vectorstore: Chroma) -> bool:
    """Chroma koleksiyonunda en az bir embedding var mi kontrol eder."""
    try:
        n = int(vectorstore._collection.count())  # type: ignore[attr-defined]
        return n > 0
    except Exception:  # noqa: BLE001
        try:
            sample = vectorstore.get(limit=1)
            return bool(sample and sample.get("ids"))
        except Exception:  # noqa: BLE001
            return False


def _docstore_has_keys(docstore) -> bool:
    """Parent docstore'da kayitli en az bir parent_id var mi kontrol eder."""
    try:
        # KV docstore (LocalFileStore tabanli) yield_keys destekler.
        for _ in docstore.yield_keys():
            return True
    except Exception:  # noqa: BLE001
        try:
            return bool(getattr(docstore, "store", {}))
        except Exception:  # noqa: BLE001
            return False
    return False


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
    skip_if_indexed: bool = True,
) -> dict[str, int]:
    """
    Verilen repository dosyalarını parent-child retrieval mimarisiyle indeksler.

    skip_if_indexed=True (varsayilan): Chroma koleksiyonu + docstore zaten dolu ise
    yeniden indekslemez (free-tier embedding kotasini korur). Sayilar 'reused' olarak
    isaretlenir; gercek toplamlar Chroma uzerinden tahmin edilir.

    Logging çıktısı sayesinde her dosyanın kaç parent ve child parçaya ayrıldığı
    konsolda görülebilir.
    """
    repo_root_path = Path(repo_root).resolve()
    totals = {"files": 0, "parents": 0, "children": 0, "reused": 0}

    if skip_if_indexed:
        try:
            vs_has = _vectorstore_has_documents(retriever.vectorstore)
            ds_has = _docstore_has_keys(retriever.docstore)
        except Exception:  # noqa: BLE001
            vs_has, ds_has = False, False
        if vs_has and ds_has:
            try:
                n_children = int(retriever.vectorstore._collection.count())  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                n_children = 0
            n_parents = 0
            try:
                for _ in retriever.docstore.yield_keys():
                    n_parents += 1
            except Exception:  # noqa: BLE001
                n_parents = 0
            totals.update({
                "files": 0,
                "parents": n_parents,
                "children": n_children,
                "reused": 1,
            })
            LOGGER.info(
                "Mevcut indeks yeniden kullaniliyor (parent=%s, child=%s); embedding atlandi.",
                n_parents,
                n_children,
            )
            return totals

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
        # Bazi belgeler splitter sonrasi yine de 0 child uretebilir (tahmin kesin degildir);
        # bu durumda Chroma bos embeddings listesiyle patlar → hatasi yakalayip atliyoruz.
        try:
            retriever.add_documents([document])
        except Exception as add_exc:  # noqa: BLE001
            err_s = str(add_exc).lower()
            if "non-empty" in err_s or ("embedding" in err_s and "upsert" in err_s):
                LOGGER.warning(
                    "%s dosyasi Chroma upsert hatasi nedeniyle atlandi (muhtemelen 0 gecerli child parca): %s",
                    file_path.name,
                    add_exc,
                )
                continue
            raise

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

    Retrieval sonucunda Writer CONTEXT filtresi uygulanir (multi-query ile aynı kurallar).
    """
    LOGGER.info("Retrieval basladi: %s", query)
    documents = retriever.invoke(query)
    filtered: list[Document] = []
    blocked = 0
    for doc in documents:
        if not isinstance(doc, Document):
            continue
        if document_blocked_for_writer_context(doc):
            blocked += 1
            continue
        filtered.append(doc)
    if blocked:
        LOGGER.info(
            "query_context: %s parent dokuman Writer CONTEXT filtresinden elendi.",
            blocked,
        )
    LOGGER.info("Retrieval tamamlandi. %s parent dokuman dondu.", len(filtered))
    return filtered


def heuristic_planner_queries(
    *,
    section_title: str,
    section_goal: str,
    max_queries: int = 6,
) -> list[str]:
    """
    LLM cagirmadan (free-tier dostu) deterministic planner sorgulari uretir.

    Section title kendisi bir sorgu; section goal cumle/cesitli ayraclarla parcalanir;
    cok kisa ve duplikat parcalar elenir. Planner LLM cagrisi olmadigi icin tam makalede
    bolum basina 1 LLM cagrisi (sadece Writer) kalir.
    """
    import re as _re

    queries: list[str] = []
    title = (section_title or "").strip()
    goal = (section_goal or "").strip()

    if title:
        queries.append(title)
    if goal:
        queries.append(goal)

    parts = _re.split(r"[.;:\n]+|,\s+", goal)
    for raw in parts:
        token_text = raw.strip().strip("-•\t ")
        if len(token_text) >= 12:
            queries.append(token_text)

    if title and goal:
        queries.append(f"{title}: {goal[:160]}".strip())

    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    if not deduped:
        deduped = [title or "implementation"]
    return deduped[: max(1, int(max_queries))]


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

    Varsayilan: free-tier kotasini korumak icin LLM cagrisi YAPILMAZ; deterministic
    heuristic_planner_queries kullanilir. Eski davranisi geri istemek icin
    .env icine `GEMINI_PLANNER_LLM=1` koy.

    Gemini tarafinda 503/429 gibi gecici hatalar icin sinirli retry + exponential
    backoff uygulanir (Is 4b: API dayanikliligi).
    """
    use_llm = os.getenv("GEMINI_PLANNER_LLM", "").strip().lower() in {"1", "true", "yes"}
    if not use_llm:
        return heuristic_planner_queries(
            section_title=section_title,
            section_goal=section_goal,
            max_queries=max_queries,
        )

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
                    "getaddrinfo",
                    "connection",
                    "temporary failure",
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
    - Writer CONTEXT filtresi: (1) cocuk metadata `exclude_from_writer_context`,
      (2) parent paketlenmeden once `document_blocked_for_writer_context` — eski indekslerde
      metadata bayragi olmasa bile `agents/ieee_json_writer.py` yolu veya prompt imzasi yakalanir.
    - Parent blokları token bütçesine göre paketler.
    """
    if not planner_queries:
        return []

    excluded_child_hits = 0
    parent_hits: dict[str, tuple[float, int]] = {}
    for query in planner_queries:
        child_hits = retriever.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=top_k_per_query,
        )
        for child_doc, score in child_hits:
            if score < similarity_threshold:
                continue
            if document_blocked_for_writer_context(child_doc):
                excluded_child_hits += 1
                continue
            parent_id = str(child_doc.metadata.get(PARENT_ID_KEY, "")).strip()
            if not parent_id:
                continue
            best_score, seen_count = parent_hits.get(parent_id, (0.0, 0))
            parent_hits[parent_id] = (max(best_score, score), seen_count + 1)

    if excluded_child_hits:
        LOGGER.info(
            "Writer CONTEXT filtresi: %s cocuk isabet exclude_from_writer_context nedeniyle atlandi.",
            excluded_child_hits,
        )

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
    excluded_parents = 0
    for doc in parent_docs:
        if not isinstance(doc, Document):
            continue
        if document_blocked_for_writer_context(doc):
            excluded_parents += 1
            continue
        block = _format_context_block(doc)
        block_tokens = _estimate_tokens(block)
        if used_tokens + block_tokens > max_context_tokens:
            continue
        packed_docs.append(doc)
        used_tokens += block_tokens

    if excluded_parents:
        LOGGER.info(
            "Writer CONTEXT filtresi: %s parent blok paketlenmeden elendi (metadata veya path/icerik).",
            excluded_parents,
        )

    packed_docs = dedupe_parent_documents_by_location(packed_docs)

    LOGGER.info(
        "Multi-query retrieval tamamlandi. %s parent secildi, yaklasik %s token paketlendi.",
        len(packed_docs),
        used_tokens,
    )
    return packed_docs
