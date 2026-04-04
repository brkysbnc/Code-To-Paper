"""
GitHub public repo klonlama ve RAG öncesi dosya ağacını süzme.

Code-to-Paper projesinde vektör indeksine girmeyecek gereksiz verileri (ör. .git,
node_modules, .env, görseller, derleme çıktıları) kaldırır. Böylece embedding ve
Chroma'ya sadece anlamlı kaynak ve dokümantasyon dosyaları kalır.

İlgili tek kaynak formatı: docs/CHUNK_FORMAT.md
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Set

from git import Repo

# --- Süzgeç sabitleri: hangi dizin / dosya türleri RAG için genelde gereksiz ---

# Bu isimdeki klasörlerin tamamı silinir (içindeki binlerce dosyaya tek tek bakılmaz).
_REMOVE_DIR_NAMES: Set[str] = {
    ".git",  # sürüm geçmişi; commit bilgisi zaten fetch_info.txt'te
    "node_modules",  # NPM bağımlılıkları; indeks şişer
    "__pycache__",  # Python bytecode önbelleği
    ".venv",
    "venv",  # sanal ortam klasörleri
    "dist",
    "build",  # paketleme / derleme çıktıları
    "target",  # Maven / Rust vb.
    ".idea",
    ".vscode",  # IDE ayarları
    ".next",  # Next.js build
    "coverage",
    "htmlcov",  # test coverage raporları
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "eggs",
    ".eggs",
}

# Uzantısı bu listede olan dosyalar silinir (görsel, arşiv, binary, log vb.).
_REMOVE_EXTENSIONS: Set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".bmp",
    ".svg",
    ".mp4",
    ".mp3",
    ".wav",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".pdf",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".pyc",
    ".pyo",
    ".log",
}

# Dosya adı tam olarak eşleşirse silinir (özellikle gizli anahtar içeren .env dosyaları).
_REMOVE_EXACT_FILENAMES: Set[str] = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "thumbs.db",
    ".ds_store",
}


def _should_remove_dir(dir_name: str) -> bool:
    """
    Verilen dizin adının tamamen silinip silinmeyeceğini döndürür.

    Python paketlerinde görülen `something.egg-info` klasörleri de indeks için gereksiz
    olduğundan suffix ile yakalanır.
    """
    lower = dir_name.lower()
    if lower in _REMOVE_DIR_NAMES:
        return True
    if lower.endswith(".egg-info"):
        return True
    return False


def _should_remove_file(path: Path) -> bool:
    """
    Tek bir dosya yolunun süzgeç kurallarına göre silinip silinmeyeceğini döndürür.

    Önce tam dosya adı (.env gibi), sonra uzantı kontrol edilir.
    """
    name_lower = path.name.lower()
    if name_lower in _REMOVE_EXACT_FILENAMES:
        return True
    suffix = path.suffix.lower()
    if suffix in _REMOVE_EXTENSIONS:
        return True
    return False


def sanitize_cloned_repo(root: str | Path) -> None:
    """
    Klonlanmış repo kökünde dolaşarak gereksiz dizin ve dosyaları diskten kaldırır.

    İki aşama:
    1) Silinecek klasörleri toplayıp derinlikten bağımsız güvenli şekilde rmtree
       (önce en derin yol, böylece üst klasör silinirken alt yol hâlâ var olmaz).
    2) Kalan ağaçta uzantı/ad kurallarına uyan tekil dosyaları unlink.

    README ve kaynak kod dosyaları bu kurallarda hedeflenmediği sürece korunur.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Geçersiz kök: {root_path}")

    # 1) Silinecek dizin yollarını topla; os.walk içinde dirnames'i budayarak
    #    node_modules gibi dev ağaçların içine inmeyi atlarız (performans).
    dirs_to_nuke: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root_path, topdown=True):
        base = Path(dirpath)
        for d in list(dirnames):
            if _should_remove_dir(d):
                dirs_to_nuke.append(base / d)
        dirnames[:] = [d for d in dirnames if not _should_remove_dir(d)]

    # En uzun yolu önce sil: iç içe iki "build" gibi edge case'lerde sıra önemli olabilir.
    for p in sorted(dirs_to_nuke, key=lambda x: len(x.parts), reverse=True):
        shutil.rmtree(p, ignore_errors=True)

    # 2) Kalan dosyalarda gereksiz uzantı / isim eşleşmesi olanları tek tek sil.
    for dirpath, _, filenames in os.walk(root_path):
        for fn in filenames:
            fp = Path(dirpath) / fn
            if fp.is_file() and _should_remove_file(fp):
                try:
                    fp.unlink()
                except OSError:
                    pass


def iter_indexable_files(
    root: str | Path,
    allowed_extensions: Iterable[str] | None = None,
) -> list[Path]:
    """
    Süzme sonrası embedding / chunk aşamasına aday dosyaların tam yol listesini döndürür.

    - README* dosyaları uzantı olmasa da (örn. README) dahil edilir.
    - allowed_extensions verilmezse docs/CHUNK_FORMAT.md ile uyumlu varsayılan uzantılar kullanılır.
    - Göreli path gerekiyorsa: fp.relative_to(Path(root).resolve())
    """
    root_path = Path(root).resolve()
    default_exts = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".cs",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".sql",
        ".md",
        ".txt",
        ".yml",
        ".yaml",
        ".json",
        ".toml",
        ".xml",
    }
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (allowed_extensions or default_exts)}

    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if not _should_remove_dir(d)]
        for fname in filenames:
            fp = Path(dirpath) / fname
            if not fp.is_file():
                continue
            if _should_remove_file(fp):
                continue
            lower = fp.name.lower()
            if lower.startswith("readme"):
                out.append(fp)
                continue
            if fp.suffix.lower() in exts:
                out.append(fp)
    return sorted(out)


def clone_public_repo(
    github_url: str,
    target_dir: str = "data/source",
    *,
    sanitize: bool = True,
) -> str:
    """
    Public GitHub deposunu belirtilen klasöre klonlar.

    Args:
        github_url: HTTPS clone URL (örn. https://github.com/org/repo.git).
        target_dir: Klonun yazılacağı yer; doluysa önce temizlenir.
        sanitize: True ise klon sonrası sanitize_cloned_repo ile süzme uygulanır.

    Returns:
        HEAD commit'inin tam SHA-1 hex string'i.

    Side effect:
        data/fetch_info.txt içine URL ve commit özeti yazılır (izlenebilirlik).
    """
    target_path = Path(target_dir)
    if target_path.exists() and any(target_path.iterdir()):
        shutil.rmtree(target_path, ignore_errors=True)
    target_path.mkdir(parents=True, exist_ok=True)

    repo = Repo.clone_from(github_url, str(target_path))
    commit_hash = repo.head.commit.hexsha

    if sanitize:
        sanitize_cloned_repo(target_path)

    info_path = Path("data") / "fetch_info.txt"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("=== PROJE KAYNAK KODU BİLGİSİ ===\n")
        f.write(f"Kaynak URL: {github_url}\n")
        f.write(f"Commit Hash: {commit_hash}\n")
        f.write("Not: Klon sonrası gereksiz dosyalar github_handler ile süzülmüştür.\n")

    return commit_hash


def clone_and_prepare(
    github_url: str,
    target_dir: str = "data/source",
) -> tuple[str, list[Path]]:
    """
    Tek çağrıda: klon + süzme + indekslenebilir dosya listesi.

    RAG pipeline'ın bir sonraki adımı (okuma, chunk, embedding) bu Path listesi
    üzerinde dönebilir.

    Returns:
        (commit_hash, indexable_absolute_paths) ikilisi.
    """
    commit = clone_public_repo(github_url, target_dir, sanitize=True)
    paths = iter_indexable_files(target_dir)
    return commit, paths
