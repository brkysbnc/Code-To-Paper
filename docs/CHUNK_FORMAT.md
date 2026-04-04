# Chunk veri sözleşmesi (Code-to-Paper)

RAG hattında herkesin uyması gereken tek kaynak format. Makale şablonu (IEEE/ACM/Springer) bundan bağımsızdır.

## Embedding

| Alan | Değer |
|------|--------|
| Model | `text-embedding-3-small` (OpenAI) |
| Boyut | **1536** float |
| Kural | Arama sorgusu da aynı model ile vektörleştirilir; boyut uyuşmazlığı kabul edilmez. |

## Chunk kaydı (dict / metadata)

Backend’in ürettiği her parça şu alanları taşır (eksik alan gönderilmez; yoksa boş string / null politikası tek tip olmalı).

| Alan | Tip | Zorunlu | Açıklama |
|------|-----|---------|----------|
| `id` | string | evet | Kararlı kimlik: aynı içerik tekrar indekslenirse aynı `id` ile güncellenmeli. Öneri: `sha256(f"{source_repo}|{file_path}|{chunk_index}|{content_hash}")` kısaltılmış hex veya tam hash string. |
| `source_repo` | string | evet | Klonlanan repo URL’si veya ekipçe sabitlenen kısa ad (tüm pipeline’da aynı biçim). |
| `file_path` | string | evet | Repo köküne göre göreli yol; ayırıcı olarak `/` kullanılması önerilir. |
| `chunk_index` | int | evet | Aynı dosyada parça sırası: 0, 1, 2, … |
| `text` | string | evet | Embedding’e giren ham metin. |
| `content_type` | string | evet | `code` \| `markdown` \| `config` \| `text` |
| `language` | string | evet | Örn. `python`, `javascript`, `markdown`, `unknown` (uzantı veya basit tespit). |
| `start_line` | int | evet | Kaynak gösterimi için başlangıç satırı (1 tabanlı). |
| `end_line` | int | evet | Bitiş satırı (dahil). |
| `content_hash` | string | evet | `text` (veya dosya+offset) için SHA-256 hex; değişim tespiti. |
| `created_at` | string | evet | ISO 8601 UTC, örn. `2026-04-04T12:00:00Z`. |

Chroma tarafında: `text` → `documents`; diğerleri → `metadatas` (Chroma metadata değerleri str/float/int olmalı; `chunk_index`, satırlar sayı olarak veya string olarak tutulabilir — ekip tek tip seçsin); `embedding` → `embeddings` listesi.

## İndekslenebilir dosyalar

- Kaynak kod uzantıları: `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.java`, `.cs`, `.cpp`, `.c`, `.h`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, `.sql`
- Dokümantasyon / metin: `README*` (uzantılı veya uzantısız), `.md`, `.txt`
- Yapılandırma (isteğe bağlı, hafif): `.yml`, `.yaml`, `.json` (sadece repo kökü veya `config/` altı gibi kurallar sonra sıkılaştırılabilir), `.toml`

## Genelde indeks dışı

`github_handler` süzgeci ile diskten kaldırılan veya indeksleyicinin atlayacağı örnekler: `.git`, `node_modules`, sanal ortamlar, build çıktıları, görseller, `.env`, büyük log/binary.

Bu dosya güncellendiğinde DB ve `rag_core` aynı alan isimlerini kullanmalıdır.
