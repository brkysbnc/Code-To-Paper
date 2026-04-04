"""
Faz 1 giriş noktası: GitHub'dan kod çekme ve hazırlama.

Eski isim `setup_phase_1` korunur; asıl iş `github_handler.clone_and_prepare` içinde.
Böylece ileride Streamlit / CLI başka modülden de aynı fonksiyonu çağırabilir.
"""

from github_handler import clone_and_prepare


def setup_phase_1(github_url: str, target_dir: str = "data/source") -> None:
    """
    Public GitHub URL'sinden repoyu indirir, gereksiz dosyaları temizler, özet yazar.

    Yapılanlar (github_handler üzerinden):
        - Hedef klasör doluysa silinip yeniden klonlanır.
        - Klon sonrası .git, node_modules, .env, görseller vb. süzülür.
        - data/fetch_info.txt oluşturulur (URL + commit hash).
        - İndekslenebilecek dosya yolları sayılır (chunk aşaması için önizleme).

    Args:
        github_url: Klonlanacak public repo HTTPS adresi.
        target_dir: Kaynak kodun yazılacağı kök (varsayılan data/source).

    Not:
        Ağ / izin hatalarında mesaj yazdırılır; üst seviye API istersen burada
        logging veya özel exception fırlatma eklenebilir.
    """
    try:
        commit, paths = clone_and_prepare(github_url, target_dir)
        print("\nFAZ 1 tamamlandi.")
        print(f"- Kodlar '{target_dir}' altina alindi (commit: {commit[:8]}...).")
        print(f"- Indekslenebilir dosya sayisi: {len(paths)}")
        print("- Detay: data/fetch_info.txt")
    except Exception as e:
        print(f"Klonlama veya hazirlama hatasi: {e}")


# Doğrudan çalıştırıldığında örnek bir public repo ile hızlı duman testi.
if __name__ == "__main__":
    setup_phase_1("https://github.com/pallets/flask.git")
