import os
import shutil
from git import Repo

def setup_phase_1(github_url, target_dir="data/source"):

    os.makedirs(target_dir, exist_ok=True)
    if os.listdir(target_dir):
        print(f"Eski kodlar temizleniyor: {target_dir}")
        shutil.rmtree(target_dir)
        os.makedirs(target_dir)

    print(f"Repo indiriliyor...\nKaynak: {github_url}\nHedef: {target_dir}")

    try:
        repo = Repo.clone_from(github_url, target_dir)
        commit_hash = repo.head.commit.hexsha
        
        info_path = os.path.join("data", "fetch_info.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write("=== PROJE KAYNAK KODU BİLGİSİ ===\n")
            f.write(f"Kaynak URL: {github_url}\n")
            f.write(f"Commit Hash: {commit_hash}\n")
            f.write("Not: Bu klasördeki kodlar RAG işlemi için otomatik çekilmiştir.\n")
            
        gitignore_path = ".gitignore"
        ignore_rule = "data/\n" 
        
        if not os.path.exists(gitignore_path):
            with open(gitignore_path, "w") as f:
                f.write(ignore_rule)
        else:
            with open(gitignore_path, "r") as f:
                content = f.read()
            if ignore_rule.strip() not in content:
                with open(gitignore_path, "a") as f:
                    f.write(f"\n{ignore_rule}")

        print("\n✅ FAZ 1 TAMAMLANDI!")
        print(f"- Kodlar '{target_dir}' klasörüne çekildi.")
        print(f"- Commit bilgisi '{info_path}' dosyasına yazıldı.")
        print("- .gitignore güncellendi.")

    except Exception as e:
        print(f"❌ Klonlama sırasında hata oluştu: {str(e)}")

# Test Etmek İçin GitHub'dan Flask pallet dosyasını çektim:
if __name__ == "__main__":
    test_url = "https://github.com/pallets/flask.git" 
    setup_phase_1(test_url)