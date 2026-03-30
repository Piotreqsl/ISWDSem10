import os
from huggingface_hub import hf_hub_download

# Konfiguracja
REPO_ID = "speakleash/bielik-11b-v2.2-instruct-GGUF"
FILENAME = "Bielik-11B-v2.2-Instruct.Q4_K_M.gguf"
LOCAL_DIR = "./local_llm"

def download_bielik():
    print(f"Rozpoczynam inicjalizację pobierania giganta: {FILENAME} ({REPO_ID})...")
    print(f"Plik zostanie bezpiecznie zapisany wyłącznie w folderze lokalnego projektu: {os.path.abspath(LOCAL_DIR)}")
    print("Może to potrwać od kilku do kilkunastu minut w zależności od Twojego łącza (ok 6.5 GB)!")
    
    # Tworzymy absolutnie odseparowany folder, o który prosiłeś (żadnych instalacji macek Ollamy w C:/Users)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False # Wymuszamy fizyczną kopię w tym bezpośrednim repozytorium
        )
        print(f"\\nSUKCES! Twój Polski agent (Bielik) został osadzony ekspercko pod ścieżką: {downloaded_path}")
    except Exception as e:
        print(f"\\nWystąpił błąd pobierania: {e}")

if __name__ == "__main__":
    download_bielik()
