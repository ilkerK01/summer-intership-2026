"""Oyun satış veri setini data/games.csv olarak indirir."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TARGET = DATA_DIR / "games.csv"

DATASET = "rush4ratio/video-game-sales-with-ratings"
SOURCE_FILE = "Video_Games_Sales_as_at_22_Dec_2016.csv"

MANUAL = f"""
Otomatik indirme başarısız oldu.

Manuel adımlar:
  1. https://www.kaggle.com/datasets/{DATASET} adresine gidin (giriş yapın)
  2. "Download" ile arşivi indirip açın
  3. İçindeki '{SOURCE_FILE}' dosyasını şuraya kopyalayın:
     {TARGET}
     (dosya adını games.csv olarak değiştirin)

Kaggle API ile alternatif:
  pip install kaggle
  kaggle datasets download -d {DATASET} -p "{DATA_DIR}" --unzip
"""


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET.exists():
        print(f"[get_data] Zaten mevcut: {TARGET}")
        return 0

    try:
        import kagglehub

        print(f"[get_data] kagglehub ile indiriliyor: {DATASET}")
        path = Path(kagglehub.dataset_download(DATASET))

        found = next((p for p in path.rglob("*.csv") if p.name == SOURCE_FILE), None)
        if found is None:
            candidates = list(path.rglob("*.csv"))
            if len(candidates) == 1:
                found = candidates[0]
            else:
                print(f"[get_data] Arşivde '{SOURCE_FILE}' bulunamadı.")
                print("[get_data] Arşivdeki dosyalar:", [p.name for p in candidates])
                print(MANUAL)
                return 1

        shutil.copy(found, TARGET)
        print(f"[get_data] Tamam -> {TARGET}")
        return 0
    except Exception as exc:
        print(f"[get_data] Hata: {type(exc).__name__}: {exc}")
        print(MANUAL)
        return 1


if __name__ == "__main__":
    sys.exit(main())
