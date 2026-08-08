"""Veri yükleme, temizleme ve segment etiketleme."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SEQUEL_NUM = re.compile(r"\b(?:[ivx]{1,6}|\d{1,4})\b")


def franchise_key(name: str) -> str:
    """Oyun adından seri (franchise) anahtarı çıkarır."""
    s = str(name).lower().split(":")[0].split(" - ")[0]
    s = _SEQUEL_NUM.sub(" ", _NON_ALNUM.sub(" ", s))
    tokens = [t for t in s.split() if t]
    return " ".join(tokens[:3]) if tokens else "?"


def segment_index(sales: np.ndarray | pd.Series) -> np.ndarray:
    """Satış miktarını (milyon adet) segment indeksine (0..3) çevirir."""
    values = np.asarray(sales, dtype=float)
    return np.digitize(values, C.SEGMENT_EDGES[1:-1], right=False).astype(int)


def segment_label(sales: np.ndarray | pd.Series) -> np.ndarray:
    """Satış miktarını okunabilir segment adına çevirir."""
    return np.array(C.SEGMENT_LABELS, dtype=object)[segment_index(sales)]


def load_raw(csv_path=None) -> pd.DataFrame:
    """Ham CSV'yi okur. Dosya yoksa açıklayıcı bir hata verir."""
    path = csv_path or C.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {path}\n"
            "Çözüm: `python -m scripts.get_data` çalıştırın veya "
            f"'{C.KAGGLE_FILE}' dosyasını Kaggle'dan indirip data/ klasörüne "
            "`games.csv` adıyla koyun.\n"
            f"https://www.kaggle.com/datasets/{C.KAGGLE_DATASET}"
        )
    return pd.read_csv(path, low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Temizlik, tip düzeltme, seri çıkarımı ve segment etiketi."""
    df = df.copy()

    df = df.drop(columns=[c for c in C.LEAKY_COLUMNS if c in df.columns])

    df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")
    df["Critic_Score"] = pd.to_numeric(df["Critic_Score"], errors="coerce")
    df["Year_of_Release"] = pd.to_numeric(df["Year_of_Release"], errors="coerce")
    df[C.TARGET] = pd.to_numeric(df[C.TARGET], errors="coerce")

    df = df.dropna(subset=["Name", "Platform", "Genre", "Year_of_Release", C.TARGET])
    df["Year_of_Release"] = df["Year_of_Release"].astype(int)

    df = df[(df["Year_of_Release"] >= C.MIN_YEAR) &
            (df["Year_of_Release"] <= C.MAX_YEAR)]
    df = df[df[C.TARGET] > 0]

    for col in ["Publisher", "Developer", "Rating"]:
        df[col] = df[col].fillna("Unknown").replace({"N/A": "Unknown"})
        df[col] = df[col].astype(str).str.strip()
    df["Platform"] = df["Platform"].astype(str).str.strip()
    df["Genre"] = df["Genre"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["Name", "Platform", "Year_of_Release"])

    if C.REQUIRE_CRITIC_SCORE:
        df = df[df["Critic_Score"].notna()]

    df["Franchise"] = df["Name"].map(franchise_key)

    df["SegmentIdx"] = segment_index(df[C.TARGET])
    df["Segment"] = np.array(C.SEGMENT_LABELS, dtype=object)[df["SegmentIdx"]]

    return df.reset_index(drop=True)


def load_clean(csv_path=None) -> pd.DataFrame:
    """`load_raw` + `clean` kısayolu."""
    return clean(load_raw(csv_path))


def assert_no_leakage(X: pd.DataFrame) -> None:
    """Eğitim öncesi son savunma hattı."""
    forbidden = (C.LEAKY_COLUMNS + C.POST_HOC_COLUMNS
                 + [C.TARGET, "Name", "Segment", "SegmentIdx"])
    leaked = [c for c in forbidden if c in X.columns]
    if leaked:
        raise ValueError(
            f"VERİ SIZINTISI: şu kolonlar feature matrisinde olmamalı: {leaked}")

    unexpected = [c for c in X.columns if c not in C.FEATURES]
    if unexpected:
        raise ValueError(f"Beklenmeyen kolon(lar): {unexpected}")
