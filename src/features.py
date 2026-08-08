"""Özellik mühendisliği."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


class FranchiseHistory:
    """Zaman güvenli seri geçmişi arama tablosu."""

    def __init__(self) -> None:
        self._table: dict[str, tuple] = {}

    def fit(self, df: pd.DataFrame) -> "FranchiseHistory":
        """df: Franchise, Year_of_Release ve hedef kolonunu içeren eğitim verisi."""
        tmp = pd.DataFrame({
            "Franchise": df["Franchise"].to_numpy(),
            "Year": df["Year_of_Release"].to_numpy(dtype=int),
            "y": np.log1p(df[C.TARGET].to_numpy(dtype=float)),
        })
        per_year = (
            tmp.groupby(["Franchise", "Year"])["y"]
            .agg(n="count", total="sum", top="max")
            .reset_index()
            .sort_values(["Franchise", "Year"])
        )

        self._table = {}
        for fr, g in per_year.groupby("Franchise", sort=False):
            self._table[str(fr)] = (
                g["Year"].to_numpy(dtype=int),
                g["n"].to_numpy(dtype=float).cumsum(),
                g["total"].to_numpy(dtype=float).cumsum(),
                np.maximum.accumulate(g["top"].to_numpy(dtype=float)),
            )
        return self

    def lookup(self, franchise: str, year: int) -> tuple[float, float, float, float]:
        """(adet, ortalama, maksimum, son oyundan bu yana geçen yıl)."""
        entry = self._table.get(str(franchise))
        if entry is None:
            return 0.0, np.nan, np.nan, np.nan

        years, cum_n, cum_sum, cum_max = entry
        i = int(np.searchsorted(years, year, side="left")) - 1
        if i < 0:
            return 0.0, np.nan, np.nan, np.nan

        return (
            float(cum_n[i]),
            float(cum_sum[i] / cum_n[i]),
            float(cum_max[i]),
            float(year - years[i]),
        )

    def transform(self, franchises, years) -> pd.DataFrame:
        rows = [self.lookup(f, int(y)) for f, y in zip(franchises, years)]
        return pd.DataFrame(rows, columns=C.FRANCHISE_FEATURES)

    @property
    def n_franchises(self) -> int:
        return len(self._table)


class FeatureBuilder:
    """Eğitim setinden öğrenilen sözlükleri tutan, stateful feature üretici."""

    def __init__(self) -> None:
        self.franchise_history = FranchiseHistory()
        self.known_platforms: list[str] = []
        self.known_genres: list[str] = []
        self.known_publishers: list[str] = []
        self.known_ratings: list[str] = []
        self.numeric_medians: dict[str, float] = {}
        self.year_min: int = C.MIN_YEAR
        self.year_max: int = C.MAX_YEAR
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        self.franchise_history.fit(df)

        self.known_platforms = sorted(df["Platform"].unique().tolist())
        self.known_genres = sorted(df["Genre"].unique().tolist())
        self.known_publishers = df["Publisher"].value_counts().index.tolist()

        order = ["E", "EC", "E10+", "T", "M", "AO", "RP", "K-A", "Unknown"]
        present = set(df["Rating"].unique())
        self.known_ratings = ([r for r in order if r in present]
                              + sorted(present - set(order)))

        self.numeric_medians = {
            c: float(df[c].median()) for c in C.NUMERIC_FEATURES
        }
        self.year_min = int(df["Year_of_Release"].min())
        self.year_max = int(df["Year_of_Release"].max())
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ham girdi -> model feature matrisi."""
        if not self._fitted:
            raise RuntimeError("FeatureBuilder.fit() önce çağrılmalı.")

        out = pd.DataFrame(index=df.index)

        for col in C.CATEGORICAL_FEATURES:
            out[col] = df[col].astype(str)

        for col in C.NUMERIC_FEATURES:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        if "Franchise" in df.columns:
            franchises = df["Franchise"].astype(str)
        else:
            from .data import franchise_key
            franchises = df["Name"].astype(str).map(franchise_key)

        years = out["Year_of_Release"].fillna(self.year_max).to_numpy()
        hist = self.franchise_history.transform(franchises.to_numpy(), years)
        for col in C.FRANCHISE_FEATURES:
            out[col] = hist[col].to_numpy()

        for col in C.CATEGORICAL_FEATURES:
            out[col] = out[col].astype("category")

        return out[C.FEATURES]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
