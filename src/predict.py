"""Servis katmanı: eğitilmiş modeli yükler ve tutarlı tahmin üretir."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import consistency as CS


@dataclass
class Prediction:
    """Tek bir oyun için tam tahmin çıktısı."""

    predicted_sales: float
    predicted_units: int
    segment: str
    segment_index: int
    segment_description: str
    confidence: float
    segment_probabilities: dict[str, float]
    interval: tuple[float, float]
    segment_range: tuple[float, float | None]
    drivers: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["interval"] = list(self.interval)
        d["segment_range"] = list(self.segment_range)
        return d


class SalesPredictor:
    """Eğitilmiş modeli sarmalayan, durumsuz tahmin servisi."""

    def __init__(self, bundle_path=None) -> None:
        path = bundle_path or C.BUNDLE_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {path}\nÖnce eğitin: python -m src.train")
        b = joblib.load(path)
        self.fb = b["feature_builder"]
        self.regressors = b["regressors"]
        self.clf = b["classifier"]
        self.sigma: float = b["sigma"]
        self.weight: float = b["fusion_weight"]
        self.decision_weights = np.asarray(
            b.get("decision_weights", np.ones(C.N_SEGMENTS)), dtype=float)
        self.categories: dict[str, list] = b["categories"]
        self.metrics: dict[str, Any] = b.get("metrics", {})

    @property
    def year_min(self) -> int:
        return int(getattr(self.fb, "year_min", C.MIN_YEAR))

    @property
    def year_max(self) -> int:
        return int(getattr(self.fb, "year_max", C.MAX_YEAR))

    @property
    def options(self) -> dict[str, Any]:
        """Arayüzün form alanlarını beslemek için tüm seçenek listeleri."""
        return {
            "platforms": self.fb.known_platforms,
            "genres": self.fb.known_genres,
            "publishers": self.fb.known_publishers,
            "ratings": self.fb.known_ratings,
            "year_range": [self.year_min, self.year_max],
            "medians": {k: round(v, 2) for k, v in self.fb.numeric_medians.items()},
            "rows": self.metrics.get("dataset", {}).get("rows_clean"),
            "n_franchises": self.metrics.get("dataset", {}).get("n_franchises"),
            "accuracy_by_year": self.metrics.get("accuracy_by_year", {}),
            "segments": [
                {
                    "label": lbl,
                    "description": C.SEGMENT_DESCRIPTIONS[lbl],
                    "min": float(C.SEGMENT_EDGES[i]),
                    "max": (None if not np.isfinite(C.SEGMENT_EDGES[i + 1])
                            else float(C.SEGMENT_EDGES[i + 1])),
                }
                for i, lbl in enumerate(C.SEGMENT_LABELS)
            ],
        }

    def _prepare(self, records: Iterable[dict[str, Any]]):
        from .data import franchise_key

        rows, notes = [], []
        med = self.fb.numeric_medians

        for r in records:
            note: list[str] = []
            name = str(r.get("Name", "")).strip()
            platform = str(r.get("Platform", "")).strip()
            genre = str(r.get("Genre", "")).strip()
            publisher = str(r.get("Publisher", "")).strip() or "Unknown"
            rating = str(r.get("Rating", "")).strip() or "Unknown"

            if platform not in self.fb.known_platforms:
                note.append(f"'{platform}' eğitim verisinde yok, tahmin daha belirsiz.")
            if genre not in self.fb.known_genres:
                note.append(f"'{genre}' bilinmeyen bir tür, tahmin daha belirsiz.")
            if publisher not in set(self.fb.known_publishers):
                note.append(f"'{publisher}' eğitim verisinde yok, yeni yayıncı "
                            "olarak değerlendirildi.")
            if rating not in self.fb.known_ratings:
                rating = "Unknown"

            try:
                year = int(r.get("Year_of_Release"))
            except (TypeError, ValueError):
                raise ValueError("Çıkış yılı okunamadı, sayı olarak girin.")
            if not (self.year_min <= year <= self.year_max):
                raise ValueError(
                    f"Çıkış yılı {year} desteklenmiyor. Veri seti yalnızca "
                    f"{self.year_min} ile {self.year_max} arası oyunları "
                    "içeriyor, bu aralıkta bir yıl girin.")

            def opt(key, lo, hi, label):
                v = r.get(key)
                if v is None or v == "":
                    note.append(f"{label} girilmedi, model bu bilgi olmadan "
                                "tahmin yaptı ve belirsizlik daha yüksek.")
                    return np.nan
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    note.append(f"{label} okunamadı, yok sayıldı.")
                    return np.nan
                if not (lo <= v <= hi):
                    raise ValueError(f"{label} {lo} ile {hi} arasında olmalı.")
                return v

            critic = opt("Critic_Score", 0, 100, "Kritik puanı")
            user = opt("User_Score", 0, 10, "Kullanıcı puanı")

            fr = franchise_key(name) if name else "?"
            prior, _, _, _ = self.fb.franchise_history.lookup(fr, year)
            if name and prior > 0:
                note.append(f"'{fr}' serisinde {year} öncesinde {int(prior)} oyun "
                            "bulundu, geçmiş performansı tahmine dahil edildi.")
            elif name:
                note.append(f"'{fr}' serisinin {year} öncesinde kaydı yok, yeni "
                            "IP olarak değerlendirildi.")

            rows.append({
                "Name": name, "Franchise": fr, "Platform": platform,
                "Genre": genre, "Publisher": publisher, "Rating": rating,
                "Year_of_Release": year, "Critic_Score": critic,
                "User_Score": user,
            })
            notes.append(note)

        X = self.fb.transform(pd.DataFrame(rows))
        for col, cats in self.categories.items():
            X[col] = pd.Categorical(X[col].astype(str), categories=cats)
        return X[C.FEATURES], notes

    def _mu(self, X: pd.DataFrame) -> np.ndarray:
        """Tohum ortalamalı regresör tahmini (log uzayında)."""
        return np.mean([m.predict(X) for m in self.regressors], axis=0)

    def _drivers(self, X: pd.DataFrame, i: int) -> list[dict[str, Any]]:
        """Karşı olgusal katkı analizi."""
        base = float(self._mu(X.iloc[[i]])[0])
        out: list[dict[str, Any]] = []

        refs = {
            "Platform": self.fb.known_platforms[0],
            "Genre": self.fb.known_genres[0],
            "Publisher": self.fb.known_publishers[0],
            "Rating": "Unknown",
        }
        for col, ref in refs.items():
            if str(X.iloc[i][col]) == str(ref):
                continue
            counter = X.iloc[[i]].copy()
            counter[col] = pd.Categorical([ref], categories=self.categories[col])
            try:
                alt = float(self._mu(counter)[0])
            except Exception:
                continue
            out.append({"feature": col, "value": str(X.iloc[i][col]), "kind": "cat",
                        "effect_log": round(base - alt, 4),
                        "direction": "artırıcı" if base >= alt else "azaltıcı"})

        for col in C.NUMERIC_FEATURES:
            cur = X.iloc[i][col]
            ref = self.fb.numeric_medians.get(col)
            if ref is None or pd.isna(cur):
                continue
            counter = X.iloc[[i]].copy()
            counter[col] = float(ref)
            try:
                alt = float(self._mu(counter)[0])
            except Exception:
                continue
            out.append({"feature": col, "value": float(cur), "kind": "num",
                        "effect_log": round(base - alt, 4),
                        "direction": "artırıcı" if base >= alt else "azaltıcı"})

        if float(X.iloc[i]["FranchisePriorCount"]) > 0:
            counter = X.iloc[[i]].copy()
            counter["FranchisePriorCount"] = 0.0
            for c in ("FranchisePriorMean", "FranchisePriorMax",
                      "FranchiseYearsSince"):
                counter[c] = np.nan
            alt = float(self._mu(counter)[0])
            out.append({
                "feature": "Franchise", "kind": "cat",
                "value": f"{int(X.iloc[i]['FranchisePriorCount'])} önceki oyun",
                "effect_log": round(base - alt, 4),
                "direction": "artırıcı" if base >= alt else "azaltıcı"})

        return sorted(out, key=lambda d: -abs(d["effect_log"]))[:6]

    def predict(self, records: list[dict[str, Any]],
                explain: bool = True) -> list[Prediction]:
        X, notes = self._prepare(records)

        mu = self._mu(X)
        p_reg = CS.regression_to_probabilities(mu, self.sigma)
        p_cls = self.clf.predict_proba(X)
        p = CS.apply_decision_weights(
            CS.fuse(p_reg, p_cls, self.weight), self.decision_weights)

        seg_idx = p.argmax(axis=1)
        sales = CS.point_estimate(mu, self.sigma, seg_idx)
        CS.verify_consistency(sales, seg_idx)

        lo_q, hi_q = CS.interval(mu, self.sigma, seg_idx, coverage=0.80)
        seg_lo = C.SEGMENT_EDGES[seg_idx]
        seg_hi = C.SEGMENT_EDGES[seg_idx + 1]

        results = []
        for i in range(len(X)):
            k = int(seg_idx[i])
            label = C.SEGMENT_LABELS[k]
            results.append(Prediction(
                predicted_sales=round(float(sales[i]), 4),
                predicted_units=int(round(float(sales[i]) * 1_000_000)),
                segment=label,
                segment_index=k,
                segment_description=C.SEGMENT_DESCRIPTIONS[label],
                confidence=round(float(p[i, k]), 4),
                segment_probabilities={
                    lbl: round(float(p[i, j]), 4)
                    for j, lbl in enumerate(C.SEGMENT_LABELS)},
                interval=(round(float(lo_q[i]), 4), round(float(hi_q[i]), 4)),
                segment_range=(float(seg_lo[i]),
                               None if not np.isfinite(seg_hi[i])
                               else float(seg_hi[i])),
                drivers=self._drivers(X, i) if explain else [],
                notes=notes[i],
            ))
        return results

    def predict_one(self, **kwargs: Any) -> Prediction:
        return self.predict([kwargs])[0]


_PREDICTOR: SalesPredictor | None = None


def get_predictor() -> SalesPredictor:
    """Süreç başına tek örnek, böylece her istekte model yüklenmez."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = SalesPredictor()
    return _PREDICTOR


if __name__ == "__main__":
    p = get_predictor()
    demo = [
        {"Name": "Call of Duty: Advanced Warfare", "Platform": "PS4",
         "Genre": "Shooter", "Publisher": "Activision", "Rating": "M",
         "Year_of_Release": 2014, "Critic_Score": 83, "User_Score": 5.7},
        {"Name": "Tamamen Yeni Bir Oyun", "Platform": "PS4",
         "Genre": "Shooter", "Publisher": "Activision", "Rating": "M",
         "Year_of_Release": 2014, "Critic_Score": 83, "User_Score": 5.7},
        {"Name": "Kucuk Bagimsiz Oyun", "Platform": "PC", "Genre": "Strategy",
         "Publisher": "Unknown", "Rating": "Unknown",
         "Year_of_Release": 2012, "Critic_Score": 55, "User_Score": 6.0},
        {"Name": "Mario Kart 9", "Platform": "Wii", "Genre": "Racing",
         "Publisher": "Nintendo", "Rating": "E",
         "Year_of_Release": 2012, "Critic_Score": 87, "User_Score": 8.5},
    ]
    for rec, res in zip(demo, p.predict(demo)):
        print(f"\n{rec['Name']} ({rec['Platform']}, {rec['Year_of_Release']}, "
              f"kritik {rec['Critic_Score']})")
        print(f"  -> {res.predicted_sales:.3f}M adet | segment: {res.segment} "
              f"(güven %{res.confidence * 100:.0f})")
        print(f"     aralık: {res.interval[0]:.3f}M - {res.interval[1]:.3f}M")
        for d in res.drivers[:3]:
            print(f"     {d['feature']:22} {str(d['value'])[:16]:>16}  "
                  f"{d['effect_log']:+.3f}")
