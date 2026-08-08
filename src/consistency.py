"""TUTARLILIK MOTORU: regresyon ve sınıflandırmayı tek bir karara bağlar."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, truncnorm

from . import config as C

_EPS = 1e-9

LOG_EDGES = np.log1p(C.SEGMENT_EDGES)


def regression_to_probabilities(mu: np.ndarray, sigma: float) -> np.ndarray:
    """Regresyon tahminini segment olasılıklarına çevirir."""
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    sigma = max(float(sigma), 1e-6)

    cdf = norm.cdf((LOG_EDGES.reshape(1, -1) - mu) / sigma)
    probs = np.diff(cdf, axis=1)

    probs = np.clip(probs, _EPS, None)
    return probs / probs.sum(axis=1, keepdims=True)


def fuse(p_reg: np.ndarray, p_cls: np.ndarray, weight: float) -> np.ndarray:
    """Logaritmik havuzlama ile iki olasılık dağılımını birleştirir."""
    w = float(np.clip(weight, 0.0, 1.0))
    log_p = w * np.log(np.clip(p_reg, _EPS, None)) + \
        (1.0 - w) * np.log(np.clip(p_cls, _EPS, None))
    log_p -= log_p.max(axis=1, keepdims=True)
    p = np.exp(log_p)
    return p / p.sum(axis=1, keepdims=True)


def _standardized_bounds(
    mu: np.ndarray, sigma: float, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """scipy.truncnorm'un beklediği standartlaştırılmış (a, b) sınırları."""
    return (LOG_EDGES[k] - mu) / sigma, (LOG_EDGES[k + 1] - mu) / sigma


def _clip_into_segment(values: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Kayan nokta hatası sınırda bir ihlal yaratmasın diye içeri kırpar."""
    lo_lin, hi_lin = C.SEGMENT_EDGES[k], C.SEGMENT_EDGES[k + 1]
    pad = np.where(np.isfinite(hi_lin), (hi_lin - lo_lin) * 1e-6, 1e-6)
    upper = np.where(np.isfinite(hi_lin), hi_lin - pad, np.inf)
    return np.clip(values, lo_lin + pad, upper)


def is_inside(mu: np.ndarray, segment_idx: np.ndarray) -> np.ndarray:
    """Regresyon tahmini zaten seçilen segmentin içine düşüyor mu?"""
    mu = np.asarray(mu, dtype=float)
    k = np.asarray(segment_idx, dtype=int)
    return (mu >= LOG_EDGES[k]) & (mu < LOG_EDGES[k + 1])


def point_estimate(
    mu: np.ndarray, sigma: float, segment_idx: np.ndarray
) -> np.ndarray:
    """Nihai nokta tahmini: MİNİMAL MÜDAHALE kuralı."""
    mu = np.asarray(mu, dtype=float)
    k = np.asarray(segment_idx, dtype=int)
    raw = np.expm1(mu)
    projected = _projected_mean(mu, sigma, k)
    return _clip_into_segment(np.where(is_inside(mu, k), raw, projected), k)


def _projected_mean(
    mu: np.ndarray, sigma: float, segment_idx: np.ndarray
) -> np.ndarray:
    """Seçilen segmentin aralığına kısıtlanmış koşullu ortalamayı döndürür."""
    mu = np.asarray(mu, dtype=float)
    sigma = max(float(sigma), 1e-6)
    k = np.asarray(segment_idx, dtype=int)
    a, b = _standardized_bounds(mu, sigma, k)

    with np.errstate(invalid="ignore", divide="ignore"):
        cond_mean = truncnorm.mean(a, b, loc=mu, scale=sigma)

    lo, hi = LOG_EDGES[k], LOG_EDGES[k + 1]
    degenerate = ~np.isfinite(cond_mean)
    if np.any(degenerate):
        fallback = np.where(np.isfinite(hi), 0.5 * (lo + np.nan_to_num(hi)), lo + 1.0)
        cond_mean = np.where(degenerate, fallback, cond_mean)

    return _clip_into_segment(np.expm1(cond_mean), k)


def interval(
    mu: np.ndarray, sigma: float, segment_idx: np.ndarray, coverage: float = 0.80
) -> tuple[np.ndarray, np.ndarray]:
    """Seçilen segmente kısıtlanmış merkezi belirsizlik aralığı."""
    mu = np.asarray(mu, dtype=float)
    sigma = max(float(sigma), 1e-6)
    k = np.asarray(segment_idx, dtype=int)
    a, b = _standardized_bounds(mu, sigma, k)

    tail = (1.0 - float(coverage)) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        lo_log = truncnorm.ppf(tail, a, b, loc=mu, scale=sigma)
        hi_log = truncnorm.ppf(1.0 - tail, a, b, loc=mu, scale=sigma)

    lo_lin, hi_lin = C.SEGMENT_EDGES[k], C.SEGMENT_EDGES[k + 1]
    lo = np.where(np.isfinite(lo_log), np.expm1(lo_log), lo_lin)
    hi = np.where(np.isfinite(hi_log), np.expm1(hi_log),
                  np.where(np.isfinite(hi_lin), hi_lin, lo_lin * 2 + 1))

    point = point_estimate(mu, sigma, k)
    lo = np.minimum(lo, point)
    hi = np.maximum(hi, point)
    return _clip_into_segment(lo, k), _clip_into_segment(hi, k)


def fit_fusion_weight(
    p_reg: np.ndarray,
    p_cls: np.ndarray,
    y_true_idx: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    y_true: np.ndarray,
    alpha: float = 0.5,
) -> tuple[float, list[dict[str, float]]]:
    """Doğrulama setinde en iyi füzyon ağırlığını arar."""
    from sklearn.metrics import f1_score

    log_true = np.log1p(np.asarray(y_true, dtype=float))
    sweep: list[dict[str, float]] = []

    for w in C.FUSION_WEIGHT_GRID:
        pred_idx = fuse(p_reg, p_cls, w).argmax(axis=1)
        prices = point_estimate(mu, sigma, pred_idx)
        sweep.append({
            "w": float(w),
            "macro_f1": float(
                f1_score(y_true_idx, pred_idx, average="macro", zero_division=0)
            ),
            "mae_log": float(np.mean(np.abs(log_true - np.log1p(prices)))),
        })

    f1s = np.array([s["macro_f1"] for s in sweep])
    maes = np.array([s["mae_log"] for s in sweep])

    def _norm(v: np.ndarray) -> np.ndarray:
        span = v.max() - v.min()
        return np.zeros_like(v) if span < 1e-12 else (v - v.min()) / span

    combined = alpha * _norm(f1s) + (1.0 - alpha) * (1.0 - _norm(maes))
    for s, c in zip(sweep, combined):
        s["combined"] = float(c)

    best_w = float(sweep[int(np.argmax(combined))]["w"])
    return best_w, sweep


def apply_decision_weights(p: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Sınıf çarpanlarını uygular ve yeniden normalize eder."""
    q = np.clip(p, _EPS, None) * np.asarray(weights, dtype=float).reshape(1, -1)
    return q / q.sum(axis=1, keepdims=True)


def fit_decision_weights(
    p_val: np.ndarray, y_true_idx: np.ndarray
) -> tuple[np.ndarray, float]:
    """macro-F1'i maksimize eden sınıf çarpanlarını koordinat yükselişiyle arar."""
    from sklearn.metrics import f1_score

    def score(c: np.ndarray) -> float:
        pred = apply_decision_weights(p_val, c).argmax(axis=1)
        return float(f1_score(y_true_idx, pred, average="macro", zero_division=0))

    best_c = np.ones(C.N_SEGMENTS)
    best = score(best_c)

    for _ in range(C.DECISION_WEIGHT_ROUNDS):
        improved = False
        for k in range(C.N_SEGMENTS):
            for m in C.DECISION_WEIGHT_GRID:
                trial = best_c.copy()
                trial[k] = m
                s = score(trial)
                if s > best + 1e-9:
                    best, best_c, improved = s, trial, True
        if not improved:
            break

    return best_c / best_c.mean(), best


def verify_consistency(values: np.ndarray, segment_idx: np.ndarray) -> None:
    """Sözleşme kontrolü: tahmin edilen satış, ilan edilen segmentin içinde mi?"""
    from .data import segment_index

    derived = segment_index(values)
    bad = np.flatnonzero(derived != np.asarray(segment_idx))
    if bad.size:
        i = bad[0]
        raise AssertionError(
            f"TUTARSIZLIK: {bad.size} kayıtta segment uyuşmuyor. "
            f"Örnek -> satış={values[i]:.4f}M, ilan edilen segment="
            f"{C.SEGMENT_LABELS[segment_idx[i]]}, fiyattan türeyen="
            f"{C.SEGMENT_LABELS[derived[i]]}"
        )
