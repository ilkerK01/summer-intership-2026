"""Eğitim betiği: iki model, kalibrasyon, tutarlılık doğrulaması."""

from __future__ import annotations

import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from . import config as C
from . import consistency as CS
from .data import assert_no_leakage, load_clean, segment_index
from .features import FeatureBuilder


def _callbacks():
    return [early_stopping(C.EARLY_STOPPING_ROUNDS, verbose=False),
            log_evaluation(0)]


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def _align(X: pd.DataFrame, categories: dict[str, list]) -> pd.DataFrame:
    """Kategori seviyelerini eğitimdekiyle birebir hizala."""
    X = X.copy()
    for col, cats in categories.items():
        X[col] = pd.Categorical(X[col].astype(str), categories=cats)
    return X[C.FEATURES]


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Hem doğrusal hem log uzayında metrik."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-6)
    log_true, log_pred = np.log1p(y_true), np.log1p(y_pred)
    ape = np.abs(y_pred - y_true) / y_true * 100.0
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(ape)),
        "MedAPE": float(np.median(ape)),
        "MAE_log": float(mean_absolute_error(log_true, log_pred)),
        "RMSE_log": float(np.sqrt(mean_squared_error(log_true, log_pred))),
        "R2_log": float(r2_score(log_true, log_pred)),
    }


def _calibration_table(conf: np.ndarray, hit: np.ndarray) -> dict[str, Any]:
    """Arayüzde gösterilen güven yüzdeleri gerçekten tutuyor mu?"""
    bins = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
            (0.8, 0.9), (0.9, 1.01)]
    rows = []
    for lo, hi in bins:
        m = (conf >= lo) & (conf < hi)
        if m.sum() < 15:
            continue
        rows.append({
            "bin": f"{lo:.0%}-{min(hi, 1.0):.0%}",
            "n": int(m.sum()),
            "stated": float(conf[m].mean()),
            "observed": float(hit[m].mean()),
            "gap": float(hit[m].mean() - conf[m].mean()),
        })
    return {
        "bins": rows,
        "overall_stated": float(conf.mean()),
        "overall_observed": float(hit.mean()),
        "overall_gap": float(hit.mean() - conf.mean()),
        "max_abs_gap": float(max((abs(r["gap"]) for r in rows), default=0.0)),
    }


def _fit_regressors(X_tr, y_log_tr, X_val, y_log_val) -> list[LGBMRegressor]:
    """N_SEED_MODELS tane farklı tohumlu regresör eğitip liste döndürür."""
    models = []
    for i in range(max(1, C.N_SEED_MODELS)):
        params = dict(C.LGBM_REGRESSOR_PARAMS)
        params["random_state"] = C.RANDOM_STATE + i * 101
        m = LGBMRegressor(**params)
        m.fit(X_tr, y_log_tr, eval_set=[(X_val, y_log_val)], eval_metric="l2",
              categorical_feature=C.CATEGORICAL_FEATURES, callbacks=_callbacks())
        models.append(m)
    return models


def _predict_mu(models: list[LGBMRegressor], X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


def _baselines(X_tr, y_tr, X_te, y_te, seg_tr, seg_te) -> dict[str, Any]:
    """Karşılaştırma tabanı, LightGBM'in değer kattığını göstermek için."""
    out: dict[str, Any] = {}

    dummy = DummyRegressor(strategy="median").fit(
        X_tr[["Year_of_Release"]], y_tr)
    out["dummy_median_regressor"] = _regression_metrics(
        y_te, dummy.predict(X_te[["Year_of_Release"]]))

    num = C.NUMERIC_FEATURES
    med = np.nanmedian(X_tr[num].to_numpy(float), axis=0)
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                        sparse_output=False)
    Ztr = np.hstack([enc.fit_transform(X_tr[C.CATEGORICAL_FEATURES].astype(str)),
                     np.where(np.isnan(X_tr[num].to_numpy(float)), med,
                              X_tr[num].to_numpy(float))])
    Zte = np.hstack([enc.transform(X_te[C.CATEGORICAL_FEATURES].astype(str)),
                     np.where(np.isnan(X_te[num].to_numpy(float)), med,
                              X_te[num].to_numpy(float))])
    ridge = Ridge(alpha=1.0).fit(Ztr, np.log1p(y_tr))
    out["ridge_onehot_regressor"] = _regression_metrics(
        y_te, np.expm1(ridge.predict(Zte)))

    dcls = DummyClassifier(strategy="most_frequent").fit(
        X_tr[["Year_of_Release"]], seg_tr)
    pred = dcls.predict(X_te[["Year_of_Release"]])
    out["dummy_classifier"] = {
        "accuracy": float(accuracy_score(seg_te, pred)),
        "macro_f1": float(f1_score(seg_te, pred, average="macro",
                                   zero_division=0)),
    }
    return out


def run_protocol(df, idx_tr, idx_val, idx_te, name, verbose=True):
    """Verilen bölmeyle eğitir ve değerlendirir. -> (artefaktlar, metrikler)"""
    y = df[C.TARGET].to_numpy(dtype=float)
    seg = df["SegmentIdx"].to_numpy(dtype=int)

    def say(m):
        if verbose:
            _log(m)

    say(f"--- {name} --- eğitim={len(idx_tr)}, kalibrasyon={len(idx_val)}, "
        f"test={len(idx_te)}")

    fb = FeatureBuilder().fit(df.iloc[idx_tr])
    X_all = fb.transform(df)
    assert_no_leakage(X_all)

    categories = {c: X_all[c].cat.categories.tolist()
                  for c in C.CATEGORICAL_FEATURES}
    X_all = _align(X_all, categories)

    X_tr, X_val, X_te = X_all.iloc[idx_tr], X_all.iloc[idx_val], X_all.iloc[idx_te]
    y_tr, y_val, y_te = y[idx_tr], y[idx_val], y[idx_te]
    seg_tr, seg_val, seg_te = seg[idx_tr], seg[idx_val], seg[idx_te]

    regs = _fit_regressors(X_tr, np.log1p(y_tr), X_val, np.log1p(y_val))
    clf = LGBMClassifier(**C.LGBM_CLASSIFIER_PARAMS)
    clf.fit(X_tr, seg_tr, eval_set=[(X_val, seg_val)],
            eval_metric="multi_logloss",
            categorical_feature=C.CATEGORICAL_FEATURES, callbacks=_callbacks())
    say(f"  en iyi iterasyon: regresör={[m.best_iteration_ for m in regs]}, "
        f"sınıflandırıcı={clf.best_iteration_}")

    mu_val = _predict_mu(regs, X_val)
    sigma = float(np.std(np.log1p(y_val) - mu_val, ddof=1))
    p_reg_val = CS.regression_to_probabilities(mu_val, sigma)
    p_cls_val = clf.predict_proba(X_val)
    weight, sweep = CS.fit_fusion_weight(
        p_reg_val, p_cls_val, seg_val, mu_val, sigma, y_val, alpha=C.FUSION_ALPHA)
    p_fused_val = CS.fuse(p_reg_val, p_cls_val, weight)
    dec_w, dec_f1 = CS.fit_decision_weights(p_fused_val, seg_val)
    say(f"  sigma={sigma:.4f} | füzyon w={weight:.2f} | "
        f"karar ağırlıkları={np.round(dec_w, 2).tolist()}")

    by_w = {s["w"]: s for s in sweep}
    for tag, wv in (("saf sınıflandırıcı", 0.0), ("seçilen füzyon", weight),
                    ("saf regresyon", 1.0)):
        s = by_w[wv]
        say(f"    {tag:>18} (w={wv:.2f}) -> macro-F1={s['macro_f1']:.4f}  "
            f"MAE_log={s['mae_log']:.4f}")

    mu_te = _predict_mu(regs, X_te)
    p_reg_te = CS.regression_to_probabilities(mu_te, sigma)
    p_cls_te = clf.predict_proba(X_te)
    p_plain = CS.fuse(p_reg_te, p_cls_te, weight)
    p_final = CS.apply_decision_weights(p_plain, dec_w)

    seg_pred = p_final.argmax(axis=1)
    sales_pred = CS.point_estimate(mu_te, sigma, seg_pred)
    CS.verify_consistency(sales_pred, seg_pred)

    raw_sales = np.expm1(mu_te)
    conflict = float((segment_index(raw_sales) != p_cls_te.argmax(axis=1)).mean())

    metrics: dict[str, Any] = {
        "protocol": name,
        "split": {"train": len(idx_tr), "valid": len(idx_val), "test": len(idx_te)},
        "regression": {
            "raw_lightgbm": _regression_metrics(y_te, raw_sales),
            "after_fusion": _regression_metrics(y_te, sales_pred),
        },
        "classification": {
            "accuracy": float(accuracy_score(seg_te, seg_pred)),
            "macro_f1": float(f1_score(seg_te, seg_pred, average="macro",
                                       zero_division=0)),
            "weighted_f1": float(f1_score(seg_te, seg_pred, average="weighted",
                                          zero_division=0)),
            "adjacent_accuracy": float(np.mean(np.abs(seg_pred - seg_te) <= 1)),
            "argmax_only_macro_f1": float(f1_score(
                seg_te, p_plain.argmax(axis=1), average="macro", zero_division=0)),
            "argmax_only_accuracy": float(accuracy_score(
                seg_te, p_plain.argmax(axis=1))),
            "classifier_only_macro_f1": float(f1_score(
                seg_te, p_cls_te.argmax(axis=1), average="macro", zero_division=0)),
            "regression_only_macro_f1": float(f1_score(
                seg_te, p_reg_te.argmax(axis=1), average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(
                seg_te, seg_pred, labels=list(range(C.N_SEGMENTS))).tolist(),
            "report": classification_report(
                seg_te, seg_pred, labels=list(range(C.N_SEGMENTS)),
                target_names=C.SEGMENT_LABELS, zero_division=0, output_dict=True),
        },
        "calibration": _calibration_table(p_final.max(axis=1), seg_pred == seg_te),
        "consistency": {
            "violations_after_fusion": 0,
            "naive_two_model_conflict_rate": conflict,
        },
        "fusion": {
            "weight": weight,
            "alpha": C.FUSION_ALPHA,
            "sigma_log": sigma,
            "decision_weights": dec_w.tolist(),
            "decision_weights_valid_macro_f1": dec_f1,
            "weight_sweep": sweep,
        },
        "baselines": _baselines(X_tr, y_tr, X_te, y_te, seg_tr, seg_te),
        "feature_importance": dict(sorted(
            zip(C.FEATURES,
                np.mean([m.feature_importances_ for m in regs], axis=0)
                .astype(int).tolist()),
            key=lambda kv: -kv[1])),
    }

    cls = metrics["classification"]
    reg = metrics["regression"]["after_fusion"]
    say(f"  TEST | segment acc={cls['accuracy']:.4f}  "
        f"macro-F1={cls['macro_f1']:.4f}  ±1={cls['adjacent_accuracy']:.4f}")
    say(f"  TEST | satış R2_log={reg['R2_log']:.4f}  MedAPE={reg['MedAPE']:.1f}%  "
        f"MAE={reg['MAE']:.3f}M")
    say(f"  TEST | kalibrasyon sapması={metrics['calibration']['overall_gap']:+.4f}")
    say(f"  Referans: füzyonsuz çelişki oranı = %{conflict * 100:.1f}")

    artifacts = {
        "feature_builder": fb, "regressors": regs, "classifier": clf,
        "sigma": sigma, "fusion_weight": weight, "decision_weights": dec_w,
        "categories": categories,
    }
    return artifacts, metrics


def _cross_validate(df) -> dict[str, Any]:
    """5 katlı CV. Her katta feature'lar o katın eğitim payından öğrenilir."""
    y = df[C.TARGET].to_numpy(dtype=float)
    kf = KFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=7)
    per_fold = []

    for tr, te in kf.split(df):
        inner = int(len(tr) * 0.82)
        tr_i, va_i = tr[:inner], tr[inner:]

        fb = FeatureBuilder().fit(df.iloc[tr_i])
        X = fb.transform(df)
        cats = {c: X[c].cat.categories.tolist() for c in C.CATEGORICAL_FEATURES}
        X = _align(X, cats)

        regs = _fit_regressors(X.iloc[tr_i], np.log1p(y[tr_i]),
                               X.iloc[va_i], np.log1p(y[va_i]))
        pred = np.expm1(_predict_mu(regs, X.iloc[te]))
        per_fold.append(_regression_metrics(y[te], pred))

    keys = ["R2_log", "R2", "MAE", "MedAPE"]
    return {
        "folds": C.CV_FOLDS,
        "mean": {k: float(np.mean([f[k] for f in per_fold])) for k in keys},
        "std": {k: float(np.std([f[k] for f in per_fold])) for k in keys},
        "per_fold_R2_log": [round(f["R2_log"], 4) for f in per_fold],
    }


def main() -> dict[str, Any]:
    t0 = time.time()

    df = load_clean()
    seg = df["SegmentIdx"].to_numpy(dtype=int)
    n_critic = int(df["Critic_Score"].notna().sum())

    _log(f"Temiz veri: {len(df):,} satır | {df['Publisher'].nunique()} yayıncı | "
         f"{df['Platform'].nunique()} platform | {df['Franchise'].nunique()} seri")
    _log(f"Kritik puanı olan: {n_critic:,} satır "
         f"(%{100 * n_critic / len(df):.1f})")
    _log("Segment dağılımı: " + ", ".join(
        f"{lbl}={int((seg == i).sum())}" for i, lbl in enumerate(C.SEGMENT_LABELS)))

    idx = np.arange(len(df))
    idx_trval, idx_te = train_test_split(
        idx, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=seg)
    idx_tr, idx_val = train_test_split(
        idx_trval, test_size=C.VALID_SIZE / (1 - C.TEST_SIZE),
        random_state=C.RANDOM_STATE, stratify=seg[idx_trval])
    artifacts, m_random = run_protocol(df, idx_tr, idx_val, idx_te,
                                       "rastgele_katmanli")

    yr = df["Year_of_Release"].to_numpy()
    idx_tr2 = np.flatnonzero(yr <= C.CHRONO_TRAIN_MAX_YEAR)
    idx_val2 = np.flatnonzero((yr > C.CHRONO_TRAIN_MAX_YEAR) &
                              (yr <= C.CHRONO_VALID_MAX_YEAR))
    idx_te2 = np.flatnonzero(yr > C.CHRONO_VALID_MAX_YEAR)
    _, m_chrono = run_protocol(df, idx_tr2, idx_val2, idx_te2, "kronolojik")

    _log(f"{C.CV_FOLDS} katlı çapraz doğrulama çalışıyor...")
    cv = _cross_validate(df)
    _log(f"  CV R2_log = {cv['mean']['R2_log']:.4f} +/- "
         f"{cv['std']['R2_log']:.4f}")

    metrics: dict[str, Any] = {
        "dataset": {
            "rows_clean": int(len(df)),
            "rows_with_critic_score": n_critic,
            "n_publishers": int(df["Publisher"].nunique()),
            "n_platforms": int(df["Platform"].nunique()),
            "n_genres": int(df["Genre"].nunique()),
            "n_franchises": int(df["Franchise"].nunique()),
            "year_range": [int(df["Year_of_Release"].min()),
                           int(df["Year_of_Release"].max())],
            "segment_counts": {lbl: int((seg == i).sum())
                               for i, lbl in enumerate(C.SEGMENT_LABELS)},
        },
        "protocols": {"random": m_random, "chronological": m_chrono},
        "cross_validation": cv,
        "n_seed_models": C.N_SEED_MODELS,
        "train_seconds": round(time.time() - t0, 1),
    }
    metrics.update({k: m_random[k] for k in
                    ("regression", "classification", "calibration",
                     "consistency", "fusion", "baselines", "feature_importance")})
    metrics["dataset"]["split"] = m_random["split"]

    bundle = {
        **artifacts,
        "segment_labels": C.SEGMENT_LABELS,
        "segment_edges": C.SEGMENT_EDGES.tolist(),
        "metrics": metrics,
    }
    joblib.dump(bundle, C.BUNDLE_PATH, compress=3)
    C.METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Model kaydedildi -> {C.BUNDLE_PATH}")
    _log(f"Metrikler kaydedildi -> {C.METRICS_PATH}")
    _log(f"Toplam süre: {metrics['train_seconds']}s")
    return metrics


if __name__ == "__main__":
    main()
