"""Tutarlılık sözleşmesi ve veri katmanı testleri."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import consistency as CS
from src.data import franchise_key, segment_index


@pytest.mark.parametrize("name,expected", [
    ("Call of Duty: Black Ops II", "call of duty"),
    ("Call of Duty 4: Modern Warfare", "call of duty"),
    ("Final Fantasy VII", "final fantasy"),
    ("Final Fantasy XIII-2", "final fantasy"),
    ("Mario Kart Wii", "mario kart wii"),
    ("", "?"),
])
def test_franchise_key(name, expected):
    assert franchise_key(name) == expected


def test_franchise_history_is_strictly_backward_looking():
    """Bir oyunun KENDİ yılındaki satışlar geçmiş istatistiğine girmemeli."""
    from src.features import FranchiseHistory

    df = pd.DataFrame({
        "Franchise": ["seri a"] * 4 + ["seri b"],
        "Year_of_Release": [2000, 2000, 2005, 2010, 2001],
        C.TARGET: [1.0, 3.0, 10.0, 20.0, 5.0],
    })
    fh = FranchiseHistory().fit(df)

    assert fh.lookup("seri a", 2000)[0] == 0.0

    cnt, mean, mx, gap = fh.lookup("seri a", 2005)
    assert cnt == 2
    assert np.isclose(mean, np.mean(np.log1p([1.0, 3.0])))
    assert np.isclose(mx, np.log1p(3.0))
    assert gap == 5

    cnt, mean, mx, gap = fh.lookup("seri a", 2010)
    assert cnt == 3
    assert np.isclose(mx, np.log1p(10.0))
    assert gap == 5

    assert fh.lookup("hic yok", 2010)[0] == 0.0


@pytest.mark.parametrize("sales,expected", [
    (0.01, 0), (0.099, 0),
    (0.10, 1), (0.30, 1), (0.499, 1),
    (0.50, 2), (1.99, 2),
    (2.00, 3), (82.74, 3),
])
def test_segment_boundaries(sales, expected):
    assert segment_index(np.array([sales]))[0] == expected


def test_regression_probabilities_sum_to_one():
    mu = np.log1p(np.array([0.02, 0.3, 1.2, 5.0, 40.0]))
    p = CS.regression_to_probabilities(mu, 0.27)
    assert p.shape == (5, C.N_SEGMENTS)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_regression_probabilities_peak_at_own_segment():
    sales = np.array([0.03, 0.25, 1.0, 8.0])
    p = CS.regression_to_probabilities(np.log1p(sales), 0.02)
    np.testing.assert_array_equal(p.argmax(axis=1), segment_index(sales))


def test_fuse_extremes_recover_inputs():
    p_reg = np.array([[0.7, 0.2, 0.05, 0.05]])
    p_cls = np.array([[0.1, 0.1, 0.3, 0.5]])
    np.testing.assert_allclose(CS.fuse(p_reg, p_cls, 1.0), p_reg, atol=1e-6)
    np.testing.assert_allclose(CS.fuse(p_reg, p_cls, 0.0), p_cls, atol=1e-6)


def test_fuse_rows_are_distributions():
    rng = np.random.default_rng(0)
    a = rng.dirichlet(np.ones(C.N_SEGMENTS), size=50)
    b = rng.dirichlet(np.ones(C.N_SEGMENTS), size=50)
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = CS.fuse(a, b, w)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
        assert (p >= 0).all()


def test_decision_weights_preserve_distribution():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(C.N_SEGMENTS), size=30)
    q = CS.apply_decision_weights(p, np.array([1.5, 0.8, 1.0, 2.0]))
    np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-9)
    assert (q >= 0).all()


_MUS = np.log1p(np.concatenate([
    np.geomspace(0.01, 20.0, 80), np.array([82.74])]))


def test_point_estimate_inside_chosen_segment_exhaustive():
    """Tahmin hangi segmente zorlanırsa zorlansın, çıktı o segmentin içinde kalmalı."""
    for sigma in (0.02, 0.27, 1.0, 3.0):
        for k in range(C.N_SEGMENTS):
            seg = np.full(_MUS.shape, k, dtype=int)
            sales = CS.point_estimate(_MUS, sigma, seg)
            assert np.isfinite(sales).all(), f"sigma={sigma}, k={k}"
            CS.verify_consistency(sales, seg)


def test_point_estimate_untouched_when_already_inside():
    """MİNİMAL MÜDAHALE: tahmin zaten segmentin içindeyse aynen korunmalı."""
    sales = np.array([0.04, 0.30, 1.10, 6.00])
    mu = np.log1p(sales)
    out = CS.point_estimate(mu, 0.27, segment_index(sales))
    np.testing.assert_allclose(out, sales, rtol=1e-6)


def test_point_estimate_projects_when_outside():
    """Tahmin dışarıdaysa segmente projekte edilmeli, çökmemeli."""
    mu = np.log1p(np.array([40.0, 60.0]))
    k = np.array([0, 0])
    sales = CS.point_estimate(mu, 0.3, k)
    CS.verify_consistency(sales, k)
    assert (sales < C.SEGMENT_EDGES[1]).all()


def test_point_estimate_monotonic_within_segment():
    mu = np.log1p(np.linspace(0.11, 0.49, 40))
    sales = CS.point_estimate(mu, 0.27, np.full(40, 1))
    assert np.all(np.diff(sales) > 0)


def test_interval_brackets_point_estimate_and_stays_in_segment():
    for sigma in (0.02, 0.27, 1.0, 3.0):
        for k in range(C.N_SEGMENTS):
            seg = np.full(_MUS.shape, k, dtype=int)
            point = CS.point_estimate(_MUS, sigma, seg)
            lo, hi = CS.interval(_MUS, sigma, seg)
            assert np.isfinite(lo).all() and np.isfinite(hi).all()
            assert (lo <= hi).all(), f"sigma={sigma}, k={k}"
            assert (lo <= point * (1 + 1e-9)).all()
            assert (point <= hi * (1 + 1e-9)).all()
            CS.verify_consistency(lo, seg)
            CS.verify_consistency(hi, seg)


def test_verify_consistency_detects_violation():
    with pytest.raises(AssertionError, match="TUTARSIZLIK"):
        CS.verify_consistency(np.array([5.0]), np.array([0]))


def test_is_inside_matches_segment_index():
    sales = np.geomspace(0.01, 80.0, 200)
    mu = np.log1p(sales)
    k = segment_index(sales)
    assert CS.is_inside(mu, k).all()
    wrong = (k + 1) % C.N_SEGMENTS
    assert not CS.is_inside(mu, wrong).any()


def test_clean_drops_leaky_and_parses_types():
    from src.data import clean

    raw = pd.DataFrame({
        "Name": ["Wii Sports", "Wii Sports", "Küçük Oyun"],
        "Platform": ["Wii", "Wii", "PC"],
        "Year_of_Release": [2006, 2006, 2012],
        "Genre": ["Sports", "Sports", "Strategy"],
        "Publisher": ["Nintendo", "Nintendo", None],
        "NA_Sales": [41.49, 41.49, 0.01],
        "EU_Sales": [29.02, 29.02, 0.01],
        "JP_Sales": [3.77, 3.77, 0.0],
        "Other_Sales": [8.46, 8.46, 0.0],
        "Global_Sales": [82.74, 82.74, 0.02],
        "Critic_Score": [76, 76, None],
        "Critic_Count": [51, 51, None],
        "User_Score": ["8", "8", "tbd"],
        "User_Count": [322, 322, None],
        "Developer": ["Nintendo", "Nintendo", None],
        "Rating": ["E", "E", None],
    })
    out = clean(raw)

    assert len(out) == 2, "mükerrer satır düşmeliydi"
    for col in C.LEAKY_COLUMNS:
        assert col not in out.columns, f"{col} sızıntılı, düşmeliydi"
    assert out["Publisher"].isna().sum() == 0
    small = out[out["Name"] == "Küçük Oyun"].iloc[0]
    assert np.isnan(small["User_Score"])
    assert set(["Franchise", "SegmentIdx", "Segment"]).issubset(out.columns)
    assert out.loc[out["Name"] == "Wii Sports", "Segment"].iat[0] == "Blockbuster"


def test_assert_no_leakage_rejects_forbidden_columns():
    from src.data import assert_no_leakage

    good = pd.DataFrame({c: [0] for c in C.FEATURES})
    assert_no_leakage(good)

    for bad_col in ["NA_Sales", "Critic_Count", "User_Count", C.TARGET, "Name"]:
        bad = good.copy()
        bad[bad_col] = 1
        with pytest.raises(ValueError, match="SIZINTI"):
            assert_no_leakage(bad)


def test_post_hoc_columns_are_not_features():
    """Puan sayaçları özellik listesinde olmamalı."""
    assert not (set(C.FEATURES) & set(C.POST_HOC_COLUMNS))
    assert not (set(C.FEATURES) & set(C.LEAKY_COLUMNS))


def _base(p):
    return dict(Platform="PS4", Genre="Shooter", Publisher="Activision",
                Rating="M", Critic_Score=80, User_Score=7.5)


@pytest.mark.skipif(not C.BUNDLE_PATH.exists(), reason="model henüz eğitilmedi")
def test_trained_model_never_contradicts_itself():
    """Geniş bir girdi ızgarasında API çıktısı asla çelişmemeli."""
    from src.predict import get_predictor

    p = get_predictor()
    records = [
        {"Name": n, "Platform": pl, "Genre": g, "Publisher": pu,
         "Rating": rt, "Year_of_Release": y, "Critic_Score": cs,
         "User_Score": us}
        for n, pl, g, pu, rt, y, cs, us in itertools.product(
            ["Call of Duty: Yeni", "Tamamen Ozgun Oyun", ""],
            p.fb.known_platforms[:6], p.fb.known_genres[:4],
            p.fb.known_publishers[:3], ["E", "M"],
            [2000, 2010, 2015], [45, 90], [None, 7.0])
    ]
    results = p.predict(records, explain=False)
    assert len(results) == len(records)

    for rec, r in zip(records, results):
        lo, hi = r.segment_range
        assert lo <= r.predicted_sales, f"{rec} -> {r}"
        if hi is not None:
            assert r.predicted_sales < hi, f"{rec} -> {r}"
        assert r.segment == C.SEGMENT_LABELS[
            segment_index(np.array([r.predicted_sales]))[0]]
        assert abs(sum(r.segment_probabilities.values()) - 1.0) < 1e-3
        assert r.interval[0] <= r.predicted_sales <= r.interval[1]


@pytest.mark.skipif(not C.BUNDLE_PATH.exists(), reason="model henüz eğitilmedi")
def test_displayed_probabilities_agree_with_chosen_segment():
    """Arayüzdeki en yüksek çubuk, ilan edilen segment olmalı."""
    from src.predict import get_predictor

    p = get_predictor()
    records = [
        {"Name": n, "Platform": pl, "Genre": "Action", "Publisher": pu,
         "Rating": "T", "Year_of_Release": y, "Critic_Score": cs,
         "User_Score": 7.0}
        for n in ("Mario Kart 9", "Yeni Seri", "")
        for pl in p.fb.known_platforms[:8]
        for pu in p.fb.known_publishers[:3]
        for y in (2005, 2014)
        for cs in (50, 85)
    ]
    for r in p.predict(records, explain=False):
        top = max(r.segment_probabilities, key=r.segment_probabilities.get)
        assert top == r.segment, f"çubuk '{top}' derken rozet '{r.segment}'"


@pytest.mark.skipif(not C.BUNDLE_PATH.exists(), reason="model henüz eğitilmedi")
def test_higher_critic_score_predicts_more_sales():
    """Kritik puanı modelin en güçlü sinyali, yönü doğru öğrenilmiş olmalı."""
    from src.predict import get_predictor

    p = get_predictor()
    low = p.predict_one(Name="Bir Oyun", Year_of_Release=2014,
                        **{**_base(p), "Critic_Score": 45})
    high = p.predict_one(Name="Bir Oyun", Year_of_Release=2014,
                         **{**_base(p), "Critic_Score": 92})
    assert high.predicted_sales > low.predicted_sales
    assert high.segment_index >= low.segment_index


@pytest.mark.skipif(not C.BUNDLE_PATH.exists(), reason="model henüz eğitilmedi")
def test_known_franchise_beats_new_ip():
    """Bilinen serinin adı, aynı metadata'ya sahip yeni IP'den yüksek çıkmalı."""
    from src.predict import get_predictor

    p = get_predictor()
    known = p.predict_one(Name="Call of Duty: Yeni Bolum",
                          Year_of_Release=2014, **_base(p))
    fresh = p.predict_one(Name="Hicbir Seriye Ait Olmayan Oyun",
                          Year_of_Release=2014, **_base(p))
    assert known.predicted_sales > fresh.predicted_sales


@pytest.mark.skipif(not C.BUNDLE_PATH.exists(), reason="model henüz eğitilmedi")
def test_out_of_range_year_is_rejected():
    """Veride gözlenmeyen çıkış yılı için tahmin üretilmemeli."""
    from src.predict import get_predictor

    p = get_predictor()
    for bad in (p.year_min - 1, p.year_max + 1, 1900, 2100):
        with pytest.raises(ValueError, match="desteklenmiyor"):
            p.predict_one(Name="X", Year_of_Release=bad, **_base(p))

    for ok in (p.year_min, p.year_max):
        assert p.predict_one(Name="X", Year_of_Release=ok,
                             **_base(p)).predicted_sales > 0


@pytest.mark.skipif(not C.BUNDLE_PATH.exists(), reason="model henüz eğitilmedi")
def test_missing_scores_are_allowed_and_noted():
    """Kritik ve kullanıcı puanı isteğe bağlı, eksikse uyarı düşmeli."""
    from src.predict import get_predictor

    p = get_predictor()
    r = p.predict_one(Name="Bir Oyun", Platform="PS4", Genre="Shooter",
                      Publisher="Activision", Rating="M",
                      Year_of_Release=2014, Critic_Score=None, User_Score=None)
    assert r.segment in C.SEGMENT_LABELS
    assert any("Kritik puanı" in n for n in r.notes)


@pytest.mark.skipif(not C.BUNDLE_PATH.exists(), reason="model henüz eğitilmedi")
def test_unknown_categories_do_not_crash():
    from src.predict import get_predictor

    r = get_predictor().predict_one(
        Name="Bilinmeyen Oyun", Platform="QuantumBox9", Genre="Metaverse",
        Publisher="Yok A.S.", Rating="ZZZ", Year_of_Release=2014,
        Critic_Score=70, User_Score=6.0)
    assert r.segment in C.SEGMENT_LABELS
    assert r.notes


@pytest.mark.skipif(not C.BUNDLE_PATH.exists(), reason="model henüz eğitilmedi")
def test_model_features_match_config():
    from src.predict import get_predictor

    p = get_predictor()
    for reg in p.regressors:
        assert set(reg.feature_name_) == set(C.FEATURES)
