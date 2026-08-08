"""Proje geneli sabitler ve yol tanımları."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_CSV = DATA_DIR / "games.csv"
BUNDLE_PATH = MODELS_DIR / "sales_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET = "rush4ratio/video-game-sales-with-ratings"
KAGGLE_FILE = "Video_Games_Sales_as_at_22_Dec_2016.csv"

TARGET = "Global_Sales"
TARGET_UNIT = "milyon adet"

LEAKY_COLUMNS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]

POST_HOC_COLUMNS = ["Critic_Count", "User_Count"]

REQUIRE_CRITIC_SCORE = False

CATEGORICAL_FEATURES = [
    "Platform",
    "Genre",
    "Publisher",
    "Rating",
]

NUMERIC_FEATURES = [
    "Year_of_Release",
    "Critic_Score",
    "User_Score",
]

FRANCHISE_FEATURES = [
    "FranchisePriorCount",
    "FranchisePriorMean",
    "FranchisePriorMax",
    "FranchiseYearsSince",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + FRANCHISE_FEATURES

USER_INPUT_FIELDS = [
    "Name", "Platform", "Genre", "Publisher", "Year_of_Release",
    "Critic_Score", "User_Score", "Rating",
]

MIN_YEAR, MAX_YEAR = 1980, 2016

SEGMENT_LABELS = ["Düşük", "Orta", "Yüksek", "Blockbuster"]

SEGMENT_EDGES = np.array([0.0, 0.10, 0.50, 2.00, np.inf])
N_SEGMENTS = len(SEGMENT_LABELS)

SEGMENT_DESCRIPTIONS = {
    "Düşük": "Niş veya sınırlı dağıtım, 100 binin altında satış.",
    "Orta": "Ortalama ticari performans, 100 ile 500 bin adet.",
    "Yüksek": "Güçlü ticari başarı, 500 bin ile 2 milyon adet.",
    "Blockbuster": "Pazar lideri, 2 milyon adet ve üzeri.",
}

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALID_SIZE = 0.20
CV_FOLDS = 5
N_SEED_MODELS = 3

FUSION_WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)

FUSION_ALPHA = 0.5

DECISION_WEIGHT_GRID = np.geomspace(0.5, 3.0, 21)
DECISION_WEIGHT_ROUNDS = 4

CHRONO_TRAIN_MAX_YEAR = 2011
CHRONO_VALID_MAX_YEAR = 2013

LGBM_REGRESSOR_PARAMS = dict(
    objective="regression",
    n_estimators=2500,
    learning_rate=0.03,
    num_leaves=48,
    min_child_samples=15,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=1.5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

LGBM_CLASSIFIER_PARAMS = dict(
    objective="multiclass",
    num_class=N_SEGMENTS,
    n_estimators=2500,
    learning_rate=0.03,
    num_leaves=48,
    min_child_samples=15,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=1.5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

EARLY_STOPPING_ROUNDS = 100
