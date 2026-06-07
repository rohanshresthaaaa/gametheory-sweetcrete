"""Step 3 preprocessing: feature loading, train/test split, scaling pipeline,
and cross-validation strategy.

This is the single source of truth imported by the modeling step (Step 4), so the
exact same split, scaling, and CV are used everywhere. Scaling is wrapped INSIDE a
pipeline so the scaler only ever sees training data within each fold -> no leakage.
"""
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- resolve repo root from this file's location ---
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
from data_cleaning import clean_sweetcrete, SELECTED_FEATURES, TARGET  # noqa: E402

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_xy(raw_path: str | None = None):
    """Return (X, y, stratify_key) from the cleaned dataset."""
    if raw_path is None:
        raw_path = str(ROOT / "data" / "raw" / "Sweetcrete_Master_Final.csv")
    df = clean_sweetcrete(raw_path)
    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET].copy()
    stratify_key = df["PCC_%"].copy()   # keep every PCC level in both splits
    return X, y, stratify_key


def get_split(test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Stratified (by PCC level) train/test split. Returns X_train, X_test, y_train, y_test."""
    X, y, stratify_key = load_xy()
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_key
    )


def build_pipeline(model):
    """Wrap any scikit-learn regressor with standard scaling.
    The scaler is fit only on training data within each fold, preventing leakage."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def cv_strategy(n_splits: int = 5, n_repeats: int = 3, random_state: int = RANDOM_STATE):
    """Repeated k-fold CV. With ~178 rows, 5 folds x 3 repeats gives a stable R^2 estimate."""
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


if __name__ == "__main__":
    Xtr, Xte, ytr, yte = get_split()
    print(f"train: {Xtr.shape[0]} rows | test: {Xte.shape[0]} rows | features: {Xtr.shape[1]}")
