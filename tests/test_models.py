"""
tests/test_models.py
====================
Unit tests for src/models.py
"""

import pytest
import numpy as np
import pandas as pd
from src.models import (
    prepare_features,
    train_and_evaluate,
    best_model,
    feature_importance_df,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Small, predictable DataFrame for ML tests."""
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        "pages":       np.random.randint(100, 600, n).astype(float),
        "year":        np.random.randint(2000, 2025, n).astype(float),
        "volume_count": np.random.randint(1, 30, n).astype(float),
        "publisher":   np.random.choice(["Marvel", "DC", "Image"], n),
        "genre":       np.random.choice(["Action", "Romance", "Horror"], n),
        "format":      np.random.choice(["Standard", "TPB"], n),
        "country":     np.random.choice(["USA", "Japan"], n),
        "language":    np.random.choice(["English", "Japanese"], n),
        "age_rating":  np.random.choice(["All Ages", "Teen", "Mature"], n),
        "status":      np.random.choice(["Ongoing", "Completed"], n),
        "theme":       np.random.choice(["Full Color", "B&W"], n),
        "rating":      np.random.uniform(6.0, 9.9, n).round(1),
    })


# ── Tests: prepare_features ───────────────────────────────────────────────────

def test_prepare_features_returns_five_items(sample_df):
    result = prepare_features(sample_df)
    assert len(result) == 5


def test_prepare_features_train_test_split(sample_df):
    X_train, X_test, y_train, y_test, _ = prepare_features(sample_df, test_size=0.2)
    total = len(X_train) + len(X_test)
    assert total == len(sample_df)
    assert abs(len(X_test) / total - 0.2) < 0.05


def test_prepare_features_no_nan_in_encoded(sample_df):
    X_train, X_test, y_train, y_test, _ = prepare_features(sample_df)
    assert not pd.isna(X_train).any().any()
    assert not pd.isna(X_test).any().any()


def test_prepare_features_correct_feature_count(sample_df):
    _, _, _, _, features = prepare_features(sample_df)
    # 3 numeric + up to 8 categorical (all present in sample)
    assert len(features) >= 3


# ── Tests: train_and_evaluate ─────────────────────────────────────────────────

def test_train_evaluate_returns_dict(sample_df):
    results = train_and_evaluate(sample_df, verbose=False)
    assert isinstance(results, dict)


def test_train_evaluate_has_four_models(sample_df):
    results = train_and_evaluate(sample_df, verbose=False)
    assert len(results) == 4


def test_train_evaluate_result_keys(sample_df):
    results = train_and_evaluate(sample_df, verbose=False)
    for name, r in results.items():
        for key in ["RMSE", "MAE", "R2", "CV_R2", "model", "preds", "y_test"]:
            assert key in r, f"Missing key '{key}' in results['{name}']"


def test_rmse_is_positive(sample_df):
    results = train_and_evaluate(sample_df, verbose=False)
    for name, r in results.items():
        assert r["RMSE"] > 0, f"RMSE should be positive for {name}"


def test_preds_same_length_as_y_test(sample_df):
    results = train_and_evaluate(sample_df, verbose=False)
    for name, r in results.items():
        assert len(r["preds"]) == len(r["y_test"]), \
            f"Pred/actual length mismatch for {name}"


def test_raises_if_target_missing():
    df = pd.DataFrame({"pages": [100, 200], "year": [2010, 2020]})
    with pytest.raises(ValueError, match="Target column"):
        train_and_evaluate(df, verbose=False)


# ── Tests: best_model ─────────────────────────────────────────────────────────

def test_best_model_returns_tuple(sample_df):
    results = train_and_evaluate(sample_df, verbose=False)
    result  = best_model(results)
    assert isinstance(result, tuple) and len(result) == 2


def test_best_model_is_highest_r2(sample_df):
    results   = train_and_evaluate(sample_df, verbose=False)
    name, info = best_model(results)
    max_r2    = max(r["R2"] for r in results.values())
    assert abs(info["R2"] - max_r2) < 1e-10


# ── Tests: feature_importance_df ─────────────────────────────────────────────

def test_feature_importance_for_rf(sample_df):
    results    = train_and_evaluate(sample_df, verbose=False)
    name, info = best_model(results)
    if hasattr(info["model"], "feature_importances_"):
        imp = feature_importance_df(name, info)
        assert isinstance(imp, pd.Series)
        assert (imp >= 0).all()


def test_feature_importance_sorted_ascending(sample_df):
    results    = train_and_evaluate(sample_df, verbose=False)
    name, info = best_model(results)
    imp        = feature_importance_df(name, info)
    assert list(imp) == sorted(imp)
