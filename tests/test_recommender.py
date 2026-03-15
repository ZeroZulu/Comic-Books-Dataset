"""
tests/test_recommender.py
=========================
Unit tests for src/recommender.py
"""

import pytest
import pandas as pd
import numpy as np
from src.recommender import ComicRecommender


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Small DataFrame with enough rows to build a similarity matrix."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "title":       [f"Comic Title {i}" for i in range(n)],
        "publisher":   np.random.choice(["Marvel", "DC", "Image"], n),
        "genre":       np.random.choice(["Action", "Romance", "Horror"], n),
        "theme":       np.random.choice(["Full Color", "Black & White"], n),
        "age_rating":  np.random.choice(["All Ages", "Teen", "Mature"], n),
        "country":     np.random.choice(["USA", "Japan"], n),
        "rating":      np.random.uniform(6.0, 9.9, n).round(1),
        "pages":       np.random.randint(100, 500, n),
        "volume_count": np.random.randint(1, 20, n),
    })


@pytest.fixture
def fitted_rec(sample_df):
    rec = ComicRecommender(top_n=5)
    rec.fit(sample_df)
    return rec, sample_df


# ── Tests: fit ────────────────────────────────────────────────────────────────

def test_fit_creates_sim_matrix(sample_df):
    rec = ComicRecommender()
    rec.fit(sample_df)
    assert rec.sim_matrix_ is not None


def test_sim_matrix_is_square(sample_df):
    rec = ComicRecommender()
    rec.fit(sample_df)
    n = len(sample_df.dropna(subset=["title"]))
    assert rec.sim_matrix_.shape == (n, n)


def test_sim_matrix_diagonal_is_one(sample_df):
    rec = ComicRecommender()
    rec.fit(sample_df)
    diag = np.diag(rec.sim_matrix_)
    assert np.allclose(diag, 1.0, atol=1e-5)


def test_fit_returns_self(sample_df):
    rec = ComicRecommender()
    result = rec.fit(sample_df)
    assert result is rec


# ── Tests: recommend ─────────────────────────────────────────────────────────

def test_recommend_returns_dataframe(fitted_rec):
    rec, df = fitted_rec
    result = rec.recommend("Comic Title 0")
    assert isinstance(result, pd.DataFrame)


def test_recommend_returns_correct_count(fitted_rec):
    rec, _ = fitted_rec
    result = rec.recommend("Comic Title 0", n=3)
    assert len(result) == 3


def test_recommend_excludes_query_title(fitted_rec):
    rec, _ = fitted_rec
    result = rec.recommend("Comic Title 0", n=5)
    assert "Comic Title 0" not in result["title"].values


def test_recommend_has_similarity_col(fitted_rec):
    rec, _ = fitted_rec
    result = rec.recommend("Comic Title 0")
    assert "similarity" in result.columns


def test_similarity_scores_in_range(fitted_rec):
    rec, _ = fitted_rec
    result = rec.recommend("Comic Title 0")
    assert (result["similarity"] >= 0).all()
    assert (result["similarity"] <= 1).all()


def test_similarity_scores_descending(fitted_rec):
    rec, _ = fitted_rec
    result = rec.recommend("Comic Title 0")
    scores = result["similarity"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_recommend_raises_on_missing_title(fitted_rec):
    rec, _ = fitted_rec
    with pytest.raises(ValueError, match="No comics found"):
        rec.recommend("XXXXXXXXXNOTEXIST")


def test_recommend_raises_before_fit():
    rec = ComicRecommender()
    with pytest.raises(RuntimeError, match="Call .fit"):
        rec.recommend("Spider-Man")


# ── Tests: avg_similarity_by_group ───────────────────────────────────────────

def test_avg_similarity_by_group_returns_series(fitted_rec):
    rec, _ = fitted_rec
    result = rec.avg_similarity_by_group("publisher")
    assert isinstance(result, pd.Series)


def test_avg_similarity_values_non_negative(fitted_rec):
    rec, _ = fitted_rec
    result = rec.avg_similarity_by_group("publisher")
    assert (result >= 0).all()


def test_avg_similarity_sorted_ascending(fitted_rec):
    rec, _ = fitted_rec
    result = rec.avg_similarity_by_group("publisher")
    assert list(result) == sorted(result)
