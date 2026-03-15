"""
tests/test_cleaning.py
======================
Unit tests for src/cleaning.py
"""

import pytest
import pandas as pd
import numpy as np
from src.cleaning import (
    standardise_columns,
    coerce_numerics,
    engineer_features,
    clean_pipeline,
    data_quality_report,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    """Minimal DataFrame mimicking the actual dataset's raw column names."""
    return pd.DataFrame({
        "Title":              ["Spider-Man #1", "Batman Vol. 2", "Dragon Ball"],
        "Studio/Publisher":   ["Marvel Comics", "DC Comics", "Shogakukan (Viz Media)"],
        "Release Year":       [2010, 2015, 2000],
        "Page Count":         [200, 350, 180],
        "Rating (out of 10)": [8.5, 7.9, 9.1],
        "Genre":              ["Superhero", "Superhero", "Action"],
        "Volume Count":       [5, 3, 42],
        "Status":             ["Ongoing", "Completed", "Completed"],
        "Language":           ["English", "English", "Japanese"],
        "Country of Origin":  ["USA", "USA", "Japan"],
        "Age Rating":         ["Teen", "Mature", "All Ages"],
        "Format":             ["Standard", "TPB", "Standard"],
        "Theme (Color Style)":["Full Color", "Full Color", "Black & White"],
        "Awards":             ["Hugo Award", None, "Eisner Award"],
        "Writer":             ["Stan Lee", "Tom King", "Akira Toriyama"],
        "Artist":             ["Steve Ditko", "David Finch", "Akira Toriyama"],
        "comic_id":           ["C001", "C002", "C003"],
    })


# ── Tests: standardise_columns ────────────────────────────────────────────────

def test_standardise_renames_publisher(raw_df):
    df = standardise_columns(raw_df)
    assert "publisher" in df.columns, "studio_publisher should be renamed to publisher"
    assert "Studio/Publisher" not in df.columns


def test_standardise_renames_year(raw_df):
    df = standardise_columns(raw_df)
    assert "year" in df.columns


def test_standardise_renames_pages(raw_df):
    df = standardise_columns(raw_df)
    assert "pages" in df.columns


def test_standardise_renames_rating(raw_df):
    df = standardise_columns(raw_df)
    assert "rating" in df.columns


def test_standardise_renames_theme(raw_df):
    df = standardise_columns(raw_df)
    assert "theme" in df.columns


def test_standardise_renames_country(raw_df):
    df = standardise_columns(raw_df)
    assert "country" in df.columns


def test_standardise_lowercases_title(raw_df):
    df = standardise_columns(raw_df)
    assert "title" in df.columns


# ── Tests: coerce_numerics ────────────────────────────────────────────────────

def test_coerce_rating_is_float(raw_df):
    df = standardise_columns(raw_df)
    df = coerce_numerics(df)
    assert df["rating"].dtype in [np.float64, np.float32]


def test_coerce_year_is_numeric(raw_df):
    df = standardise_columns(raw_df)
    df = coerce_numerics(df)
    assert pd.api.types.is_numeric_dtype(df["year"])


def test_coerce_bad_values_become_nan():
    df = pd.DataFrame({"rating": ["8.5", "not_a_number", "9.1"]})
    df = coerce_numerics(df)
    assert df["rating"].isna().sum() == 1


# ── Tests: engineer_features ──────────────────────────────────────────────────

def test_engineer_creates_decade(raw_df):
    df = standardise_columns(coerce_numerics(raw_df) if True else raw_df)
    df = coerce_numerics(df)
    df = engineer_features(df)
    assert "decade" in df.columns


def test_decade_values_correct(raw_df):
    df = clean_pipeline(raw_df, verbose=False)
    assert df.loc[df["year"] == 2010, "decade"].iloc[0] == 2010
    assert df.loc[df["year"] == 2015, "decade"].iloc[0] == 2010
    assert df.loc[df["year"] == 2000, "decade"].iloc[0] == 2000


def test_engineer_creates_era(raw_df):
    df = clean_pipeline(raw_df, verbose=False)
    assert "era" in df.columns


def test_era_values_in_known_labels(raw_df):
    df = clean_pipeline(raw_df, verbose=False)
    valid_eras = {"Golden Age", "Silver Age", "Bronze Age",
                  "Modern Age", "21st Century", "Contemporary", np.nan}
    assert set(df["era"].astype(str).unique()) <= {str(e) for e in valid_eras}


def test_rating_label_uses_correct_bins(raw_df):
    """Rating 8.5 should map to 'Good', not 'Excellent' or empty."""
    df = clean_pipeline(raw_df, verbose=False)
    row = df[df["rating"] == 8.5]
    assert not row.empty
    assert str(row["rating_label"].iloc[0]) == "Good"


def test_title_word_count(raw_df):
    df = clean_pipeline(raw_df, verbose=False)
    assert "title_word_count" in df.columns
    assert df.loc[df["title"] == "Spider-Man #1", "title_word_count"].iloc[0] == 2


# ── Tests: clean_pipeline ─────────────────────────────────────────────────────

def test_pipeline_no_duplicates(raw_df):
    df_dup = pd.concat([raw_df, raw_df]).reset_index(drop=True)
    df_out = clean_pipeline(df_dup, verbose=False)
    assert df_out.duplicated().sum() == 0


def test_pipeline_preserves_row_count_without_dups(raw_df):
    df_out = clean_pipeline(raw_df, verbose=False)
    assert len(df_out) == len(raw_df)


def test_pipeline_returns_dataframe(raw_df):
    assert isinstance(clean_pipeline(raw_df, verbose=False), pd.DataFrame)


# ── Tests: data_quality_report ────────────────────────────────────────────────

def test_quality_report_columns(raw_df):
    report = data_quality_report(raw_df)
    for col in ["dtype", "missing_count", "missing_pct", "nunique"]:
        assert col in report.columns


def test_quality_report_missing_count(raw_df):
    report = data_quality_report(raw_df)
    # Awards has 1 None → 1 missing
    assert report.loc["Awards", "missing_count"] == 1
