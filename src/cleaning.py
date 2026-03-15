"""
cleaning.py — Data Cleaning & Feature Engineering Pipeline
============================================================
Mirrors the logic in notebook Cell 09 exactly.

Handles the non-standard column names in the Comic Books Dataset:
    Studio/Publisher   → publisher
    Release Year       → year
    Page Count         → pages
    Rating (out of 10) → rating
    Theme (Color Style)→ theme
    Country of Origin  → country

Usage
-----
    from src.cleaning import clean_pipeline
    df_clean = clean_pipeline(df)
"""

import pandas as pd
import numpy as np
from typing import Optional


# ── Column rename map ──────────────────────────────────────────────────────────
RENAME_MAP: dict[str, str] = {
    # Publisher
    "studio_publisher":    "publisher",
    "publisher":           "publisher",
    # Year
    "release_year":        "year",
    "publication_year":    "year",
    "year":                "year",
    # Pages
    "page_count":          "pages",
    "num_pages":           "pages",
    "pages":               "pages",
    # Rating
    "rating_(out_of_10)":  "rating",
    "user_rating":         "rating",
    "rating":              "rating",
    # Price (not in this dataset but future-proof)
    "price_usd":           "price",
    "cover_price":         "price",
    "price":               "price",
    # Theme
    "theme_(color_style)": "theme",
    # Country
    "country_of_origin":   "country",
    "country":             "country",
    # Volume
    "volume_count":        "volume_count",
}

# Numeric columns to coerce
NUMERIC_COLS = ["price", "rating", "pages", "year", "issue_number", "volume_count"]

# Era bins aligned to actual data range (year 2000–2026 confirmed from data)
ERA_BINS   = [0, 1950, 1969, 1985, 1999, 2010, 2030]
ERA_LABELS = ["Golden Age", "Silver Age", "Bronze Age",
              "Modern Age", "21st Century", "Contemporary"]

# Rating bins aligned to actual data range (6.0 – 9.9)
RATING_BINS   = [0, 6.5, 7.5, 8.5, 10]
RATING_LABELS = ["Below Average", "Average", "Good", "Excellent"]


# ── Individual steps ───────────────────────────────────────────────────────────

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase, strip whitespace, replace spaces/slashes with underscores,
    then rename to standard names using RENAME_MAP.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s/]+", "_", regex=True)
    )
    df = df.rename(
        columns={k: v for k, v in RENAME_MAP.items() if k in df.columns}
    )
    return df


def coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Force known numeric columns to float, coercing unparseable values to NaN."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features:
        decade          — year floored to decade (Int64, nullable)
        era             — named comic era (Categorical)
        rating_label    — rating bucket aligned to actual 6–10 range
        title_word_count— number of words in title
    """
    df = df.copy()

    if "year" in df.columns:
        df["decade"] = (df["year"] // 10 * 10).astype("Int64")
        df["era"] = pd.cut(df["year"], bins=ERA_BINS, labels=ERA_LABELS)

    if "rating" in df.columns:
        # Bins match actual data range (6.0–9.9) — not [0,2,4,6,8,10]
        df["rating_label"] = pd.cut(
            df["rating"],
            bins=RATING_BINS,
            labels=RATING_LABELS,
        )

    if "price" in df.columns:
        valid_prices = df["price"].dropna()
        if len(valid_prices) > 0:
            df["price_bucket"] = pd.qcut(
                df["price"],
                q=4,
                labels=["Budget", "Mid-Range", "Premium", "Collector"],
                duplicates="drop",
            )

    if "title" in df.columns:
        df["title_word_count"] = df["title"].str.split().str.len()

    return df


def clean_pipeline(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full cleaning pipeline:
        1. Standardise & rename columns
        2. Coerce numeric types
        3. Drop exact duplicates (optional)
        4. Engineer derived features

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as loaded from CSV.
    drop_duplicates : bool
        Whether to drop exact duplicate rows. Default True.
    verbose : bool
        Print progress summary. Default True.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for analysis.

    Examples
    --------
    >>> import pandas as pd
    >>> from src.cleaning import clean_pipeline
    >>> df = pd.read_csv("data/comic_books_10000_dataset.csv")
    >>> df_clean = clean_pipeline(df)
    """
    df_clean = standardise_columns(df)
    df_clean = coerce_numerics(df_clean)

    if drop_duplicates:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = before - len(df_clean)
        if verbose:
            print(f"Removed {removed} duplicate rows.")

    df_clean = engineer_features(df_clean)

    if verbose:
        print(f"\n✅ Cleaned dataset: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
        print("📋 Final columns:", df_clean.columns.tolist())

    return df_clean


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising missing values and dtypes for each column.

    Returns
    -------
    pd.DataFrame with columns: [dtype, missing_count, missing_pct, nunique]
    """
    report = pd.DataFrame({
        "dtype":         df.dtypes,
        "missing_count": df.isnull().sum(),
        "missing_pct":   (df.isnull().sum() / len(df) * 100).round(2),
        "nunique":       df.nunique(),
    })
    return report.sort_values("missing_pct", ascending=False)
