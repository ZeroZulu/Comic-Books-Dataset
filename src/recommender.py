"""
recommender.py — Content-Based Comic Recommender System
=========================================================
Mirrors notebook Cell 29 exactly.

Feature vector:
  1. TF-IDF on title text           (max_features=300, stop_words='english')
  2. Normalised numerics            (rating, pages, volume_count)
  3. One-hot categoricals           (genre, publisher, theme, age_rating, country)

Similarity metric: cosine similarity

Memory note: 10,000 × 10,000 float32 similarity matrix ≈ 763 MB.
A warning is printed if the estimated size exceeds 500 MB.

Usage
-----
    from src.recommender import ComicRecommender

    rec = ComicRecommender()
    rec.fit(df_clean)

    # Returns a DataFrame with title, publisher, genre, theme, age_rating,
    # rating, pages, and similarity score
    recs = rec.recommend("Amazing Spider-Man", n=5)
    print(recs)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class ComicRecommender:
    """
    Content-based comic book recommender using cosine similarity.

    Parameters
    ----------
    top_n : int
        Default number of recommendations to return. Default 5.
    tfidf_features : int
        Max TF-IDF vocabulary size for title vectorisation. Default 300.
    numeric_cols : list[str], optional
        Numeric feature columns to include. Defaults to
        ['rating', 'pages', 'volume_count'].
    cat_cols : list[str], optional
        Categorical columns for one-hot encoding. Defaults to
        ['genre', 'publisher', 'theme', 'age_rating', 'country'].

    Attributes
    ----------
    rec_df_ : pd.DataFrame
        Internal copy of the fitted DataFrame (index-reset).
    sim_matrix_ : np.ndarray
        (n_samples, n_samples) cosine similarity matrix.
    """

    DEFAULT_NUMERIC = ["rating", "pages", "volume_count"]
    DEFAULT_CATS    = ["genre", "publisher", "theme", "age_rating", "country"]
    DISPLAY_COLS    = ["title", "publisher", "genre", "theme",
                       "age_rating", "rating", "pages", "similarity"]

    def __init__(
        self,
        top_n: int = 5,
        tfidf_features: int = 300,
        numeric_cols: list[str] | None = None,
        cat_cols: list[str] | None = None,
    ):
        self.top_n          = top_n
        self.tfidf_features = tfidf_features
        self.numeric_cols   = numeric_cols or self.DEFAULT_NUMERIC
        self.cat_cols       = cat_cols or self.DEFAULT_CATS
        self.rec_df_        = None
        self.sim_matrix_    = None

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "ComicRecommender":
        """
        Build the feature matrix and cosine similarity matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataframe (must contain 'title' column).

        Returns
        -------
        self
        """
        self.rec_df_ = df.dropna(subset=["title"]).reset_index(drop=True)
        feature_blocks = []

        # 1. TF-IDF on titles
        tfidf_rec    = TfidfVectorizer(stop_words="english",
                                       max_features=self.tfidf_features)
        title_matrix = tfidf_rec.fit_transform(
            self.rec_df_["title"].str.lower()
        ).toarray()
        feature_blocks.append(title_matrix)

        # 2. Scaled numeric features
        num_feat = [c for c in self.numeric_cols if c in self.rec_df_.columns]
        if num_feat:
            num_data   = self.rec_df_[num_feat].fillna(
                self.rec_df_[num_feat].median()
            )
            num_scaled = StandardScaler().fit_transform(num_data)
            feature_blocks.append(num_scaled)

        # 3. One-hot categorical features
        for col in self.cat_cols:
            if col in self.rec_df_.columns:
                dummies = pd.get_dummies(
                    self.rec_df_[col], prefix=col, dummy_na=True
                ).astype(float)
                feature_blocks.append(dummies.values)

        feature_matrix = np.hstack(feature_blocks)

        # Memory guard
        est_mb = (feature_matrix.shape[0] ** 2 * 4) / (1024 ** 2)
        if est_mb > 500:
            print(f"⚠️  Similarity matrix will be ~{est_mb:.0f} MB. Computing...")

        self.sim_matrix_ = cosine_similarity(feature_matrix)
        print(f"✅ Similarity matrix: {self.sim_matrix_.shape} "
              f"({self.sim_matrix_.nbytes / 1024**2:.1f} MB)")
        return self

    # ── Recommend ─────────────────────────────────────────────────────────────
    def recommend(self, title_query: str, n: int | None = None) -> pd.DataFrame:
        """
        Return the top-n most similar comics for a given title query.

        Parameters
        ----------
        title_query : str
            Partial or full comic title (case-insensitive substring match).
        n : int, optional
            Number of recommendations. Defaults to self.top_n.

        Returns
        -------
        pd.DataFrame
            Columns: title, publisher, genre, theme, age_rating, rating,
                     pages, similarity (score 0–1).

        Raises
        ------
        RuntimeError
            If .fit() has not been called yet.
        """
        if self.rec_df_ is None or self.sim_matrix_ is None:
            raise RuntimeError("Call .fit(df) before .recommend().")

        n = n or self.top_n

        matches = self.rec_df_[
            self.rec_df_["title"].str.contains(title_query, case=False, na=False)
        ]
        if matches.empty:
            raise ValueError(f"No comics found matching '{title_query}'.")

        idx        = matches.index[0]
        sim_scores = sorted(
            enumerate(self.sim_matrix_[idx]),
            key=lambda x: x[1],
            reverse=True,
        )[1 : n + 1]

        recs = self.rec_df_.iloc[[i for i, _ in sim_scores]].copy()
        recs["similarity"] = [round(s, 4) for _, s in sim_scores]

        display_cols = [c for c in self.DISPLAY_COLS if c in recs.columns]
        return recs[display_cols].reset_index(drop=True)

    def avg_similarity_by_group(
        self, group_col: str = "publisher", top_n: int = 6
    ) -> pd.Series:
        """
        Compute average cosine similarity per group (publisher / genre).
        Lower score = more diverse title catalog.

        Returns
        -------
        pd.Series sorted ascending (most diverse first).
        """
        if self.rec_df_ is None or self.sim_matrix_ is None:
            raise RuntimeError("Call .fit(df) first.")

        mat = self.sim_matrix_.copy()
        np.fill_diagonal(mat, 0)
        avg_sim = mat.mean(axis=1)

        tmp = self.rec_df_.copy()
        tmp["avg_similarity"] = avg_sim

        top_groups = tmp[group_col].value_counts().head(top_n).index
        return (
            tmp[tmp[group_col].isin(top_groups)]
            .groupby(group_col)["avg_similarity"]
            .mean()
            .sort_values()
            .round(4)
        )
