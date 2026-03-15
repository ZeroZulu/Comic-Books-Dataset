"""
visualizations.py — Reusable Dark-Themed Plot Functions
=========================================================
All functions reproduce the charts from the notebook using the same
GitHub-dark colour palette. Each function returns the figure object
so callers can save or further customise it.

Palette
-------
    HERO_BLUE = '#58A6FF'
    HERO_RED  = '#F78166'
    HERO_GRN  = '#3FB950'
    PALETTE   = ['#58A6FF','#F78166','#3FB950','#D2A8FF','#FFA657','#79C0FF']
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from typing import Optional


# ── Global Style ──────────────────────────────────────────────────────────────
PALETTE   = ["#58A6FF", "#F78166", "#3FB950", "#D2A8FF", "#FFA657", "#79C0FF"]
HERO_BLUE = "#58A6FF"
HERO_RED  = "#F78166"
HERO_GRN  = "#3FB950"
BG_DARK   = "#0D1117"
BG_PANEL  = "#161B22"
FG_TEXT   = "#C9D1D9"
EDGE_COL  = "#30363D"
DIM_BAR   = "#2D333B"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_PANEL,
    font_color=FG_TEXT,
)

def apply_dark_style() -> None:
    """Apply global Matplotlib dark style matching the notebook."""
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor":   BG_PANEL,
        "axes.edgecolor":   EDGE_COL,
        "axes.labelcolor":  FG_TEXT,
        "xtick.color":      FG_TEXT,
        "ytick.color":      FG_TEXT,
        "text.color":       FG_TEXT,
        "grid.color":       "#21262D",
        "font.family":      "DejaVu Sans",
        "figure.dpi":       130,
    })


# ── Section 4 — Basic Analysis ────────────────────────────────────────────────

def plot_publisher_distribution(
    df: pd.DataFrame,
    top_n: int = 15,
    col: str = "publisher",
) -> plt.Figure:
    """
    Horizontal bar chart of top-N publishers.
    Bars above the mean are highlighted in HERO_BLUE; others in DIM_BAR.
    A dashed mean reference line is added.
    """
    counts     = df[col].value_counts().head(top_n)
    mean_count = counts.mean()
    colors     = [HERO_BLUE if v >= mean_count else DIM_BAR for v in counts.values]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(counts.index, counts.values, color=colors,
                  edgecolor=EDGE_COL, linewidth=0.5)
    ax.axhline(mean_count, color=HERO_RED, linestyle="--",
               linewidth=1.2, label=f"Mean: {mean_count:.0f}")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 8,
                f"{int(h):,}", ha="center", va="bottom", fontsize=8, color=FG_TEXT)

    ax.set_title(f"📚 Top {top_n} Publishers by Comic Count",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Publisher")
    ax.set_ylabel("Number of Comics")
    ax.tick_params(axis="x", rotation=40)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_genre_donut(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Interactive Plotly donut chart of genre distribution."""
    counts = df["genre"].value_counts().head(top_n)
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title="🎭 Genre Distribution",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(**PLOTLY_LAYOUT, title_font_size=18)
    return fig


def plot_numerical_distributions(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
) -> plt.Figure:
    """
    Histograms with mean/median lines and skew/kurtosis annotation boxes.
    Uses named functions for skew/kurt to avoid pandas lambda-name collision.
    """
    if cols is None:
        cols = [c for c in ["rating", "pages", "year", "volume_count"] if c in df.columns]

    n = len(cols)
    if n == 0:
        raise ValueError("No numeric columns found.")

    colors = [HERO_BLUE, HERO_RED, HERO_GRN, "#D2A8FF"]
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle("📊 Key Numerical Distributions", fontsize=15, fontweight="bold")
    if n == 1:
        axes = [axes]

    for ax, col, color in zip(axes, cols, colors):
        data     = df[col].dropna()
        skew_val = data.skew()
        kurt_val = data.kurt()

        ax.hist(data, bins=40, color=color, alpha=0.82,
                edgecolor=BG_DARK, linewidth=0.3)
        ax.axvline(data.mean(),   color="white",  linestyle="--",
                   linewidth=1.3, label=f"Mean: {data.mean():.1f}")
        ax.axvline(data.median(), color="yellow", linestyle=":",
                   linewidth=1.3, label=f"Median: {data.median():.1f}")

        ax.set_title(f'{col.replace("_", " ").title()} Distribution',
                     fontsize=12, fontweight="bold")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax.text(0.97, 0.95, f"Skew: {skew_val:+.2f}\nKurt: {kurt_val:+.2f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262D",
                          edgecolor=EDGE_COL, alpha=0.9))

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Lower-triangle correlation heatmap using coolwarm palette."""
    num_df = df.select_dtypes(include="number")
    corr   = num_df.corr()
    mask   = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5, linecolor=BG_DARK,
                cbar_kws={"shrink": 0.8})
    ax.set_title("🔥 Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Section 5 — Intermediate Analysis ────────────────────────────────────────

def plot_publisher_rating(df: pd.DataFrame, min_count: int = 20) -> plt.Figure:
    """
    Horizontal bar chart of publisher average rating (min_count filter).
    X-axis zoomed in to the actual range so differences are visible.
    """
    pub_rating = (
        df.groupby("publisher")
        .agg(avg_rating=("rating", "mean"), count=("rating", "count"))
        .query(f"count >= {min_count}")
        .sort_values("avg_rating", ascending=True)
        .tail(15)
    )
    if pub_rating.empty:
        pub_rating = (
            df.groupby("publisher")
            .agg(avg_rating=("rating", "mean"), count=("rating", "count"))
            .query("count >= 5")
            .sort_values("avg_rating", ascending=True)
            .tail(15)
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        pub_rating.index, pub_rating["avg_rating"],
        color=[HERO_BLUE if v >= pub_rating["avg_rating"].median() else DIM_BAR
               for v in pub_rating["avg_rating"]],
    )
    ax.axvline(pub_rating["avg_rating"].mean(), color=HERO_RED, linestyle="--",
               linewidth=1.5, label=f'Mean: {pub_rating["avg_rating"].mean():.2f}')
    ax.set_xlim(
        max(0, pub_rating["avg_rating"].min() - 0.3),
        pub_rating["avg_rating"].max() + 0.3,
    )
    ax.set_title(f"⭐ Publisher Average Rating (min. {min_count} comics)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Rating (out of 10)")
    ax.legend()
    for bar, val in zip(bars, pub_rating["avg_rating"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_decade_trends(df: pd.DataFrame) -> go.Figure:
    """Plotly dual-axis chart: comic count per decade + avg rating line."""
    agg_dict = {"count": ("title", "count"), "avg_rating": ("rating", "mean")}
    decade_data = (
        df.groupby("decade").agg(**agg_dict)
        .dropna().reset_index()
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=decade_data["decade"].astype(str), y=decade_data["count"],
               name="Comic Count", marker_color=HERO_BLUE, opacity=0.8),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=decade_data["decade"].astype(str), y=decade_data["avg_rating"],
                   name="Avg Rating", line=dict(color=HERO_RED, width=2.5),
                   mode="lines+markers"),
        secondary_y=True,
    )
    fig.update_layout(
        title="📅 Comics Published Per Decade + Avg Rating Trend",
        **PLOTLY_LAYOUT, title_font_size=16, xaxis_title="Decade",
    )
    fig.update_yaxes(title_text="Number of Comics", secondary_y=False)
    fig.update_yaxes(title_text="Average Rating", secondary_y=True)
    return fig


def plot_genre_publisher_heatmap(
    df: pd.DataFrame, top_n: int = 8
) -> plt.Figure:
    """Seaborn heatmap of comic counts by Publisher × Genre."""
    top_pubs   = df["publisher"].value_counts().head(top_n).index
    top_genres = df["genre"].value_counts().head(top_n).index
    heat_data  = (
        df[df["publisher"].isin(top_pubs) & df["genre"].isin(top_genres)]
        .groupby(["publisher", "genre"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(heat_data, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                linewidths=0.4, linecolor=BG_DARK, cbar_kws={"shrink": 0.8})
    ax.set_title("🗺️  Publisher × Genre Distribution Matrix",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Publisher")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    return fig


def plot_era_analysis(df: pd.DataFrame) -> tuple[plt.Figure, plt.Figure]:
    """
    Returns two figures:
        fig1 — Bar chart of comics per era with dual-axis avg rating
        fig2 — Violin plot of rating distribution by era
    """
    era_order = ["Golden Age", "Silver Age", "Bronze Age",
                 "Modern Age", "21st Century", "Contemporary"]

    era_data = (
        df.groupby("era", observed=True)
        .agg(count=("title", "count"), avg_rating=("rating", "mean"))
        .reindex([e for e in era_order if e in df["era"].cat.categories])
        .dropna()
    )

    # ── Fig 1: bar + dual axis ────────────────────────────────────────────────
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle("🕰️ Era & Theme Breakdown", fontsize=15, fontweight="bold")

    x_pos = range(len(era_data))
    bars  = axes[0].bar(x_pos, era_data["count"], color=PALETTE[:len(era_data)],
                        edgecolor=BG_DARK, linewidth=0.5)
    ax2   = axes[0].twinx()
    ax2.plot(x_pos, era_data["avg_rating"], "o--", color="yellow",
             linewidth=2, markersize=7, label="Avg Rating")
    ax2.set_ylabel("Avg Rating", color="yellow")
    ax2.tick_params(axis="y", labelcolor="yellow")
    axes[0].set_xticks(list(x_pos))
    axes[0].set_xticklabels(era_data.index, rotation=30, ha="right")
    axes[0].set_title("Comics per Era + Avg Rating", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.2)
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 2,
                         f"{int(h):,}", ha="center", va="bottom", fontsize=7)

    if "theme" in df.columns:
        theme_counts = df["theme"].value_counts().head(12)
        colors_t     = plt.cm.plasma(np.linspace(0.2, 0.9, len(theme_counts)))
        axes[1].barh(theme_counts.index[::-1], theme_counts.values[::-1], color=colors_t)
        axes[1].set_title("Top 12 Themes (Color Style)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Number of Comics")
        axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()

    # ── Fig 2: violin ─────────────────────────────────────────────────────────
    era_rating_df = df[df["era"].notna()][["era", "rating"]].dropna()
    era_groups    = [era_rating_df[era_rating_df["era"] == e]["rating"].values
                     for e in era_order if e in era_rating_df["era"].values]
    era_labels_   = [e for e in era_order if e in era_rating_df["era"].values]

    fig2, ax = plt.subplots(figsize=(12, 5))
    parts = ax.violinplot(era_groups, positions=range(len(era_groups)),
                          showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], PALETTE):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    ax.set_xticks(range(len(era_labels_)))
    ax.set_xticklabels(era_labels_, rotation=20, ha="right")
    ax.set_title("🎻 Rating Distribution by Era", fontsize=14, fontweight="bold")
    ax.set_ylabel("Rating (out of 10)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return fig1, fig2


# ── Section 6 — Advanced Analysis ────────────────────────────────────────────

def plot_cluster_pca(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    best_k: int,
    inertias: list[float],
    k_range: range,
) -> plt.Figure:
    """PCA scatter coloured by cluster, alongside the elbow curve."""
    pca       = PCA(n_components=2)
    X_2d      = pca.fit_transform(X_scaled)
    total_var = pca.explained_variance_ratio_.sum()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(list(k_range), inertias, "o-", color=HERO_BLUE,
                 linewidth=2, markersize=7)
    axes[0].axvline(best_k, color=HERO_RED, linestyle="--",
                    linewidth=1.5, label=f"Selected K={best_k}")
    axes[0].set_title("🔍 Elbow Method — Optimal K", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    scatter_colors = [PALETTE[l % len(PALETTE)] for l in labels]
    axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=scatter_colors,
                    alpha=0.5, s=14, edgecolors="none")
    axes[1].set_title(
        f"🎯 KMeans (K={best_k}) via PCA  [{total_var:.1%} variance explained]",
        fontsize=12, fontweight="bold",
    )
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[1].grid(alpha=0.2)
    for k in range(best_k):
        axes[1].scatter([], [], color=PALETTE[k % len(PALETTE)],
                        label=f"Cluster {k + 1}", s=60)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_model_comparison(results: dict) -> plt.Figure:
    """
    3-panel bar chart: RMSE, R², CV-R² across all models.
    Best model bar highlighted per panel.
    Bar labels handle negative values correctly.
    """
    model_names = list(results.keys())
    rmse_vals   = [results[n]["RMSE"]  for n in model_names]
    r2_vals     = [results[n]["R2"]    for n in model_names]
    cv_vals     = [results[n]["CV_R2"] for n in model_names]
    best_idx    = r2_vals.index(max(r2_vals))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, vals, title, ylabel, hi_color in zip(
        axes,
        [rmse_vals, r2_vals, cv_vals],
        ["RMSE (lower = better)", "R² Score (higher = better)", "CV R² — 5-Fold"],
        ["RMSE", "R²", "CV R²"],
        [HERO_BLUE, HERO_GRN, HERO_RED],
    ):
        bar_colors = [hi_color if i == best_idx else DIM_BAR
                      for i in range(len(model_names))]
        ax.bar(model_names, vals, color=bar_colors)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            offset = abs(max(vals, default=0.01)) * 0.03
            va     = "bottom" if v >= 0 else "top"
            ax.text(i, v + (offset if v >= 0 else -offset),
                    f"{v:.3f}", ha="center", va=va, fontsize=9)

    best_name = model_names[best_idx]
    plt.suptitle(f"🤖 Model Performance Comparison  (Best: {best_name})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def plot_tfidf_wordcloud(
    top_words: list[tuple[str, float]],
    top_df: pd.DataFrame,
) -> plt.Figure:
    """WordCloud + TF-IDF bar chart side by side."""
    wc_freq = {w: s for w, s in top_words}
    wc = WordCloud(background_color=BG_PANEL, colormap="Blues",
                   width=600, height=400, max_words=100
                   ).generate_from_frequencies(wc_freq)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(wc, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("☁️ Title Word Cloud (TF-IDF)", fontsize=14, fontweight="bold")

    axes[1].barh(top_df["term"][::-1], top_df["score"][::-1],
                 color=HERO_BLUE, alpha=0.9)
    axes[1].set_title("Top 25 Title Terms by TF-IDF Score",
                      fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Mean TF-IDF Score")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig
