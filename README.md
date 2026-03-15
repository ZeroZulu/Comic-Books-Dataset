# 🦸 Comic Books Dataset — From Panels to Patterns

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/rudrakumargupta/comic-books-dataset-10000-entries)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

> **A complete, end-to-end data science project on 10,000 comic books — from raw data and exploratory analysis through NLP, machine learning, and a content-based recommender system.**

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Analysis Pipeline](#-analysis-pipeline)
- [Key Findings](#-key-findings)
- [Installation](#-installation)
- [Usage](#-usage)
- [Module Reference](#-module-reference)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project transforms a 10,000-entry comic books dataset into a full data science showcase.  
It covers **17 distinct analysis sections** across Basic, Intermediate, and Advanced tiers.

**What makes this project unique:**

| Feature | Detail |
|---|---|
| 🎨 Dark-themed visualizations | GitHub-dark palette throughout all plots |
| 🧹 Robust cleaning pipeline | Handles non-standard column names, NaN guards, proper encoding |
| 📊 Skewness annotations | Every histogram annotated with skew + kurtosis |
| 🗺️ Geographic analysis | Country × Genre matrix, language distribution |
| 🕰️ Era analysis | Violin plots + dual-axis bar charts across comic eras |
| 🤖 4-model ML comparison | Linear, Ridge, Random Forest, Gradient Boosting with 5-fold CV |
| 🔗 Content-based recommender | TF-IDF + categorical + numeric cosine similarity |
| 💾 Zero duplicate computation | Similarity matrix computed once and reused |

---

## 📦 Dataset

**Source:** [Comic Books Dataset (10,000 entries) — Kaggle](https://www.kaggle.com/datasets/rudrakumargupta/comic-books-dataset-10000-entries)

| Raw Column | Renamed To | Type | Notes |
|---|---|---|---|
| `Title` | `title` | string | Comic book title |
| `Studio/Publisher` | `publisher` | string | Publishing house |
| `Genre` | `genre` | string | Genre category |
| `Writer` | `writer` | string | Primary writer |
| `Artist` | `artist` | string | Penciler/illustrator |
| `Release Year` | `year` | int | Publication year (2000–2026) |
| `Format` | `format` | string | Standard, TPB, HC, etc. |
| `Theme (Color Style)` | `theme` | string | Visual style category |
| `Country of Origin` | `country` | string | Country of publication |
| `Page Count` | `pages` | int | Page count (48–14,400) |
| `Rating (out of 10)` | `rating` | float | Community rating (6.0–9.9) |
| `Status` | `status` | string | Ongoing / Completed / Hiatus |
| `Language` | `language` | string | Publication language |
| `Age Rating` | `age_rating` | string | All Ages / Teen / Mature |
| `Awards` | `awards` | string | Award entries (39.8% have awards) |
| `Volume Count` | `volume_count` | int | Number of volumes (1–56) |

**Key statistics:** 10,000 rows · 17 raw columns · 6,016 missing cells · 0 duplicates  
**Engineered features:** `decade`, `era`, `rating_label`, `title_word_count`

---

## 📁 Project Structure

```
comic-books-project/
│
├── 📓 notebooks/
│   └── comic_books_analysis.ipynb    ← Main Kaggle/Jupyter notebook (33 cells)
│
├── 🐍 src/
│   ├── __init__.py
│   ├── cleaning.py                   ← Full cleaning & feature engineering pipeline
│   ├── visualizations.py             ← Reusable plot functions (dark theme)
│   ├── models.py                     ← ML training, CV evaluation, comparison
│   └── recommender.py                ← ComicRecommender class (cosine similarity)
│
├── 🧪 tests/
│   ├── test_cleaning.py
│   ├── test_recommender.py
│   └── test_models.py
│
├── 📊 outputs/
│   ├── figures/                      ← Exported chart PNGs
│   └── models/                       ← Saved model artifacts (.pkl)
│
├── 📄 requirements.txt
├── 📄 .gitignore
├── 📄 LICENSE
└── 📄 README.md
```

---

## 🔬 Analysis Pipeline

```
Raw CSV (10,000 rows × 17 cols)
         │
         ▼
┌─────────────────────────────────────────────┐
│  SECTION 2 — DATA LOADING & OVERVIEW        │
│  • Shape, dtypes, memory usage              │
│  • Missing value bar chart + duplicate pie  │
│  • describe(include='all')                  │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  SECTION 3 — CLEANING & FEATURE ENGINEERING │
│  • Column name standardisation              │
│  • Rename map for non-standard columns      │
│  • Numeric coercion with error='coerce'     │
│  • decade, era, rating_label, word_count    │
│  • Rating bins aligned to actual range 6–10 │
└──────────────────────┬──────────────────────┘
                       │
         ┌─────────────┴──────────────┐
         ▼                            ▼
┌──────────────────┐       ┌──────────────────────────┐
│  SECTION 4       │       │  SECTION 5               │
│  BASIC ANALYSIS  │       │  INTERMEDIATE ANALYSIS   │
│                  │       │                          │
│  4.1 Publisher   │       │  5.1 Publisher × Rating  │
│      bar chart   │       │      (xlim zoomed in)    │
│  4.2 Genre donut │       │  5.2 Decade dual-axis    │
│  4.3 Histograms  │       │  5.3 Pages vs Rating OLS │
│      + skew/kurt │       │  5.4 Writers & Artists   │
│  4.4 Correlation │       │  5.5 Genre×Publisher heat│
│      heatmap     │       │  5.6 Age Rating + Status │
└──────────────────┘       │  5.7 Country + Language  │
                           │  5.8 Era + Theme + Violin│
                           └──────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────┐
                    │  SECTION 6 — ADVANCED ANALYSIS   │
                    │                                  │
                    │  6.1 TF-IDF WordCloud + bar      │
                    │  6.2 KMeans (auto-K) + PCA 2D    │
                    │      → cluster interpretation    │
                    │  6.3 4-model comparison          │
                    │      Linear/Ridge/RF/GBM + CV    │
                    │      feature importance + A vs P │
                    │  6.4 Content-based Recommender   │
                    │      TF-IDF + cat + numeric      │
                    │  6.5 Title similarity by pub     │
                    │      (reuses cell 6.4 matrix)    │
                    │  6.6 Publication volume by year  │
                    └──────────────────────────────────┘
```

---

## 💡 Key Findings

| # | Area | Finding |
|---|---|---|
| 1 | **Publisher** | Marvel leads with 1,218 titles (12.2%); DC and Shogakukan are distant 2nd/3rd |
| 2 | **Genre** | Top genres span Western superhero and Japanese manga — an internationally diverse dataset |
| 3 | **Ratings** | Tightly clustered 7.5–9.0 (mean 8.06, std 0.53), near-zero skew — likely curated/synthetic |
| 4 | **Era** | Contemporary-era comics dominate volume; Modern Age shows highest avg ratings |
| 5 | **Geography** | USA & Japan dominate; English & Japanese lead languages; Country × Genre matrix confirms style clusters |
| 6 | **Awards** | 3,984 comics (39.8%) carry award entries across 15 unique award categories |
| 7 | **Clustering** | Auto-elbow K=3: *Epic Collections* (high pages), *Vintage Titles* (pre-2010), *Standard Volumes* |
| 8 | **ML Prediction** | All 4 models achieve R² ≈ 0.03–0.05 — ratings are **not** predictable from metadata alone |
| 9 | **Recommender** | Cosine similarity scores up to 0.68+; thematically coherent results across genre/publisher |
| 10 | **Title Diversity** | Indie/smaller publishers show *lower* avg title similarity — more diverse catalogs |

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/comic-books-project.git
cd comic-books-project

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Notebook Locally
```bash
jupyter lab notebooks/comic_books_analysis.ipynb
```

### Run on Kaggle
1. Go to the [dataset page](https://www.kaggle.com/datasets/rudrakumargupta/comic-books-dataset-10000-entries)
2. Click **New Notebook** → upload `notebooks/comic_books_analysis.ipynb`
3. **Run All ▶️**

### Use the Modules Directly

```python
import pandas as pd
from src.cleaning import clean_pipeline
from src.models import train_and_evaluate
from src.recommender import ComicRecommender

# Load and clean
df = pd.read_csv("data/comic_books_10000_dataset.csv")
df_clean = clean_pipeline(df)

# Train models
results = train_and_evaluate(df_clean)

# Get recommendations
rec = ComicRecommender()
rec.fit(df_clean)
print(rec.recommend("Amazing Spider-Man", n=5))
```

---

## 📚 Module Reference

### `src/cleaning.py`
| Function | Description |
|---|---|
| `standardise_columns(df)` | Lowercase, strip, unify column names |
| `coerce_numerics(df)` | Force numeric columns to float, coerce errors to NaN |
| `engineer_features(df)` | Add `decade`, `era`, `rating_label`, `title_word_count` |
| `clean_pipeline(df)` | Full pipeline: standardise → rename → coerce → dedupe → engineer |

### `src/visualizations.py`
| Function | Description |
|---|---|
| `plot_publisher_distribution(df)` | Bar chart with above-mean highlight + mean line |
| `plot_numerical_distributions(df)` | Histograms with skew/kurt annotations |
| `plot_genre_heatmap(df)` | Publisher × Genre seaborn heatmap |
| `plot_era_analysis(df)` | Dual-axis bar + violin rating distribution |
| `plot_cluster_pca(X_scaled, labels)` | PCA scatter coloured by cluster |

### `src/models.py`
| Function | Description |
|---|---|
| `prepare_features(df)` | Encode categoricals (OrdinalEncoder, no leakage), return X, y |
| `train_and_evaluate(df)` | Train 4 models, return metrics dict with RMSE, MAE, R², CV-R² |
| `best_model(results)` | Return name + dict of best model by R² |
| `plot_model_comparison(results)` | 3-panel bar chart: RMSE, R², CV-R² |

### `src/recommender.py`
| Method | Description |
|---|---|
| `ComicRecommender(top_n, tfidf_features)` | Constructor |
| `.fit(df)` | Build feature matrix + cosine similarity matrix |
| `.recommend(title, n)` | Return top-n similar comics as DataFrame with similarity score |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-analysis`
3. Commit your changes: `git commit -m "Add: new clustering analysis"`
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

*Made with ❤️ and too many comic books.*
