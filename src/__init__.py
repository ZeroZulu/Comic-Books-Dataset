"""
Comic Books Analysis — Source Package
======================================
Modules:
    cleaning       — Data cleaning and feature engineering pipeline
    visualizations — Reusable dark-themed plot functions
    models         — ML training, evaluation and comparison
    recommender    — Content-based recommender system
"""

from .cleaning import clean_pipeline
from .recommender import ComicRecommender

__all__ = ["clean_pipeline", "ComicRecommender"]
__version__ = "1.0.0"
