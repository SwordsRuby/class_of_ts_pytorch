"""Векторизация текстов (TF-IDF / Bag-of-Words).

Обеспечивает преобразование предобработанных текстов в числовые
векторы признаков, используемые моделями классификации.
"""
from __future__ import annotations

from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer

from src import config


def build_tfidf_vectorizer(max_features: int | None = None, ngram_range: tuple[int, int] | None = None) -> TfidfVectorizer:
    """Создаёт конфигурируемый TF-IDF векторизатор.

    Args:
        max_features: максимальное число признаков (по умолчанию
            ``config.MAX_FEATURES``).
        ngram_range: диапазон n-грамм (по умолчанию ``config.NGRAM_RANGE``).

    Returns:
        Настроенный экземпляр ``TfidfVectorizer``.
    """
    return TfidfVectorizer(
        lowercase=False,  # тексты уже приведены к нижнему регистру в предобработке
        max_features=max_features or config.MAX_FEATURES,
        ngram_range=ngram_range or config.NGRAM_RANGE,
        token_pattern=r"(?u)\b\w+\b",
    )


def fit_vectorizer(vectorizer: TfidfVectorizer, texts: Iterable[str]):
    """Обучает векторизатор на корпусе текстов.

    Args:
        vectorizer: экземпляр ``TfidfVectorizer``.
        texts: обучающие тексты.

    Returns:
        Матрица признаков для обучающих текстов.
    """
    return vectorizer.fit_transform(texts)


def transform_texts(vectorizer: TfidfVectorizer, texts: Iterable[str]):
    """Преобразует тексты в матрицу признаков с помощью обученного векторизатора.

    Args:
        vectorizer: обученный ``TfidfVectorizer``.
        texts: тексты для преобразования.

    Returns:
        Матрица признаков (sparse matrix).
    """
    return vectorizer.transform(texts)