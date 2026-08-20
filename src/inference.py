"""Инференс (предсказание) для новых текстов.

Реализует функцию ``predict(text)``, которая принимает новый текст и
возвращает категорию и вероятность (confidence). Для работы требуется
обученная модель и обученный TF-IDF векторизатор.
"""
from __future__ import annotations

import joblib
from dataclasses import dataclass
from typing import Any

from src import config
from src.models.pytorch_mlp import PyTorchClassifier
from src.models.sklearn_model import SklearnClassifier
from src.preprocessing import preprocess_text


@dataclass
class PredictionResult:
    """Результат предсказания категории текста.

    Attributes:
        text: исходный текст.
        category: предсказанная категория.
        probability: вероятность (confidence) предсказания.
        probabilities: вероятности по всем категориям (категория -> float).
    """

    text: str
    category: str
    probability: float
    probabilities: dict[str, float]


class TextClassifier:
    """Инференс-обёртка, объединяющая векторизатор и модель.

    Args:
        vectorizer: обученный ``TfidfVectorizer``.
        model: обученная модель (SklearnClassifier или PyTorchClassifier).
        model_type: тип модели (``"sklearn"`` или ``"pytorch"``).
    """

    def __init__(self, vectorizer, model: Any, model_type: str = "sklearn"):
        self.vectorizer = vectorizer
        self.model = model
        self.model_type = model_type
        self.categories: list[str] = getattr(model, "categories", list(config.CATEGORIES))

    def predict(self, text: str, remove_stopwords: bool | None = None) -> PredictionResult:
        """Предсказывает категорию для одного текста.

        Args:
            text: новый текст заявки.
            remove_stopwords: удалять ли стоп-слова при предобработке
                (по умолчанию ``config.REMOVE_STOPWORDS``).

        Returns:
            ``PredictionResult`` с категорией и вероятностью.
        """
        if remove_stopwords is None:
            remove_stopwords = config.REMOVE_STOPWORDS

        processed = preprocess_text(text, remove_stopwords_flag=remove_stopwords)
        vector = self.vectorizer.transform([processed])

        probabilities = self.model.predict_proba(vector)[0]
        # Порядок колонок вероятностей определяется моделью.
        probability_labels = self.model.get_probability_labels()
        best_idx = int(probabilities.argmax())

        return PredictionResult(
            text=text,
            category=probability_labels[best_idx],
            probability=float(probabilities[best_idx]),
            probabilities={cat: float(p) for cat, p in zip(probability_labels, probabilities)},
        )

    def predict_many(self, texts: list[str], remove_stopwords: bool | None = None) -> list[PredictionResult]:
        """Предсказывает категории для списка текстов.

        Args:
            texts: список текстов.
            remove_stopwords: удалять ли стоп-слова.

        Returns:
            Список ``PredictionResult``.
        """
        return [self.predict(text, remove_stopwords) for text in texts]


def load_classifier(
    vectorizer_path=None,
    model_path=None,
    model_type: str = "sklearn",
) -> TextClassifier:
    """Загружает векторизатор и модель из файлов.

    Args:
        vectorizer_path: путь к векторизатору (по умолчанию
            ``config.VECTORIZER_PATH``).
        model_path: путь к модели (по умолчанию путь соответствующего типа).
        model_type: тип модели (``"sklearn"`` или ``"pytorch"``).

    Returns:
        Готовый к работе ``TextClassifier``.
    """
    vectorizer_path = vectorizer_path or config.VECTORIZER_PATH
    vectorizer = joblib.load(vectorizer_path)

    if model_type == "pytorch":
        model_path = model_path or config.PYTORCH_MODEL_PATH
        model = PyTorchClassifier.load(model_path)
    else:
        model_path = model_path or config.SKLEARN_MODEL_PATH
        model = SklearnClassifier.load(model_path)

    return TextClassifier(vectorizer=vectorizer, model=model, model_type=model_type)


def predict(text: str, model_type: str = "sklearn") -> PredictionResult:
    """Быстрая функция предсказания с загрузкой стандартных артефактов.

    Args:
        text: новый текст заявки.
        model_type: тип модели (``"sklearn"`` или ``"pytorch"``).

    Returns:
        ``PredictionResult`` с категорией и вероятностью.

    Raises:
        FileNotFoundError: если модель/векторизатор не обучены.
    """
    classifier = load_classifier(model_type=model_type)
    return classifier.predict(text)