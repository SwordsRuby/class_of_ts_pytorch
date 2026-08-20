"""Классическая модель машинного обучения на базе Scikit-learn.

Поддерживает LogisticRegression и SVM (SVC) с общим интерфейсом
обучения, предсказания и расчёта вероятностей.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from src import config


class SklearnClassifier:
    """Обёртка над моделями Scikit-learn для многоклассовой классификации.

    Args:
        model_type: тип модели — ``"logistic"`` или ``"svm"``.
        categories: упорядоченный список категорий (меток классов).
        random_state: сид для воспроизводимости.
    """

    def __init__(self, model_type: str = "logistic", categories: list[str] | None = None, random_state: int | None = None):
        self.model_type = model_type
        self.categories: list[str] = categories or list(config.CATEGORIES)
        self.random_state = random_state if random_state is not None else config.RANDOM_SEED
        self._model: Any = self._build_model()

    def _build_model(self) -> Any:
        """Создаёт базовую модель заданного типа."""
        if self.model_type == "svm":
            base = SVC(kernel="linear", probability=True, random_state=self.random_state)
            # One-vs-Rest для корректных вероятностей при мультиклассе.
            return OneVsRestClassifier(base)
        # По умолчанию — логистическая регрессия (мультикласс по умолчанию).
        return LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnClassifier":
        """Обучает модель на матрице признаков и метках.

        Args:
            X: матрица признаков (обучающие тексты, векторизованные).
            y: массив меток категорий.

        Returns:
            Сам объект модели.
        """
        self._model.fit(X, y)
        # sklearn возвращает метки как строки; порядок колонок predict_proba
        # совпадает с порядком ``classes_``, полученным на обучающих данных.
        self._classes: list[str] = list(self._model.classes_)
        return self

    def get_probability_labels(self) -> list[str]:
        """Возвращает порядок классов, соответствующий колонкам predict_proba.

        Returns:
            Список категорий в порядке колонок вероятностей.
        """
        return self._classes if hasattr(self, "_classes") else self.categories

    def predict(self, X: np.ndarray) -> list[str]:
        """Предсказывает категории для матрицы признаков.

        Args:
            X: матрица признаков.

        Returns:
            Список предсказанных категорий.
        """
        return [str(label) for label in self._model.predict(X)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает вероятности принадлежности каждому классу.

        Args:
            X: матрица признаков.

        Returns:
            Массив вероятностей формы (n_samples, n_classes).
            Колонки соответствуют :meth:`get_probability_labels`.
        """
        return np.asarray(self._model.predict_proba(X))

    def save(self, path=None) -> str:
        """Сохраняет модель в файл через joblib.

        Args:
            path: путь сохранения (по умолчанию ``config.SKLEARN_MODEL_PATH``).

        Returns:
            Путь к сохранённому файлу.
        """
        path = path or config.SKLEARN_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return str(path)

    @staticmethod
    def load(path=None) -> "SklearnClassifier":
        """Загружает модель из файла.

        Args:
            path: путь к файлу модели.

        Returns:
            Загруженный экземпляр ``SklearnClassifier``.
        """
        path = path or config.SKLEARN_MODEL_PATH
        return joblib.load(path)