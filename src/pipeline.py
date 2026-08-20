"""Оркестрация полного пайплайна классификации.

Связывает этапы AGENTS.md в единый поток:
данные -> предобработка -> векторизация -> обучение -> оценка -> инференс.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
from sklearn.model_selection import train_test_split

from src import config
from src.data_generation import get_or_generate_dataset
from src.evaluation import evaluate_predictions, format_metrics_short, save_confusion_matrix
from src.features import build_tfidf_vectorizer, fit_vectorizer
from src.models.pytorch_mlp import PyTorchClassifier
from src.models.sklearn_model import SklearnClassifier
from src.preprocessing import preprocess_corpus


@dataclass
class PipelineResults:
    """Сводные результаты обучения и оценки моделей.

    Attributes:
        X_test: признаки тестовой выборки.
        y_test: истинные метки тестовой выборки.
        sklearn_metrics: метрики Scikit-learn модели.
        pytorch_metrics: метрики PyTorch модели.
        categories: упорядоченный список категорий.
    """

    X_test: Any = None
    y_test: Any = None
    sklearn_metrics: dict[str, Any] | None = None
    pytorch_metrics: dict[str, Any] | None = None
    categories: list[str] = field(default_factory=lambda: list(config.CATEGORIES))

    def summary(self) -> str:
        """Возвращает краткую сводку по обеим моделям."""
        lines = ["Сводка результатов:"]
        lines.append(f"  Scikit-learn: {format_metrics_short(self.sklearn_metrics) if self.sklearn_metrics else 'не обучена'}")
        lines.append(f"  PyTorch MLP:  {format_metrics_short(self.pytorch_metrics) if self.pytorch_metrics else 'не обучена'}")
        return "\n".join(lines)


class TextClassificationPipeline:
    """Оркестратор пайплайна классификации текстовых заявок.

    Args:
        train_sklearn: обучать ли Scikit-learn модель.
        train_pytorch: обучать ли PyTorch модель.
    """

    def __init__(self, train_sklearn: bool = True, train_pytorch: bool = True):
        config.ensure_directories()
        self.train_sklearn = train_sklearn
        self.train_pytorch = train_pytorch
        self.categories: list[str] = list(config.CATEGORIES)
        self.vectorizer = None
        self.sklearn_model: SklearnClassifier | None = None
        self.pytorch_model: PyTorchClassifier | None = None

    def run(self, regenerate: bool = False) -> PipelineResults:
        """Выполняет весь пайплайн и возвращает результаты.

        Args:
            regenerate: перегенерировать ли датасет принудительно.

        Returns:
            ``PipelineResults`` с метриками и тестовыми данными.
        """
        # 1. Данные.
        df = get_or_generate_dataset()
        if regenerate:
            from src.data_generation import generate_dataset, save_dataset

            df = generate_dataset()
            save_dataset(df)

        # 2. Предобработка.
        texts = preprocess_corpus(df["text"].tolist(), remove_stopwords_flag=config.REMOVE_STOPWORDS)
        labels = df["label"].tolist()

        # Разделение 80/20.
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_SEED,
            stratify=labels,
        )

        # 3. Векторизация (TF-IDF) на тренировочной части.
        self.vectorizer = build_tfidf_vectorizer()
        X_train = fit_vectorizer(self.vectorizer, X_train_texts)
        X_test = self.vectorizer.transform(X_test_texts)

        # Сохраняем векторизатор для последующего инференса.
        config.VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, config.VECTORIZER_PATH)

        results = PipelineResults(X_test=X_test, y_test=y_test, categories=self.categories)

        # 4. Обучение и оценка моделей.
        if self.train_sklearn:
            self.sklearn_model = self._train_sklearn(X_train, y_train)
            y_pred = self.sklearn_model.predict(X_test)
            results.sklearn_metrics = evaluate_predictions(y_test, y_pred, self.categories)
            self._persist_sklearn(results.sklearn_metrics)

        if self.train_pytorch:
            self.pytorch_model = self._train_pytorch(X_train, y_train)
            y_pred = self.pytorch_model.predict(X_test)
            results.pytorch_metrics = evaluate_predictions(y_test, y_pred, self.categories)
            self._persist_pytorch(results.pytorch_metrics)

        print(results.summary())
        return results

    def _train_sklearn(self, X_train, y_train) -> SklearnClassifier:
        """Обучает Scikit-learn модель."""
        print("\n[Scikit-learn] Обучение модели...")
        model = SklearnClassifier(model_type=config.SKLEARN_MODEL_TYPE, categories=self.categories)
        model.fit(X_train, y_train)
        model.save()
        print(f"[Scikit-learn] Модель сохранена: {config.SKLEARN_MODEL_PATH}")
        return model

    def _train_pytorch(self, X_train, y_train) -> PyTorchClassifier:
        """Обучает PyTorch MLP модель."""
        print("\n[PyTorch] Обучение MLP модели...")
        model = PyTorchClassifier(input_dim=X_train.shape[1], categories=self.categories)
        model.fit(X_train, y_train)
        model.save()
        print(f"[PyTorch] Модель сохранена: {config.PYTORCH_MODEL_PATH}")
        return model

    def _persist_sklearn(self, metrics: dict[str, Any]) -> None:
        """Сохраняет отчёт и матрицу ошибок Scikit-learn модели."""
        _save_report_and_cm(metrics, config.SKLEARN_REPORT_PATH, config.SKLEARN_CM_PATH, "Scikit-learn")

    def _persist_pytorch(self, metrics: dict[str, Any]) -> None:
        """Сохраняет отчёт и матрицу ошибок PyTorch модели."""
        _save_report_and_cm(metrics, config.PYTORCH_REPORT_PATH, config.PYTORCH_CM_PATH, "PyTorch MLP")


def _save_report_and_cm(metrics: dict[str, Any], report_path: Path, cm_path: Path, title: str) -> None:
    """Сохраняет текстовый отчёт и изображение матрицы ошибок."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(metrics["report_text"])

    save_confusion_matrix(metrics["confusion_matrix"], metrics["categories"], cm_path, title=title)
    print(f"Отчёт сохранён: {report_path}")
    print(f"Матрица ошибок сохранена: {cm_path}")


def run_pipeline(train_sklearn: bool = True, train_pytorch: bool = True, regenerate: bool = False) -> PipelineResults:
    """Запускает полный пайплайн классификации.

    Args:
        train_sklearn: обучать ли Scikit-learn модель.
        train_pytorch: обучать ли PyTorch модель.
        regenerate: перегенерировать ли датасет.

    Returns:
        ``PipelineResults`` с результатами.
    """
    pipeline = TextClassificationPipeline(train_sklearn=train_sklearn, train_pytorch=train_pytorch)
    return pipeline.run(regenerate=regenerate)