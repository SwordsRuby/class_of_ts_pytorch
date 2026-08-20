"""Оценка качества классификации.

Вычисляет метрики (Accuracy, Precision, Recall, F1-score) и строит
матрицу ошибок (confusion matrix) для визуализации.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # без графического интерфейса

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src import config


def evaluate_predictions(y_true, y_pred, categories: list[str] | None = None) -> dict[str, Any]:
    """Вычисляет метрики качества классификации.

    Args:
        y_true: истинные метки.
        y_pred: предсказанные метки.
        categories: упорядоченный список категорий.

    Returns:
        Словарь с accuracy, classification_report и confusion matrix.
    """
    categories = categories or list(config.CATEGORIES)

    report = classification_report(y_true, y_pred, labels=categories, zero_division=0, output_dict=True)
    report_text = classification_report(y_true, y_pred, labels=categories, zero_division=0)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "report": report,
        "report_text": report_text,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=categories),
        "categories": categories,
    }


def save_confusion_matrix(
    cm: np.ndarray,
    categories: list[str],
    path: Path | None = None,
    title: str = "Confusion Matrix",
) -> Path:
    """Строит и сохраняет тепловую карту матрицы ошибок.

    Args:
        cm: матрица ошибок.
        categories: подписи классов.
        path: путь сохранения изображения.
        title: заголовок графика.

    Returns:
        Путь к сохранённому изображению.
    """
    path = path or (config.RESULTS_DIR / "confusion_matrix.png")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(categories)),
        yticks=np.arange(len(categories)),
        xticklabels=categories,
        yticklabels=categories,
        title=title,
        ylabel="Истинная категория",
        xlabel="Предсказанная категория",
    )

    # Поворот подписей оси X для читаемости.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Подписи значений внутри ячеек.
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def format_metrics_short(metrics: dict[str, Any]) -> str:
    """Возвращает краткую строку с основными метриками.

    Args:
        metrics: словарь результатов :func:`evaluate_predictions`.

    Returns:
        Строка вида ``Accuracy=0.90 Macro F1=0.89``.
    """
    report = metrics["report"]
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    return f"Accuracy={metrics['accuracy']:.4f} Macro F1={macro_f1:.4f}"