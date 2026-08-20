"""Пакет моделей классификации.

Экспортирует классическую (Scikit-learn) и нейросетевую (PyTorch MLP)
модели с единым интерфейсом ``fit`` / ``predict`` / ``predict_proba``.
"""
from src.models.pytorch_mlp import MLPClassifier, PyTorchClassifier
from src.models.sklearn_model import SklearnClassifier

__all__ = ["SklearnClassifier", "PyTorchClassifier", "MLPClassifier"]