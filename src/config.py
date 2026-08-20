"""Централизованная конфигурация проекта.

Содержит категории классификации, пути к директориям и гиперпараметры
моделей. Используется всеми модулями проекта для воспроизводимости
и удобства настройки.
"""
from __future__ import annotations

from pathlib import Path

# --- Корневая директория проекта -------------------------------------------
# Определяется относительно расположения данного файла (src/config.py).
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# --- Категории классификации -----------------------------------------------
CATEGORIES: list[str] = [
    "Веб-разработка",
    "Мобильное приложение",
    "Дизайн",
    "Аналитика",
    "Маркетинг",
]

# --- Пути к данным и результатам -------------------------------------------
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = BASE_DIR / "results"

DATASET_PATH: Path = RAW_DATA_DIR / "requests_dataset.csv"

# Пути для сохранённых артефактов моделей и векторизатора.
SKLEARN_MODEL_PATH: Path = PROCESSED_DATA_DIR / "sklearn_model.joblib"
PYTORCH_MODEL_PATH: Path = PROCESSED_DATA_DIR / "pytorch_model.pt"
VECTORIZER_PATH: Path = PROCESSED_DATA_DIR / "tfidf_vectorizer.joblib"

# Пути к отчётам и визуализациям.
SKLEARN_REPORT_PATH: Path = RESULTS_DIR / "sklearn_report.txt"
PYTORCH_REPORT_PATH: Path = RESULTS_DIR / "pytorch_report.txt"
SKLEARN_CM_PATH: Path = RESULTS_DIR / "sklearn_confusion_matrix.png"
PYTORCH_CM_PATH: Path = RESULTS_DIR / "pytorch_confusion_matrix.png"

# --- Параметры датасета -----------------------------------------------------
EXAMPLES_PER_CATEGORY: int = 10
RANDOM_SEED: int = 42

# --- Параметры предобработки и векторизации ---------------------------------
REMOVE_STOPWORDS: bool = True
MAX_FEATURES: int = 1000
NGRAM_RANGE: tuple[int, int] = (1, 2)

# --- Параметры обучения -----------------------------------------------------
TEST_SIZE: float = 0.2  # 80% train / 20% test
VAL_SIZE: float = 0.0   # Доля валидационной выборки (0 — не используется)

# Гиперпараметры PyTorch MLP.
PYTORCH_HIDDEN_SIZES: list[int] = [128, 64]  # 2 скрытых слоя
PYTORCH_EPOCHS: int = 200
PYTORCH_BATCH_SIZE: int = 8
PYTORCH_LEARNING_RATE: float = 1e-3
PYTORCH_WEIGHT_DECAY: float = 1e-4

# Гиперпараметры Scikit-learn модели.
SKLEARN_MODEL_TYPE: str = "logistic"  # "logistic" | "svm"


def ensure_directories() -> None:
    """Создаёт все необходимые директории проекта, если они отсутствуют."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)