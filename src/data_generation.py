"""Генерация синтетического датасета текстовых заявок.

Создаёт 50 примеров (по 10 на каждую из 5 категорий) на основе шаблонов
и ключевых слов, типичных для заявок каждой категории. Датасет
сохраняется в CSV-файл и возвращается в виде DataFrame.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src import config

# Ключевые слова и типовые формулировки для каждой категории.
# Каждый элемент — (список_ключевых_слов, список_шаблонов).
_CATEGORY_TEMPLATES: dict[str, tuple[list[str], list[str]]] = {
    "Веб-разработка": (
        ["сайт", "лендинг", "веб", "frontend", "backend", "API", "адаптивность", "верстка", "react", "python"],
        [
            "Нужно разработать {kw} для нашего бизнеса",
            "Требуется {kw} с адаптивным дизайном и админ-панелью",
            "Просьба сделать {kw}, сайт должен открываться быстро",
            "Разработать корпоративный {kw} с формой обратной связи",
            "{kw} с интеграцией API платежной системы",
        ],
    ),
    "Мобильное приложение": (
        ["приложение", "мобильное", "ios", "android", "flutter", "react native", "push-уведомления", "мобильная версия"],
        [
            "Разработать мобильное {kw} под iOS и Android",
            "Нужно приложение с поддержкой {kw}",
            "Создать мобильное приложение для интернет-магазина",
            "Требуется {kw} с push-уведомлениями и офлайн-режимом",
            "Сделать мобильную версию сервиса на {kw}",
        ],
    ),
    "Дизайн": (
        ["дизайн", "логотип", "макет", "интерфейс", "UI", "UX", "фирменный стиль", "прототип", "баннер", "цветовая схема"],
        [
            "Нужен {kw} для нового продукта",
            "Разработать {kw} в фирменном стиле компании",
            "Создать {kw} интерфейса и прототип экранов",
            "Требуется {kw} с понятной цветовой схемой",
            "Сделать баннер и {kw} для рекламной кампании",
        ],
    ),
    "Аналитика": (
        ["аналитика", "отчет", "дашборд", "метрики", "данные", "анализ", "kpi", "воронка", "статистика", "прогноз"],
        [
            "Нужна {kw} по продажам за последний квартал",
            "Построить {kw} с ключевыми метриками",
            "Провести анализ данных и подготовить отчет",
            "Собрать {kw} по воронке продаж и KPI",
            "Сделать статистику и прогноз по {kw}",
        ],
    ),
    "Маркетинг": (
        ["маркетинг", "реклама", "продвижение", "seo", "smm", "контент", "рассылка", "тап-маркетинг", "канал"],
        [
            "Нужна рекламная кампания по {kw}",
            "Разработать стратегию {kw} для роста продаж",
            "Запустить {kw} в социальных сетях",
            "Настроить контекстную рекламу и {kw}",
            "Подготовить план {kw} и контент-рассылку",
        ],
    ),
}


def _build_texts(category: str, keywords: list[str], templates: list[str], n: int, rng: random.Random) -> list[str]:
    """Собирает ``n`` текстов для заданной категории на основе шаблонов.

    Каждый текст заполняется одним из ключевых слов категории.
    """
    texts: list[str] = []
    for _ in range(n):
        template = rng.choice(templates)
        keyword = rng.choice(keywords)
        texts.append(template.format(kw=keyword))
    return texts


def generate_dataset(
    examples_per_category: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Генерирует синтетический датасет заявок.

    Args:
        examples_per_category: количество примеров на категорию
            (по умолчанию ``config.EXAMPLES_PER_CATEGORY``).
        seed: сид генератора случайных чисел.

    Returns:
        DataFrame с колонками ``text`` и ``label``.
    """
    examples_per_category = examples_per_category or config.EXAMPLES_PER_CATEGORY
    seed = seed if seed is not None else config.RANDOM_SEED
    rng = random.Random(seed)

    texts: list[str] = []
    labels: list[str] = []

    for category, (keywords, templates) in _CATEGORY_TEMPLATES.items():
        category_texts = _build_texts(category, keywords, templates, examples_per_category, rng)
        texts.extend(category_texts)
        labels.extend([category] * len(category_texts))

    # Перемешиваем строки, чтобы классы не шли подряд.
    rows = list(zip(texts, labels))
    rng.shuffle(rows)
    texts, labels = zip(*rows) if rows else ([], [])

    return pd.DataFrame({"text": list(texts), "label": list(labels)})


def save_dataset(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Сохраняет датасет в CSV-файл.

    Args:
        df: датасет для сохранения.
        path: путь сохранения (по умолчанию ``config.DATASET_PATH``).

    Returns:
        Путь к сохранённому файлу.
    """
    path = path or config.DATASET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Загружает датасет из CSV-файла.

    Args:
        path: путь к файлу (по умолчанию ``config.DATASET_PATH``).

    Returns:
        DataFrame с колонками ``text`` и ``label``.
    """
    path = path or config.DATASET_PATH
    return pd.read_csv(path, encoding="utf-8")


def get_or_generate_dataset(path: Path | None = None) -> pd.DataFrame:
    """Возвращает существующий датасет либо генерирует и сохраняет новый.

    Args:
        path: путь к файлу датасета.

    Returns:
        DataFrame с данными.
    """
    path = path or config.DATASET_PATH
    if path.exists():
        return load_dataset(path)
    df = generate_dataset()
    save_dataset(df, path)
    return df