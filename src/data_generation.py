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
# Шаблоны составлены так, чтобы сочетаться с любым ключевым словом,
# и охватывают разные глаголы ("создать", "сделать", "разработать" и т.д.),
# чтобы модель обобщала по смыслу, а не запоминала конкретные фразы.
_CATEGORY_TEMPLATES: dict[str, tuple[list[str], list[str]]] = {
    "Веб-разработка": (
        ["сайт", "лендинг", "веб-сайт", "интернет-магазин", "frontend", "backend", "api", "react", "python", "верстка"],
        [
            "Нужно разработать {kw} для компании",
            "Создать {kw} с адаптивным дизайном",
            "Разработать корпоративный {kw} с админ-панелью",
            "Требуется {kw} с формой обратной связи",
            "Сделать {kw}, который быстро загружается",
            "Нужен {kw} для нашего бизнеса",
            "Задача: разработать {kw}",
            "Написать {kw} с интеграцией api",
            "Создание {kw} для стартапа",
            "Разработать {kw} и подключить аналитику",
        ],
    ),
    "Мобильное приложение": (
        ["приложение", "мобильное приложение", "ios", "android", "flutter", "react native"],
        [
            "Разработать мобильное приложение",
            "Создать {kw} под ios и android",
            "Нужно {kw} с push-уведомлениями",
            "Разработать {kw} для интернет-магазина",
            "Сделать мобильную версию сервиса на {kw}",
            "Требуется {kw} с офлайн-режимом",
            "Задача: разработать {kw}",
            "Написать {kw} для ios и android",
            "Создание {kw} с уведомлениями",
            "Разработать {kw} с push-уведомлениями",
        ],
    ),
    "Дизайн": (
        ["логотип", "макет", "интерфейс", "ui", "ux", "фирменный стиль", "баннер", "прототип", "дизайн", "цветовая схема"],
        [
            "Разработать {kw}",
            "Нужен {kw} для нового продукта",
            "Создать {kw} в фирменном стиле компании",
            "Нарисовать {kw}",
            "Сделать {kw} интерфейса и прототип экранов",
            "Разработать ui и {kw}",
            "Нужен {kw} с понятной цветовой схемой",
            "Задача: создать {kw}",
            "Разработать {kw} для рекламной кампании",
            "Создание {kw} для бренда",
        ],
    ),
    "Аналитика": (
        ["аналитика", "отчет", "дашборд", "метрики", "данные", "анализ", "kpi", "воронка", "статистика", "прогноз"],
        [
            "Построить {kw} с ключевыми метриками",
            "Нужен {kw} по продажам за квартал",
            "Провести {kw} данных и подготовить отчет",
            "Собрать {kw} по воронке продаж",
            "Подготовить {kw} по ключевым показателям",
            "Нужна аналитика по {kw}",
            "Сделать статистику и {kw}",
            "Задача: построить {kw}",
            "Разработать {kw} для руководства",
            "Анализ {kw} за последний месяц",
        ],
    ),
    "Маркетинг": (
        ["реклама", "продвижение", "seo", "smm", "контент", "рассылка", "маркетинг", "кампания", "канал", "аудитория"],
        [
            "Запустить {kw} в социальных сетях",
            "Разработать стратегию {kw}",
            "Настроить {kw} для роста продаж",
            "Подготовить план {kw}",
            "Запустить рекламную кампанию",
            "Продвинуть товар через {kw}",
            "Задача: запустить {kw}",
            "Разработать {kw} кампанию",
            "Создать {kw} стратегию",
            "Настроить таргетированную {kw}",
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