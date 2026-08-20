#!/usr/bin/env python3
"""Скрипт генерации синтетического датасета текстовых заявок.

Этап 1 (AGENTS.md): генерация данных — 50 текстов (10 на категорию).
Запуск:
    python -m scripts.generate_data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Позволяет импортировать пакет ``src`` из корня проекта.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data_generation import generate_dataset, save_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация синтетического датасета заявок")
    parser.add_argument(
        "--per-category",
        type=int,
        default=config.EXAMPLES_PER_CATEGORY,
        help="Количество примеров на категорию (по умолчанию 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Сид генератора случайных чисел",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DATASET_PATH,
        help="Путь сохранения датасета (CSV)",
    )
    args = parser.parse_args()

    config.ensure_directories()

    df = generate_dataset(examples_per_category=args.per_category, seed=args.seed)
    save_dataset(df, args.output)

    print(f"Датасет сохранён: {args.output}")
    print(f"Всего примеров: {len(df)}")
    print("Распределение по категориям:")
    for label, count in df["label"].value_counts().sort_index().items():
        print(f"  {label}: {count}")

    print("\nПримеры текстов:")
    for _, row in df.head(5).iterrows():
        print(f"  [{row['label']}] {row['text']}")


if __name__ == "__main__":
    main()