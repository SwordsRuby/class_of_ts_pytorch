#!/usr/bin/env python3
"""Скрипт обучения классификаторов текстовых заявок.

Обучает Scikit-learn и/или PyTorch модели, оценивает качество
и сохраняет артефакты (модели, отчёты, матрицы ошибок).

Запуск:
    python -m scripts.train                 # обе модели
    python -m scripts.train --sklearn        # только Scikit-learn
    python -m scripts.train --pytorch        # только PyTorch
    python -m scripts.train --regenerate     # перегенерировать датасет
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение классификаторов текстовых заявок")
    parser.add_argument("--sklearn", action="store_true", help="Обучить только Scikit-learn модель")
    parser.add_argument("--pytorch", action="store_true", help="Обучить только PyTorch модель")
    parser.add_argument("--regenerate", action="store_true", help="Перегенерировать датасет перед обучением")
    args = parser.parse_args()

    train_sklearn = args.sklearn or not args.pytorch
    train_pytorch = args.pytorch or not args.sklearn

    results = run_pipeline(
        train_sklearn=train_sklearn,
        train_pytorch=train_pytorch,
        regenerate=args.regenerate,
    )

    print("\n" + results.summary())


if __name__ == "__main__":
    main()