#!/usr/bin/env python3
"""Скрипт инференса (предсказания) категории текстовой заявки.

Запуск:
    python -m scripts.predict --text "Нужно сделать сайт для компании"
    python -m scripts.predict --text "..." --model pytorch
    echo "..." | python -m scripts.predict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import load_classifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Предсказание категории текстовой заявки")
    parser.add_argument("--text", type=str, help="Текст заявки (либо через stdin)")
    parser.add_argument(
        "--model",
        type=str,
        choices=["sklearn", "pytorch"],
        default="sklearn",
        help="Тип модели (по умолчанию sklearn)",
    )
    args = parser.parse_args()

    text = args.text
    if text is None:
        text = sys.stdin.read().strip()
    if not text:
        parser.error("Не указан текст заявки (--text или stdin)")

    classifier = load_classifier(model_type=args.model)
    result = classifier.predict(text)

    print(f"Текст:        {result.text}")
    print(f"Категория:    {result.category}")
    print(f"Вероятность:  {result.probability:.4f}")
    print("\nВероятности по всем категориям:")
    for category, prob in sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {category}: {prob:.4f}")


if __name__ == "__main__":
    main()