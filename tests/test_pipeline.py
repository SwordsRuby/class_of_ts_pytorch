"""Тесты для ключевых модулей пайплайна классификации.

Покрывают: генерацию данных, предобработку, векторизацию, обучение
моделей, оценку и инференс.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.data_generation import generate_dataset
from src.evaluation import evaluate_predictions
from src.features import build_tfidf_vectorizer, fit_vectorizer
from src.inference import TextClassifier
from src.models.pytorch_mlp import PyTorchClassifier
from src.models.sklearn_model import SklearnClassifier
from src.preprocessing import preprocess_corpus, preprocess_text


class TestDataGeneration:
    def test_dataset_shape(self):
        df = generate_dataset(examples_per_category=10, seed=42)
        assert len(df) == 50
        assert set(df.columns) == {"text", "label"}

    def test_balanced_categories(self):
        df = generate_dataset(examples_per_category=10, seed=1)
        assert df["label"].value_counts().to_dict() == {cat: 10 for cat in config.CATEGORIES}

    def test_nonempty_texts(self):
        df = generate_dataset(examples_per_category=5, seed=7)
        assert all(len(t) > 0 for t in df["text"])


class TestPreprocessing:
    def test_lowercase_and_tokenization(self):
        result = preprocess_text("Нужно Разработать Сайт!", remove_stopwords_flag=False)
        # Текст приводится к нижнему регистру; слово "сайт" сохраняется в основе.
        assert result == result.lower()
        assert "сайт" in result

    def test_stemming_normalizes_word_forms(self):
        # Разные падежи/числа слова "сайт" сводятся к одной основе.
        assert preprocess_text("создание сайта", False).split()[-1] == \
            preprocess_text("разработка сайтов", False).split()[-1]

    def test_stopword_removal(self):
        result = preprocess_text("Нужно разработать сайт для бизнеса")
        assert "для" not in result
        assert "сайт" in result

    def test_preprocess_corpus(self):
        texts = ["Привет Мир", "Веб разработка"]
        out = preprocess_corpus(texts, remove_stopwords_flag=False)
        assert len(out) == 2
        assert all(t == t.lower() for t in out)


class TestFeatures:
    def test_tfidf_shape(self):
        texts = ["разработать сайт", "мобильное приложение", "дизайн логотип"]
        vectorizer: TfidfVectorizer = build_tfidf_vectorizer(max_features=100)
        X = fit_vectorizer(vectorizer, texts)
        assert X.shape[0] == 3
        assert X.shape[1] <= 100


class TestModels:
    def test_sklearn_train_and_predict(self):
        texts = ["разработать сайт", "сделать приложение", "нарисовать логотип", "анализ данных", "реклама товара"]
        labels = ["Веб-разработка", "Мобильное приложение", "Дизайн", "Аналитика", "Маркетинг"]
        vectorizer = build_tfidf_vectorizer(max_features=100)
        X = fit_vectorizer(vectorizer, texts)

        model = SklearnClassifier(categories=config.CATEGORIES)
        model.fit(X, labels)
        preds = model.predict(X)
        assert preds == labels

    def test_pytorch_train_and_predict(self):
        texts = ["разработать сайт", "сделать приложение", "нарисовать логотип", "анализ данных", "реклама товара"]
        labels = ["Веб-разработка", "Мобильное приложение", "Дизайн", "Аналитика", "Маркетинг"]
        vectorizer = build_tfidf_vectorizer(max_features=100)
        X = fit_vectorizer(vectorizer, texts)

        model = PyTorchClassifier(input_dim=X.shape[1], categories=config.CATEGORIES)
        model.fit(X, labels, epochs=10, verbose=False)
        preds = model.predict(X)
        assert len(preds) == 5
        probs = model.predict_proba(X)
        assert probs.shape == (5, len(config.CATEGORIES))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


class TestEvaluation:
    def test_metrics_computed(self):
        y_true = ["A", "A", "B", "B"]
        y_pred = ["A", "A", "B", "A"]
        metrics = evaluate_predictions(y_true, y_pred, categories=["A", "B"])
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["confusion_matrix"].shape == (2, 2)
        assert "report_text" in metrics


class TestInference:
    def test_text_classifier_predict(self):
        texts = [
            "разработать сайт для бизнеса",
            "сделать мобильное приложение",
            "создать логотип",
            "собрать отчет по данным",
            "запустить рекламу товара",
        ]
        labels = ["Веб-разработка", "Мобильное приложение", "Дизайн", "Аналитика", "Маркетинг"]
        vectorizer = build_tfidf_vectorizer(max_features=100)
        X = fit_vectorizer(vectorizer, texts)

        model = SklearnClassifier(categories=config.CATEGORIES)
        model.fit(X, labels)

        classifier = TextClassifier(vectorizer=vectorizer, model=model)
        result = classifier.predict("нужно разработать сайт для компании")
        assert result.category in config.CATEGORIES
        assert 0.0 <= result.probability <= 1.0
        assert set(result.probabilities.keys()) == set(config.CATEGORIES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])