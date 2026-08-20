"""Нейросетевая модель — многослойный перцептрон (MLP) на PyTorch.

Обучается на TF-IDF признаках и классифицирует тексты по категориям.
Архитектура: входной слой (размерность признаков) -> 1-2 скрытых
полносвязных слоя с ReLU и Dropout -> выходной слой с softmax.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import config


class MLPClassifier(nn.Module):
    """Многослойный перцептрон для классификации текстов.

    Args:
        input_dim: размерность входных признаков (TF-IDF векторов).
        hidden_sizes: размеры скрытых слоёв.
        num_classes: число классов (категорий).
        dropout: вероятность дропаута.
    """

    def __init__(self, input_dim: int, hidden_sizes: list[int], num_classes: int, dropout: float = 0.3):
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход сети.

        Args:
            x: тензор признаков формы (batch_size, input_dim).

        Returns:
            Логиты (не нормализованные оценки) формы (batch_size, num_classes).
        """
        return self.net(x)


class PyTorchClassifier:
    """Обёртка для обучения и инференса MLP-модели на PyTorch.

    Args:
        input_dim: размерность входных признаков.
        categories: упорядоченный список категорий.
        hidden_sizes: размеры скрытых слоёв.
        seed: сид для воспроизводимости.
    """

    def __init__(
        self,
        input_dim: int,
        categories: list[str] | None = None,
        hidden_sizes: list[int] | None = None,
        seed: int | None = None,
    ):
        self.input_dim = input_dim
        self.categories: list[str] = categories or list(config.CATEGORIES)
        self.hidden_sizes: list[int] = hidden_sizes or list(config.PYTORCH_HIDDEN_SIZES)
        self.seed = seed if seed is not None else config.RANDOM_SEED
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._set_seed(self.seed)
        self.model = MLPClassifier(
            input_dim=input_dim,
            hidden_sizes=self.hidden_sizes,
            num_classes=len(self.categories),
        ).to(self.device)

        # Публичный словарь для сохранения/загрузки дополнительных метаданных.
        self.metadata: dict[str, Any] = {"input_dim": input_dim, "categories": self.categories}

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Фиксирует сиды для воспроизводимости обучения."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(
        self,
        X_train,
        y_train,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        verbose: bool = True,
    ) -> "PyTorchClassifier":
        """Обучает MLP на матрице признаков.

        Args:
            X_train: обучающие признаки (sparse/dense матрица).
            y_train: обучающие метки (строки категорий).
            epochs: число эпох.
            batch_size: размер батча.
            learning_rate: скорость обучения.
            verbose: выводить ли потери по эпохам.

        Returns:
            Сам объект модели.
        """
        epochs = epochs or config.PYTORCH_EPOCHS
        batch_size = batch_size or config.PYTORCH_BATCH_SIZE
        learning_rate = learning_rate or config.PYTORCH_LEARNING_RATE

        # Преобразование в плотные тензоры.
        X = np.asarray(X_train.toarray()) if hasattr(X_train, "toarray") else np.asarray(X_train)
        X = X.astype(np.float32)

        label_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        y = np.array([label_to_idx[label] for label in y_train], dtype=np.int64)

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=config.PYTORCH_WEIGHT_DECAY,
        )

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                print(f"  Эпоха {epoch}/{epochs}, loss={total_loss / len(dataset):.4f}")

        return self

    def _to_tensor(self, X) -> torch.Tensor:
        """Преобразует признаки в тензор на нужном устройстве."""
        X = np.asarray(X.toarray()) if hasattr(X, "toarray") else np.asarray(X)
        X = X.astype(np.float32)
        return torch.from_numpy(X).to(self.device)

    def get_probability_labels(self) -> list[str]:
        """Возвращает порядок классов, соответствующий колонкам predict_proba.

        Returns:
            Список категорий в порядке колонок вероятностей.
        """
        return self.categories

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        """Возвращает вероятности принадлежности каждому классу.

        Args:
            X: матрица признаков.

        Returns:
            Массив вероятностей формы (n_samples, n_classes).
            Колонки соответствуют :meth:`get_probability_labels`.
        """
        self.model.eval()
        tensor = self._to_tensor(X)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X) -> list[str]:
        """Предсказывает категории для матрицы признаков.

        Args:
            X: матрица признаков.

        Returns:
            Список предсказанных категорий.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return [self.categories[int(i)] for i in indices]

    def save(self, path=None) -> str:
        """Сохраняет модель и метаданные в файл.

        Args:
            path: путь сохранения (по умолчанию ``config.PYTORCH_MODEL_PATH``).

        Returns:
            Путь к сохранённому файлу.
        """
        path = path or config.PYTORCH_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata["model_state_dict"] = self.model.state_dict()
        self.metadata["hidden_sizes"] = self.hidden_sizes
        torch.save(self.metadata, path)
        return str(path)

    @classmethod
    def load(cls, path=None) -> "PyTorchClassifier":
        """Загружает модель из файла.

        Args:
            path: путь к файлу модели.

        Returns:
            Загруженный экземпляр ``PyTorchClassifier``.
        """
        path = path or config.PYTORCH_MODEL_PATH
        metadata = torch.load(path, map_location="cpu", weights_only=False)

        classifier = cls(
            input_dim=metadata["input_dim"],
            categories=metadata["categories"],
            hidden_sizes=metadata.get("hidden_sizes"),
        )
        classifier.model.load_state_dict(metadata["model_state_dict"])
        classifier.model.to(classifier.device)
        classifier.model.eval()
        return classifier