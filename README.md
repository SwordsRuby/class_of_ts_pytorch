# Локальный классификатор текстовых заявок

Легковесный классификатор на Python для автоматической маршрутизации входящих
текстовых заявок (ТЗ) по **5 категориям**:

- **Веб-разработка**
- **Мобильное приложение**
- **Дизайн**
- **Аналитика**
- **Маркетинг**

Ключевые требования: использование PyTorch / Scikit-learn, генерация
собственного датасета (10 примеров на категорию), выполнение в локальной среде.

## Архитектура решения

Проект построен по модульному принципу:

1. **Генерация данных** — синтетический датасет из 50 текстов (10 на категорию).
2. **Предобработка текста** — нижний регистр, удаление стоп-слов, токенизация.
3. **Векторизация** — TF-IDF.
4. **Обучение модели** — два подхода для сравнения:
   - Классический ML: `LogisticRegression` / `SVM` (Scikit-learn).
   - Нейросеть: многослойный перцептрон (MLP) на PyTorch с 2 скрытыми слоями.
   - Разделение данных: тренировочная (80%) и тестовая (20%).
5. **Оценка качества** — Accuracy, Precision, Recall, F1-score, матрица ошибок.
6. **Инференс** — функция `predict(text)`, возвращающая категорию и вероятность.

## Структура проекта

```
class_of_ts_pytorch/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/requests_dataset.csv     # сгенерированный датасет
│   └── processed/                   # сохранённые модели и векторизатор
├── src/
│   ├── config.py                    # категории, пути, гиперпараметры
│   ├── data_generation.py           # генерация датасета
│   ├── preprocessing.py             # предобработка текста
│   ├── features.py                  # TF-IDF векторизация
│   ├── models/
│   │   ├── sklearn_model.py         # LogisticRegression / SVM
│   │   └── pytorch_mlp.py           # MLP на PyTorch
│   ├── evaluation.py                # метрики и матрица ошибок
│   ├── inference.py                 # predict(text)
│   └── pipeline.py                  # оркестрация пайплайна
├── scripts/
│   ├── generate_data.py             # генерация данных
│   ├── train.py                     # обучение моделей
│   └── predict.py                   # инференс из CLI
├── results/                         # отчёты и матрицы ошибок
└── tests/                           # unit-тесты
```

## Установка и настройка

```bash
# Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Для CPU-версии PyTorch (если не установился из requirements)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Использование

### 1. Генерация датасета

```bash
python -m scripts.generate_data
```

Создаёт `data/raw/requests_dataset.csv` с 50 примерами (10 на категорию).

### 2. Обучение моделей

```bash
# Обучить обе модели
python -m scripts.train

# Только Scikit-learn или только PyTorch
python -m scripts.train --sklearn
python -m scripts.train --pytorch

# Перегенерировать датасет перед обучением
python -m scripts.train --regenerate
```

Результаты сохраняются в `results/` (отчёты и матрицы ошибок), а модели —
в `data/processed/`.

### 3. Предсказание (инференс)

```bash
python -m scripts.predict --text "Нужно сделать сайт для компании"
python -m scripts.predict --text "Разработать мобильное приложение" --model pytorch
echo "Создать логотип" | python -m scripts.predict
```

Программа выводит предсказанную категорию, вероятность и распределение
вероятностей по всем категориям.

### 4. Программный вызов

```python
from src.pipeline import run_pipeline

# Полный пайплайн: генерация, обучение, оценка
results = run_pipeline()

# Инференс через стандартные артефакты
from src.inference import predict
result = predict("Нужно сделать сайт для компании")
print(result.category, result.probability)
```

## Тесты

```bash
python -m pytest tests/ -v
```

## Примечания

- Для воспроизводимости используется фиксированный `RANDOM_SEED = 42`.
- Удаление стоп-слов настраивается флагом `REMOVE_STOPWORDS` в `src/config.py`.
- Тип классической модели (`logistic` / `svm`) задаётся параметром
  `SKLEARN_MODEL_TYPE` в `src/config.py`.