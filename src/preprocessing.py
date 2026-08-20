"""Предобработка текстовых данных.

Реализует этапы: приведение к нижнему регистру, удаление стоп-слов
(опционально), токенизацию и нормализацию текста.
"""
from __future__ import annotations

import re
from typing import Iterable

from nltk.stem import SnowballStemmer

# Набор русских и английских стоп-слов (часто встречающиеся слова,
# не несущие смысловой нагрузки).
_RUSSIAN_STOPWORDS: set[str] = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него",
    "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом",
    "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо",
    "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без",
    "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда",
    "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним",
    "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас",
    "были", "куда", "зачем", "всех", "никогда", "можно", "при", "наконец",
    "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через",
    "эти", "нас", "про", "всего", "них", "какая", "много", "разве", "три",
    "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда",
    "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда",
    "конечно", "всю", "между", "а",
}

_ENGLISH_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "for", "on",
    "with", "is", "are", "was", "were", "be", "been", "this", "that", "it",
    "as", "at", "by", "from", "up", "down", "into", "out", "over", "then",
}

# Символы, которые считаются разделителями токенов.
_TOKEN_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)

# Кэш объединённого множества стоп-слов.
_STOPWORDS_CACHE: set[str] | None = None

# Стеммер для нормализации словоформ русского языка.
# Приводит разные падежи/числа к основе ("сайта" -> "сайт", "сайте" -> "сайт"),
# что критично для корректной TF-IDF векторизации русских текстов.
_STEMMER = SnowballStemmer("russian")


def _stem_tokens(tokens: list[str]) -> list[str]:
    """Стеммит (нормализует) каждый токен.

    Args:
        tokens: список токенов.

    Returns:
        Список основ слов.
    """
    return [_STEMMER.stem(token) for token in tokens]


def get_stopwords(remove_stopwords: bool = True) -> set[str]:
    """Возвращает множество стоп-слов.

    Args:
        remove_stopwords: если ``False``, возвращает пустое множество.

    Returns:
        Множество стоп-слов (русских и английских).
    """
    global _STOPWORDS_CACHE
    if not remove_stopwords:
        return set()
    if _STOPWORDS_CACHE is None:
        _STOPWORDS_CACHE = _RUSSIAN_STOPWORDS | _ENGLISH_STOPWORDS
    return _STOPWORDS_CACHE


def tokenize(text: str) -> list[str]:
    """Токенизирует текст на слова.

    Приводит к нижнему регистру, удаляет пунктуацию и лишние пробелы.

    Args:
        text: исходный текст.

    Returns:
        Список токенов (слов).
    """
    text = text.lower()
    text = _TOKEN_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text.split()


def remove_stopwords(tokens: list[str], stopwords: set[str] | None = None) -> list[str]:
    """Удаляет стоп-слова из списка токенов.

    Args:
        tokens: список токенов.
        stopwords: множество стоп-слов. Если не задано, используется
            стандартный набор.

    Returns:
        Список токенов без стоп-слов.
    """
    if stopwords is None:
        stopwords = get_stopwords(remove_stopwords=True)
    return [token for token in tokens if token not in stopwords]


def preprocess_text(text: str, remove_stopwords_flag: bool = True) -> str:
    """Полный цикл предобработки одного текста.

    Args:
        text: исходный текст.
        remove_stopwords_flag: удалять ли стоп-слова.

    Returns:
        Нормализованный текст (основы слов через пробел).

    Примечание:
        Стемминг нормализует словоформы (падежи/числа), что позволяет
        вектору запроса совпадать с признаками модели, даже если слово
        стоит в другой грамматической форме.
    """
    tokens = tokenize(text)
    if remove_stopwords_flag:
        tokens = remove_stopwords(tokens)
    tokens = _stem_tokens(tokens)
    return " ".join(tokens)


def preprocess_corpus(texts: Iterable[str], remove_stopwords_flag: bool = True) -> list[str]:
    """Применяет предобработку к коллекции текстов.

    Args:
        texts: итерируемая коллекция текстов.
        remove_stopwords_flag: удалять ли стоп-слова.

    Returns:
        Список нормализованных текстов.
    """
    return [preprocess_text(text, remove_stopwords_flag) for text in texts]