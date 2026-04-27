import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from config.settings import STOPWORDS_LANGUAGE, TEXT_COL


class TextProcessor:
    """
    Pipeline de preprocesamiento de texto con NLTK.

    Pasos aplicados (en orden) y su justificación:

    1. Tokenización (`word_tokenize`)
       Divide cada mensaje en unidades léxicas (palabras / signos).
       Es el primer paso obligado: todo el resto del pipeline opera
       sobre tokens, no sobre la cadena completa.

    2. Conversión a minúsculas
       Unifica "Free" y "free" en un solo token. Sin esto el modelo
       trataría mayúsculas y minúsculas como vocabulario distinto,
       inflando dimensionalidad y diluyendo frecuencias.

    3. Eliminación de signos de puntuación
       La puntuación rara vez aporta señal para clasificar spam/ham
       y ensucia el conteo de tokens. Se filtran usando
       `string.punctuation`.

    4. Eliminación de stopwords (inglés)
       Palabras como "the", "is", "to" aparecen en ambas clases con
       altísima frecuencia y no discriminan. Quitarlas hace que el
       Top-20 y el WordCloud reflejen vocabulario informativo.

    5. Lemmatización (WordNetLemmatizer)
       Se eligió **lemmatización** sobre stemming porque:
         - Devuelve formas reales del diccionario (running -> run),
           lo que produce WordClouds y top-words legibles.
         - Es más conservadora: stemming (Porter/Snowball) puede
           recortar de forma agresiva ("univers" para "university"),
           degradando la interpretabilidad sin mejorar mucho la
           clasificación en datasets pequeños como SMS Spam.
         - El costo extra de WordNet es despreciable para ~5k
           mensajes cortos.
    """

    NLTK_RESOURCES = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self._ensure_nltk_resources()
        self.punctuation_set = set(string.punctuation)
        self.stopwords_set = set(stopwords.words(STOPWORDS_LANGUAGE))
        self.lemmatizer = WordNetLemmatizer()

    @classmethod
    def _ensure_nltk_resources(cls) -> None:
        for resource_path, package_name in cls.NLTK_RESOURCES:
            cls._download_if_missing(resource_path, package_name)

    @staticmethod
    def _download_if_missing(resource_path: str, package_name: str) -> None:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)

    def process(self) -> pd.DataFrame:
        processed_dataset = self.dataset.copy()
        processed_dataset[TEXT_COL] = processed_dataset[TEXT_COL].map(self._transform)
        return processed_dataset

    def _transform(self, text) -> str:
        if not isinstance(text, str):
            return ""

        tokens = word_tokenize(text)
        normalized_tokens = self._normalize_tokens(tokens)
        meaningful_tokens = self._remove_noise(normalized_tokens)
        lemmatized_tokens = self._lemmatize(meaningful_tokens)
        return " ".join(lemmatized_tokens)

    def _normalize_tokens(self, tokens: list[str]) -> list[str]:
        return [token.lower() for token in tokens]

    def _remove_noise(self, tokens: list[str]) -> list[str]:
        return [
            token
            for token in tokens
            if self._is_meaningful_token(token)
        ]

    def _is_meaningful_token(self, token: str) -> bool:
        is_punctuation = token in self.punctuation_set
        is_non_alphabetic = not token.isalpha()
        is_stopword = token in self.stopwords_set
        return not (is_punctuation or is_non_alphabetic or is_stopword)

    def _lemmatize(self, tokens: list[str]) -> list[str]:
        return [self.lemmatizer.lemmatize(token) for token in tokens]
