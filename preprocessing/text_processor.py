import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from config.settings import TEXT_COL


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

    _NLTK_RESOURCES = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._ensure_nltk_resources()
        self._punctuation = set(string.punctuation)
        self._stopwords = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    @classmethod
    def _ensure_nltk_resources(cls):
        for path, pkg in cls._NLTK_RESOURCES:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)

    def process(self) -> pd.DataFrame:
        processed = self.df.copy()
        processed[TEXT_COL] = processed[TEXT_COL].map(self._transform)
        return processed

    def _transform(self, text) -> str:
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text)
        tokens = [t.lower() for t in tokens]
        tokens = [t for t in tokens if t not in self._punctuation]
        tokens = [t for t in tokens if t.isalpha()]
        tokens = [t for t in tokens if t not in self._stopwords]
        tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)
