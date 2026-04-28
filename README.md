# Lab 6 — SMS Spam/Ham EDA & Text Preprocessing

CLI tool for exploratory data analysis (EDA) and text preprocessing over an
English SMS spam/ham dataset. Built for the *Inteligencia Artificial* course
(7º semestre).

## Features

- Load and validate the `spam_ham.csv` dataset.
- Inspect random messages, totals and class distribution.
- Plot message-length density per class (spam / ham).
- Plot top-N words and word clouds per class.
- Run an NLTK-based preprocessing pipeline (tokenization → lowercase →
  punctuation removal → stopword removal → lemmatization).
- Toggle EDA between the original and the preprocessed dataset to compare.

## Project structure

```
lab6/
├── main.py                     # Entry point
├── spam_ham.csv                # Dataset (Label;SMS_TEXT, latin-1)
├── requirements.txt
├── app/
│   ├── app.py                  # MainApp orchestrator
│   └── menu.py                 # CLI menu
├── config/
│   └── settings.py             # Paths, columns, labels, plot config
├── data/
│   └── loader.py               # CSV reader + label cleaning
├── eda/
│   └── analyzer.py             # EDA / plots / wordclouds
└── preprocessing/
    └── text_processor.py       # NLTK preprocessing pipeline
```

## Requirements

- Python 3.10+
- Dependencies listed in [requirements.txt](requirements.txt) (pandas,
  matplotlib, nltk, wordcloud, scipy, …).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

NLTK resources (`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`) are
downloaded automatically the first time the preprocessing step runs.

## Usage

```bash
python main.py
```

You will see a menu with the following options:

| Opción | Acción                                        |
|--------|-----------------------------------------------|
| 1      | 5 mensajes aleatorios                         |
| 2      | Total de mensajes                             |
| 3      | Distribución por clase                        |
| 4      | Densidad de longitud — spam                   |
| 5      | Densidad de longitud — ham                    |
| 6      | Top palabras — spam                           |
| 7      | Top palabras — ham                            |
| 8      | WordCloud — spam                              |
| 9      | WordCloud — ham                               |
| 10     | Ejecutar preprocesamiento                     |
| 11     | Alternar dataset (original / preprocesado)    |
| 0      | Salir                                         |

The active dataset (`original` or `preprocesado`) is shown above each prompt;
options 1–9 always run against whichever dataset is active.

## Preprocessing pipeline

Implemented in [preprocessing/text_processor.py](preprocessing/text_processor.py):

1. **Tokenization** with `nltk.word_tokenize`.
2. **Lowercasing** to collapse `Free` / `free`.
3. **Punctuation removal** via `string.punctuation`.
4. **Stopword removal** (English) — high-frequency, non-discriminative words.
5. **Lemmatization** with `WordNetLemmatizer` — chosen over stemming to keep
   real dictionary forms (`running → run`) and produce readable wordclouds /
   top-words.

## Configuration

Tweak `config/settings.py` to change the CSV path/separator/encoding, column
names, sample size, top-N count or wordcloud dimensions.
