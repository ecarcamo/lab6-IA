import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from config.settings import (
    LABEL_COL,
    RANDOM_SAMPLE_SIZE,
    TEXT_COL,
    TOP_WORDS_COUNT,
    WORDCLOUD_BACKGROUND,
    WORDCLOUD_HEIGHT,
    WORDCLOUD_WIDTH,
)


class EDAAnalyzer:

    NON_WORD_PATTERN = r"[^\w\s]"

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def show_random_messages(self, sample_size: int = RANDOM_SAMPLE_SIZE) -> None:
        sample = self.dataset[[LABEL_COL, TEXT_COL]].sample(sample_size)
        for _, message_row in sample.iterrows():
            print(f"\n[{message_row[LABEL_COL]}] {message_row[TEXT_COL]}")

    def total_messages(self) -> None:
        print(f"Total: {len(self.dataset)}")

    def plot_distribution(self) -> None:
        class_counts = self.dataset[LABEL_COL].value_counts()
        class_proportions = self.dataset[LABEL_COL].value_counts(normalize=True)

        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        class_counts.plot(kind="bar", ax=axes[0], title="Cantidad por clase")
        class_proportions.plot(kind="pie", autopct="%1.1f%%", ax=axes[1])
        axes[1].set_ylabel("")
        plt.show()

    def plot_length_density(self, label: str) -> None:
        message_lengths = self._compute_message_lengths(label)
        message_lengths.plot(kind="density", title=f"Densidad ({label})")
        plt.show()

    def plot_top_words(self, label: str, top_n: int = TOP_WORDS_COUNT) -> None:
        top_words = self._compute_top_words(label, top_n)
        top_words.sort_values().plot(kind="barh", title=f"Top {top_n} ({label})")
        plt.show()

    def plot_wordcloud(self, label: str) -> None:
        joined_text = self._join_messages(label)
        wordcloud = WordCloud(
            width=WORDCLOUD_WIDTH,
            height=WORDCLOUD_HEIGHT,
            background_color=WORDCLOUD_BACKGROUND,
        ).generate(joined_text)

        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title(f"WordCloud ({label})")
        plt.show()

    def _filter_by_label(self, label: str) -> pd.DataFrame:
        return self.dataset[self.dataset[LABEL_COL] == label]

    def _compute_message_lengths(self, label: str) -> pd.Series:
        messages = self._filter_by_label(label)[TEXT_COL]
        return messages.astype(str).str.len()

    def _compute_top_words(self, label: str, top_n: int) -> pd.Series:
        messages = self._filter_by_label(label)[TEXT_COL]
        words = (
            messages.dropna()
            .astype(str)
            .str.lower()
            .str.replace(self.NON_WORD_PATTERN, "", regex=True)
            .str.split()
            .explode()
        )
        return words.value_counts().head(top_n)

    def _join_messages(self, label: str) -> str:
        messages = self._filter_by_label(label)[TEXT_COL].dropna().astype(str)
        return " ".join(messages)
