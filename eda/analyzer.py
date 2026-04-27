import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

from config.settings import LABEL_COL, TEXT_COL


class EDAAnalyzer:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def show_random_messages(self, n=5):
        sample = self.df[[LABEL_COL, TEXT_COL]].sample(n)

        for _, row in sample.iterrows():
            print(f"\n[{row[LABEL_COL]}] {row[TEXT_COL]}")

    def total_messages(self):
        print(f"Total: {len(self.df)}")

    def plot_distribution(self):
        counts = self.df[LABEL_COL].value_counts()
        proportions = self.df[LABEL_COL].value_counts(normalize=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        counts.plot(kind='bar', ax=axes[0], title='Cantidad por clase')
        proportions.plot(kind='pie', autopct='%1.1f%%', ax=axes[1])

        axes[1].set_ylabel("")
        plt.show()

    def plot_length_density(self, label):
        subset = self.df[self.df[LABEL_COL] == label]
        lengths = subset[TEXT_COL].astype(str).str.len()

        lengths.plot(kind='density', title=f'Densidad ({label})')
        plt.show()

    def plot_top_words(self, label, top_n=20):
        subset = self.df[self.df[LABEL_COL] == label]

        words = (
            subset[TEXT_COL]
            .dropna()
            .astype(str)
            .str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.split()
            .explode()
        )

        top = words.value_counts().head(top_n)

        top.sort_values().plot(kind='barh', title=f'Top {top_n} ({label})')
        plt.show()

    def plot_wordcloud(self, label):
        subset = self.df[self.df[LABEL_COL] == label]

        text = " ".join(subset[TEXT_COL].dropna().astype(str))

        wc = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.imshow(wc)
        plt.axis('off')
        plt.title(f'WordCloud ({label})')
        plt.show()