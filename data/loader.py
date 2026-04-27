import pandas as pd
from config.settings import LABEL_COL, TEXT_COL


class DataLoader:

    @staticmethod
    def load(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(
            file_path,
            sep=';',
            encoding='latin-1',
            engine='python'
        )

        DataLoader._validate(df)
        df = DataLoader._clean_labels(df)
        return df

    @staticmethod
    def _validate(df: pd.DataFrame):
        required = [LABEL_COL, TEXT_COL]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Faltan columnas: {missing}")

    @staticmethod
    def _clean_labels(df: pd.DataFrame) -> pd.DataFrame:
        df[LABEL_COL] = df[LABEL_COL].astype(str).str.replace('"', '').str.strip()
        return df