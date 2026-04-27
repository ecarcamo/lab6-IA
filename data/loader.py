import pandas as pd

from config.settings import (
    CSV_ENCODING,
    CSV_SEPARATOR,
    LABEL_COL,
    TEXT_COL,
)


class DataLoader:

    REQUIRED_COLUMNS = [LABEL_COL, TEXT_COL]

    @staticmethod
    def load(file_path: str) -> pd.DataFrame:
        raw_dataset = DataLoader._read_csv(file_path)
        DataLoader._validate_columns(raw_dataset)
        clean_dataset = DataLoader._clean_labels(raw_dataset)
        return clean_dataset

    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        return pd.read_csv(
            file_path,
            sep=CSV_SEPARATOR,
            encoding=CSV_ENCODING,
            engine="python",
        )

    @staticmethod
    def _validate_columns(dataset: pd.DataFrame) -> None:
        missing_columns = [
            column
            for column in DataLoader.REQUIRED_COLUMNS
            if column not in dataset.columns
        ]
        if missing_columns:
            raise ValueError(f"Faltan columnas: {missing_columns}")

    @staticmethod
    def _clean_labels(dataset: pd.DataFrame) -> pd.DataFrame:
        cleaned = dataset.copy()
        cleaned[LABEL_COL] = (
            cleaned[LABEL_COL]
            .astype(str)
            .str.replace('"', "")
            .str.strip()
        )
        return cleaned
