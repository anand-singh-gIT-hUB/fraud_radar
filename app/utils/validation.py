import pandas as pd
from typing import Tuple, List


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    missing = [col for col in required_columns if col not in df.columns]
    return (len(missing) == 0), missing


def clean_numeric(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """Convert required columns to float32, coerce bad values to NaN."""
    df = df.copy()
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df.dropna(subset=required_columns, inplace=True)
    return df
