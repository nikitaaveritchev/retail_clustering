import pandas as pd


def load_excel(path: str, sheet_name=0) -> pd.DataFrame:
    """Load Excel data from the provided path."""
    return pd.read_excel(path, sheet_name=sheet_name)
