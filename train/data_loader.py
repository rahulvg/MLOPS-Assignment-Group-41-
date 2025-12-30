# src/data_loader.py

import pandas as pd


def load_data(path: str):
    """
    Load processed heart disease dataset.

    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target
    """
    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError("Target column missing from dataset")

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y
