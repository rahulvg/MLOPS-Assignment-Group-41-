import pandas as pd
import pytest

from train.data_loader import load_data


DATA_PATH = "data/processed/heart_disease_clean.csv"


def test_load_data_returns_X_y():
    """Data loader should return features and target."""
    X, y = load_data(DATA_PATH)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) > 0
    assert len(X) == len(y)


def test_target_column_removed_from_X():
    """Target column should not be part of features."""
    X, y = load_data(DATA_PATH)

    assert "target" not in X.columns


def test_target_binary_values():
    """Target should contain only binary values (0 or 1)."""
    _, y = load_data(DATA_PATH)

    assert set(y.unique()).issubset({0, 1})


def test_missing_target_column_raises_error(tmp_path):
    """Missing target column should raise an error."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    temp_file = tmp_path / "temp.csv"
    df.to_csv(temp_file, index=False)

    with pytest.raises(ValueError):
        load_data(temp_file)
