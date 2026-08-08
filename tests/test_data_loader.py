"""
Tests for src/data_loader.py

These run with no internet access and no pre-existing data/*.csv files:
breast_cancer comes from sklearn, heart/diabetes fall back to the synthetic
generators, so the suite is fully self-contained and fast (CI-safe).
"""
import pandas as pd

from src.data_loader import DATASETS


def test_all_datasets_registered():
    assert set(DATASETS.keys()) == {"breast_cancer", "heart", "diabetes"}


def test_each_dataset_loads_with_target_column():
    for key, loader in DATASETS.items():
        df, source_label = loader()
        assert isinstance(df, pd.DataFrame)
        assert "target" in df.columns, f"{key} is missing a 'target' column"
        assert isinstance(source_label, str) and len(source_label) > 0


def test_each_dataset_has_rows_and_binary_target():
    for key, loader in DATASETS.items():
        df, _ = loader()
        assert len(df) > 50, f"{key} dataset looks too small: {len(df)} rows"
        unique_targets = set(df["target"].unique().tolist())
        assert unique_targets.issubset({0, 1}), f"{key} target isn't binary: {unique_targets}"


def test_each_dataset_has_no_fully_empty_columns():
    for key, loader in DATASETS.items():
        df, _ = loader()
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        assert empty_cols == [], f"{key} has fully-empty columns: {empty_cols}"
