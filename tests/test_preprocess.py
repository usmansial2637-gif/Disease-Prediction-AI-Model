"""
Tests for src/preprocess.py — cleaning, outlier handling, feature
selection, and the train/test split + scaling step.
"""
import numpy as np

from src.data_loader import DATASETS
from src.preprocess import (
    clean,
    detect_and_handle_outliers,
    select_features,
    split_and_scale,
)


def _load(dataset_key="breast_cancer"):
    df, _ = DATASETS[dataset_key]()
    return df


def test_clean_removes_exact_duplicates():
    import pandas as pd

    df = _load()
    df_with_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned = clean(df_with_dupe)
    assert len(cleaned) == len(df)


def test_detect_and_handle_outliers_caps_values():
    df = clean(_load())
    capped_df, report_df = detect_and_handle_outliers(df, method="iqr", action="cap")
    assert len(capped_df) == len(df)  # capping never drops rows
    assert "n_outliers" in report_df.columns
    assert (report_df["n_outliers"] >= 0).all()


def test_select_features_keeps_subset_and_scores_all():
    df = clean(_load())
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)
    idx, names, scores_df = select_features(X_train, y_train, feature_names)

    assert 0 < len(names) <= len(feature_names)
    assert len(idx) == len(names)
    assert len(scores_df) == len(feature_names)  # every feature gets scored


def test_split_and_scale_shapes_and_scaling():
    df = clean(_load())
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    n_features = len(feature_names)
    assert X_train.shape[1] == n_features
    assert X_test.shape[1] == n_features
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)

    # Scaled training data should be roughly standardized (mean ~0, std ~1)
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X_train.std(axis=0), 1, atol=1e-6)
