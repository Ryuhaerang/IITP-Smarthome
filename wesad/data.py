"""Dataset loading, cleaning, and splitting utilities for WESAD features."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from wesad.config import ExperimentConfig


@dataclass
class DataPreparationResult:
    """Container holding transformed arrays and metadata for training/eval."""
    train_features: np.ndarray
    train_labels: np.ndarray
    val_features: np.ndarray
    val_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    feature_columns: List[str]
    dropped_all_nan: List[str]
    label_to_id: Dict[str, int]
    subject_splits: Dict[str, List[str]]
    imputer: SimpleImputer
    scaler: StandardScaler


def load_dataset_as_dataframe(dataset_path: str) -> pd.DataFrame:
    """Load the saved HF dataset and expand the nested feature dict into columns."""
    dataset = load_from_disk(dataset_path)
    df = dataset.to_pandas()
    feature_frame = pd.json_normalize(df["features"])
    combined = pd.concat([df[["user_id", "label"]], feature_frame], axis=1)
    return combined


def drop_nan_columns(
    df: pd.DataFrame, feature_columns: Sequence[str]
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Remove columns that are NaN in every row; return remaining feature names."""
    all_nan_cols = [col for col in feature_columns if df[col].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
    remaining = [col for col in feature_columns if col not in all_nan_cols]
    return df, all_nan_cols, remaining


def split_by_subject(
    user_ids: Iterable[str], train_frac: float, val_frac: float, seed: int
) -> Tuple[List[str], List[str], List[str]]:
    """Create subject-level splits to avoid leakage between train/val/test."""
    unique_users = sorted(set(user_ids))
    if len(unique_users) < 3:
        raise ValueError("Need at least 3 unique subjects to form train/val/test splits.")

    rng = np.random.default_rng(seed)
    shuffled = unique_users[:]
    rng.shuffle(shuffled)
    total = len(shuffled)

    n_train = max(1, int(round(train_frac * total)))
    n_val = max(1, int(round(val_frac * total)))
    while n_train + n_val >= total:
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
    train_users = shuffled[:n_train]
    val_users = shuffled[n_train : n_train + n_val]
    test_users = shuffled[n_train + n_val :]
    if not test_users:
        raise AssertionError("Test split ended up empty. Check fraction settings.")
    return train_users, val_users, test_users


def filter_by_users(df: pd.DataFrame, users: Sequence[str]) -> pd.DataFrame:
    """Slice the dataframe to only include data from the given user IDs."""
    return df[df["user_id"].isin(users)].reset_index(drop=True)


def extract_features_and_labels(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_to_id: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract numpy arrays of features/labels, validating label coverage."""
    features = df.loc[:, feature_columns].to_numpy(dtype=np.float64)
    labels = df["label"].map(label_to_id)
    if labels.isna().any():
        unknown = sorted(set(df["label"]) - set(label_to_id))
        raise ValueError(f"Encountered unknown labels: {unknown}")
    return features, labels.to_numpy(dtype=np.int64)


def prepare_wesad_data(config: ExperimentConfig) -> DataPreparationResult:
    """Full preprocessing pipeline: load, split, impute, scale, and package."""
    df = load_dataset_as_dataframe(config.data.dataset_path)
    feature_cols = [col for col in df.columns if col not in ("user_id", "label")]

    df, dropped_all_nan, feature_cols = drop_nan_columns(df, feature_cols)

    train_users, val_users, test_users = split_by_subject(
        df["user_id"], config.data.train_frac, config.data.val_frac, config.data.seed
    )

    train_df = filter_by_users(df, train_users)
    val_df = filter_by_users(df, val_users)
    test_df = filter_by_users(df, test_users)

    label_to_id = {label: idx for idx, label in enumerate(config.label_order)}

    X_train_raw, y_train = extract_features_and_labels(train_df, feature_cols, label_to_id)
    X_val_raw, y_val = extract_features_and_labels(val_df, feature_cols, label_to_id)
    X_test_raw, y_test = extract_features_and_labels(test_df, feature_cols, label_to_id)

    # fill any remaining NaNs with training-set means 
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_val_imputed = imputer.transform(X_val_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    # normalize features based on training-set stats
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imputed).astype(np.float32)
    X_val = scaler.transform(X_val_imputed).astype(np.float32)
    X_test = scaler.transform(X_test_imputed).astype(np.float32)

    subject_splits = {
        "train_users": train_users,
        "val_users": val_users,
        "test_users": test_users,
    }

    return DataPreparationResult(
        train_features=X_train,
        train_labels=y_train,
        val_features=X_val,
        val_labels=y_val,
        test_features=X_test,
        test_labels=y_test,
        feature_columns=list(feature_cols),
        dropped_all_nan=dropped_all_nan,
        label_to_id=label_to_id,
        subject_splits=subject_splits,
        imputer=imputer,
        scaler=scaler,
    )
