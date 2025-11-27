# src/utils/preprocessing.py
import numpy as np


def fill_missing(
    X: np.ndarray,
    strategy: str = "median",
    constant_value: float = 0.0,
):
    """
    Fill missing values (np.nan) in X.
    X is assumed numeric (already encoded if categorical).

        strategy:
            - "median": fill using the column median
            - "mean": fill using the column mean
            - "zero": fill with 0
            - "constant": fill with the provided constant_value

        Return:
            X_filled: np.ndarray (copy, not in-place)
    """
    X = np.array(X, dtype=float)  # ensure float type so np.nan can be used
    X_filled = X.copy()

    if strategy == "zero":
        X_filled[np.isnan(X_filled)] = 0.0
        return X_filled

    if strategy == "constant":
        X_filled[np.isnan(X_filled)] = constant_value
        return X_filled

    # mean/median per column
    for j in range(X_filled.shape[1]):
        col = X_filled[:, j]
        mask_nan = np.isnan(col)

        if not np.any(mask_nan):
            continue

        if strategy == "mean":
            value = np.nanmean(col)
        elif strategy == "median":
            value = np.nanmedian(col)
        else:
            raise NotImplementedError(f"Strategy {strategy} not supported yet")

        col[mask_nan] = value
        X_filled[:, j] = col

    return X_filled


def one_hot_encode(labels: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """
    One-hot encode integer labels.
    labels: shape (n_samples,), contains integer class labels (0..K-1)
    num_classes: if None, inferred from max(labels) + 1
    """
    labels = np.asarray(labels, dtype=int)

    if num_classes is None:
        num_classes = int(labels.max()) + 1

    n_samples = labels.shape[0]
    one_hot = np.zeros((n_samples, num_classes), dtype=float)
    one_hot[np.arange(n_samples), labels] = 1.0
    return one_hot


def standardize(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
):
    """
        Standardize features: (x - mean) / std.
        If mean and std are None they will be computed from X and returned with X_scaled.
        If mean and std are provided, use them (for test data).

        Return:
            X_scaled, mean, std
    """
    X = np.asarray(X, dtype=float)

    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        # avoid division by zero
        std[std == 0.0] = 1.0

    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def label_encode(labels: np.ndarray) -> np.ndarray:
    """
    Label encode string or categorical labels to integers.
    labels: shape (n_samples,), contains string or categorical labels
    Returns:
        encoded_labels: shape (n_samples,), integer labels from 0 to K-1
    """
    labels = np.asarray(labels)
    unique_labels, encoded_labels = np.unique(labels, return_inverse=True)
    return encoded_labels