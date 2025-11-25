import numpy as np

def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1-score macro average.
    The F1 from each class is computed, and their unweighted mean is returned.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores = []

    for c in classes:
        # True Positive, False Positive, False Negative
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        # precision and recall per class
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1)

    if len(f1_scores) == 0:
        return 0.0

    return float(np.mean(f1_scores))
