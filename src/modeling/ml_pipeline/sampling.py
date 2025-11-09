import numpy as np
from sklearn.utils import resample


def downsample_majority_class(
    X, y,
    target_ratio=3.0,
    random_state=42,
):
    """
    Downsample majority class (label 0) to approximately `target_ratio`:1 relative to the
    minority class (label 1). Returns the concatenated balanced arrays without any SMOTE.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary labels where 0 = non-irrigated (majority), 1 = irrigated (minority).
    target_ratio : float
        Desired majority:minority ratio after downsampling.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_balanced : np.ndarray
    y_balanced : np.ndarray
    """
    X_major = X[y == 0]
    X_minor = X[y == 1]

    n_major_target = int(len(X_minor) * target_ratio)
    X_major_down, _ = resample(
        X_major,
        np.zeros(len(X_major)),
        replace=False,
        n_samples=min(n_major_target, len(X_major)),
        random_state=random_state,
    )

    X_bal = np.vstack([X_major_down, X_minor])
    y_bal = np.hstack([
        np.zeros(len(X_major_down), dtype=int),
        np.ones(len(X_minor), dtype=int),
    ])

    return X_bal, y_bal