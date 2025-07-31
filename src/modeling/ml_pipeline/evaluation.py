from sklearn.metrics import accuracy_score, f1_score

def model_metrics(y_pred, y_test):
    """
    Computes accuracy and F1 score for the first two mask bands:
        - Band 1: Irrigation Type (0–5)
        - Band 2: Irrigation Presence (0/1)
    Args:
        y_pred: numpy array of shape (N, >=2)
        y_test: numpy array of shape (N, >=2)
    Returns:
        dict: Dictionary with per-band accuracy and F1 scores.
    """
    metrics = {}

    # Ensure 2D shape
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:
        # Only Band 1
        y_pred = y_pred.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    # Only use the first two bands
    y_pred = y_pred[:, :2]
    y_test = y_test[:, :2]

    # Band 1: Irrigation type (categorical: 0–5)
    acc1 = accuracy_score(y_test[:, 0], y_pred[:, 0])
    f1_1 = f1_score(y_test[:, 0], y_pred[:, 0], average='weighted')
    metrics["irrigation_type"] = {
        "accuracy": acc1,
        "f1_score": f1_1
    }

    # Band 2: Irrigation presence (binary: 0/1)
    acc2 = accuracy_score(y_test[:, 1], y_pred[:, 1])
    f1_2 = f1_score(y_test[:, 1], y_pred[:, 1], average='binary')
    metrics["irrigation_presence"] = {
        "accuracy": acc2,
        "f1_score": f1_2
    }

    return metrics