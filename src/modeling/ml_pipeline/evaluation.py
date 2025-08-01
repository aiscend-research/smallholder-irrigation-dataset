from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def model_metrics(y_pred, y_test):
    """
    Computes accuracy, precision, recall, and F1 score for the first two mask bands:
        - Band 1: Irrigation Type (0–5, categorical)
        - Band 2: Irrigation Presence (0/1, binary)
    Args:
        y_pred: numpy array of shape (N, >=2)
        y_test: numpy array of shape (N, >=2)
    Returns:
        dict: Dictionary with per-band metrics.
    """
    metrics = {}

    # Ensure 2D shape
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:
        y_pred = y_pred.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    # Only use the first two bands
    y_pred = y_pred[:, :2]
    y_test = y_test[:, :2]

    # Band 1: Irrigation type (categorical: 0–5)
    acc1 = accuracy_score(y_test[:, 0], y_pred[:, 0])
    prec1 = precision_score(y_test[:, 0], y_pred[:, 0], average='weighted', zero_division=0)
    rec1 = recall_score(y_test[:, 0], y_pred[:, 0], average='weighted', zero_division=0)
    f1_1 = f1_score(y_test[:, 0], y_pred[:, 0], average='weighted', zero_division=0)
    metrics["irrigation_type"] = {
        "accuracy": acc1,
        "precision": prec1,
        "recall": rec1,
        "f1_score": f1_1
    }

    # Band 2: Irrigation presence (binary: 0/1)
    acc2 = accuracy_score(y_test[:, 1], y_pred[:, 1])
    prec2 = precision_score(y_test[:, 1], y_pred[:, 1], average='binary', zero_division=0)
    rec2 = recall_score(y_test[:, 1], y_pred[:, 1], average='binary', zero_division=0)
    f1_2 = f1_score(y_test[:, 1], y_pred[:, 1], average='binary', zero_division=0)
    metrics["irrigation_presence"] = {
        "accuracy": acc2,
        "precision": prec2,
        "recall": rec2,
        "f1_score": f1_2
    }

    return metrics