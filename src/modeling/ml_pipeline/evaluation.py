from sklearn.metrics import accuracy_score, f1_score

from sklearn.metrics import accuracy_score, f1_score

def model_metrics(y_pred, y_test):
    """
    Computes accuracy and F1 score for each label (column) in multi-label classification.
    
    Args:
        y_pred: numpy array of shape (N, num_labels)
        y_test: numpy array of shape (N, num_labels)
    
    Returns:
        dict: Dictionary with per-label accuracy and F1 scores.
    """
    metrics = {}

    # Ensure 2D shape
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    num_labels = y_pred.shape[1]

    for i in range(num_labels):
        label_name = f"label_{i}"
        acc = accuracy_score(y_test[:, i], y_pred[:, i])
        f1 = f1_score(y_test[:, i], y_pred[:, i], average='weighted')
        metrics[label_name] = {
            "accuracy": acc,
            "f1_score": f1
        }

    return metrics