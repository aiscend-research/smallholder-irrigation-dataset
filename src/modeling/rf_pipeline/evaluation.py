from sklearn.metrics import accuracy_score, f1_score

def predict(clf, X_test):
    y_pred = clf.predict(X_test)
    return y_pred

def model_metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1