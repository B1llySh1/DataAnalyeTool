# model_training.py
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score \
                            , precision_recall_fscore_support, log_loss

def calculate_score_base_on_metric(X, y, model, metric, test_set, X_test=None, y_test=None):
    model_clone = clone(model)
    model_clone.fit(X, y)
    if test_set:
        y_pred = model_clone.predict(X_test)
        y_true = y_test
    else:
        y_pred = model_clone.predict(X)
        y_true = y
    supported = [["MSE", "MAE", "r2"],["accuracy", "precision", "recall", "f1"], ["log_loss"]]
    if metric == "MSE":
        return mean_squared_error(y_true, y_pred)
    elif metric == "MAE":
        return mean_absolute_error(y_true, y_pred)
    elif metric == "r2":
        return r2_score(y_true, y_pred)
    elif metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "precision":
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return precision
    elif metric == "recall":
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return recall
    elif metric == "f1":
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return f1
    elif metric == "log_loss":
        return log_loss(y_true, y_pred)
    else:
        raise KeyError("Unsupported Metric")

def calculate_influence_base_on_metric(metric, base_score, current_score):
    # Errors: lower better
    if metric in ["MSE", "MAE", "log_loss"]:
        influence = current_score - base_score
    # Accuracy: higher better
    elif metric in ["accuracy", "precision", "recall", "f1", "r2"]:
        influence = base_score - current_score
    else:
        raise KeyError("Unsupported Metric")
    return influence


def calculate_loo_influence_parallel(index, X, y, model, metric, base_score, X_test=None, y_test=None, test_set=False):
    X_train_loo = X.drop(X.index[index])
    y_train_loo = np.delete(y, index)
    # leave-one-out training logic
    # X_train = np.delete(X, index, axis=0)
    # y_train = np.delete(y, index, axis=0)
    # X_test = X[index].reshape(1, -1)
    # y_test = y[index].reshape(1,)
    current_score = calculate_score_base_on_metric(X_train_loo, y_train_loo, model, metric, test_set, X_test=X_test, y_test=y_test)
    return calculate_influence_base_on_metric(metric, base_score, current_score)
