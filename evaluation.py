"""7. Hitung confusion matrix dan empat metrik untuk kelas positif liver."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate(y_test, prediction, y_balanced):
    return {
        'train_counts': np.bincount(y_balanced, minlength=2).tolist(),
        'matrix': confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
        'accuracy': accuracy_score(y_test, prediction),
        'precision': precision_score(y_test, prediction, zero_division=0),
        'recall': recall_score(y_test, prediction, zero_division=0),
        'f1': f1_score(y_test, prediction, zero_division=0),
    }
