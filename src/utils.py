import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def calculate_sensitivity_specivity(y_true, y_pred, num_classes):
    """
    Calculates sensitivity and specificity for multi-class classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes in the classification task
    
    Returns:
        sensitivity: List of sensitivity values for each class
        specificity: List of specificity values for each class
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    sensitivity = []
    specificity = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        sensitivity.append(TP / (TP + FN) if (TP + FN) != 0 else 0)
        specificity.append(TN / (TN + FP) if (TN + FP) != 0 else 0)

    return sensitivity, specificity
