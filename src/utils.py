import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix

def print_label_distribution(loader:DataLoader, class_names: list, name: str):
    """
    Print the label distribution in the given dataloader.

    Args:
        loader(DataLoader): The DataLoader whose distribution is supposed to be calculated.
        slotclass (list): List of class names corresponding to the labels.
        name (str): The name of the dataset (train, test, or validation).
    """
    label_counts = {}
    # Iterate through the loader and count occurrences of each label
    for _, labels in loader:
        for label in labels:
            label = label.item()  # Convert tensor to integer
            label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\n{name} loader label distribution:")
    for label in sorted(label_counts.keys()):
        print(f"Class {class_names[label]}: {label_counts[label]} samples")

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
