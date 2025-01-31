import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def save_images_from_dataloader(dataloader, dataset, save_dir, num_images=3):
    """
    Save images from the given dataloader.

    Args:
        dataloader: DataLoader object (train, val, or test loader).
        dataset: Corresponding dataset (train_dataset, val_dataset, or test_dataset).
        save_dir: Directory where images will be saved.
        num_images: Number of images to save.
    """
    os.makedirs(save_dir, exist_ok=True)

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}  # Reverse class mapping

    images, labels = next(iter(dataloader))  # Fetch a batch
    images = images[:num_images]
    labels = labels[:num_images]

    for i in range(num_images):
        img = images[i].numpy().transpose((1, 2, 0))  # Convert tensor to image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean  # Denormalize
        img = np.clip(img, 0, 1)

        label_name = idx_to_class[labels[i].item()]
        label_dir = os.path.join(save_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)

        img_path = os.path.join(label_dir, f"sample_{i}.png")
        plt.imsave(img_path, img)

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
