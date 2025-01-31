import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import os
from src.models.factory import ModelFactory
from src.dataset import create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained classification model.")
    parser.add_argument("--model_name", type=str, default='resnet50', help="Model name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint.")
    return parser.parse_args()

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss, test_correct, test_preds, test_targets = 0, 0, [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            test_loss += loss.item()

    test_acc = 100 * test_correct / len(test_loader.dataset)
    test_precision = precision_score(test_targets, test_preds, average='macro')
    test_recall = recall_score(test_targets, test_preds, average='macro')
    test_f1 = f1_score(test_targets, test_preds, average='macro')

    return test_loss / len(test_loader), test_acc, test_precision, test_recall, test_f1

def main():
    # Parse command-line arguments
    args = parse_args()

    # Set device and load test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader, _, _, test_dataset  = create_dataloaders(
        None, 'data/isbi2025-ps3c-test-dataset.csv', 'data', args.batch_size
    )

    # Load the model and checkpoint
    model = ModelFactory(args.model_name, num_classes=3).get_model().to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(
        model, test_loader, criterion, device
    )

    # Print and save evaluation results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}")
    print(f"Test Precision: {test_precision:.2f}")
    print(f"Test Recall: {test_recall:.2f}")
    print(f"Test F1 Score: {test_f1:.2f}")

if __name__ == "__main__":
    main()