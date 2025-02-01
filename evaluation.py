import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import os
import logging
from datetime import datetime
from src.models.factory import ModelFactory
from src.dataset import create_dataloaders, UnlabeledDataset

def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = get_timestamp().replace(" ", "_").replace(":", "-")
    log_file = os.path.join(log_dir, f'evaluation_log_{timestamp}.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def log_args(log_file, args):
    """Log the command-line arguments to the log file."""
    with open(log_file, 'a') as f:
        f.write("Training Arguments:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write("\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained classification model.")
    parser.add_argument("--model_name", type=str, default='resnet50', help="Model name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint.")
    parser.add_argument("--log_dir", type=str, default='./artifacts/logs', help="Directory to save logs.")
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

# Define the transformations
transform = {
    'test': transforms.Compose([
        transforms.Resize(224),  # Resize image to 224x224
        transforms.ToTensor(),   # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
    ])
}

def main():
    # Parse command-line arguments
    args = parse_args()

    # Set up logging
    log_file = setup_logging(args.log_dir)
    logging.info(f"Evaluation started with model: {args.model_name}")

    # Log the arguments
    log_args(log_file, args)
    logging.info(f"Training started with model: {args.model_name}")

    # Set device and load test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = UnlabeledDataset('data/isbi2025-ps3c-test-dataset.csv', 'data', transform['test'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load the model and checkpoint
    model = ModelFactory(args.model_name, num_classes=3).get_model().to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(
        model, test_loader, criterion, device
    )

    # Log evaluation results
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.2f}")
    logging.info(f"Test Precision: {test_precision:.2f}")
    logging.info(f"Test Recall: {test_recall:.2f}")
    logging.info(f"Test F1 Score: {test_f1:.2f}")

    # Print evaluation results to console
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}")
    print(f"Test Precision: {test_precision:.2f}")
    print(f"Test Recall: {test_recall:.2f}")
    print(f"Test F1 Score: {test_f1:.2f}")

if __name__ == "__main__":
    main()