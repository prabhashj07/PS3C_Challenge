import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import argparse
import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from src.models.factory import ModelFactory
from src.dataset import UnlabeledDataset

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
    parser.add_argument("--test_csv", type=str, default='data/isbi2025-ps3c-test-dataset.csv', required=True, help="Path to the test CSV file.")
    parser.add_argument("--data_dir", type=str, default='data', required=True, help="Directory containing the dataset images.")
    parser.add_argument("--model_name", type=str, default='resnet50', help="Model name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint.")
    parser.add_argument("--log_dir", type=str, default='./artifacts/logs', help="Directory to save logs.")
    return parser.parse_args()

def predict_labels(model, test_loader, device):
    """Predict labels for unlabeled data."""
    model.eval()
    image_names, all_preds = [], []

    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  
            all_preds.extend(predicted.cpu().numpy()) 
            image_names.extend(img_names)  

    return image_names, all_preds

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss, test_correct, test_preds, test_targets = 0, 0, [], []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:  
                images, labels = batch
                labels = labels.to(device)
            else:  
                images = batch[0]
                labels = None

            images = images.to(device)
            outputs = model(images)

            if labels is not None:  
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_preds.extend(predicted.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())
            else:
                _, predicted = torch.max(outputs, 1)
                test_preds.extend(predicted.cpu().numpy())

    if labels is not None:
        test_acc = 100 * test_correct / len(test_loader.dataset)
        test_precision = precision_score(test_targets, test_preds, average='weighted')
        test_recall = recall_score(test_targets, test_preds, average='weighted')
        test_f1 = f1_score(test_targets, test_preds, average='weighted')
        class_f1_report = classification_report(test_targets, test_preds, target_names=[f'Class {i}' for i in range(3)])
        avg_f1_score = test_f1
        return test_loss / len(test_loader), test_acc, test_precision, test_recall, test_f1, class_f1_report, avg_f1_score
    else:
        return None, None, None, None, None, None, None

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

def main():
    # Parse command-line arguments
    args = parse_args()

    # Set up logging
    log_file = setup_logging(args.log_dir)
    logging.info(f"Evaluation started with model: {args.model_name}")

    # Log the arguments
    log_args(log_file, args)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load the test dataset (images only)
    test_dataset = UnlabeledDataset(args.test_csv, args.data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    model = ModelFactory(args.model_name, num_classes=3).get_model().to(device)
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False) 
        logging.info("Model loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Checkpoint file not found: {args.model_path}")
    except KeyError as e:
        logging.error(f"Mismatch in model architecture or state dict keys: {e}")
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")

    # Evaluate the model (only if labels are available)
    if hasattr(test_dataset, 'labels'):
        try:
            test_loss, test_acc, test_precision, test_recall, test_f1, class_f1_scores, avg_f1_score = evaluate_model(
                model, test_loader, criterion, device
            )
            logging.info(f"Test Loss: {test_loss:.4f}")
            logging.info(f"Test Accuracy: {test_acc:.2f}%")
            logging.info(f"Test Precision: {test_precision:.4f}")
            logging.info(f"Test Recall: {test_recall:.4f}")
            logging.info(f"Test F1 Score: {test_f1:.4f}")
            logging.info(f"Class-wise F1 Scores: {class_f1_scores}")
            logging.info(f"Average F1-Score: {avg_f1_score:.4f}")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            return
    else:
        logging.info("No labels available for evaluation. Skipping evaluation metrics.")

    # Predict labels for the test images
    try:
        image_names, all_preds = predict_labels(model, test_loader, device)
    except Exception as e:
        logging.error(f"Error during label prediction: {e}")
        return

    class_labels = ['healthy', 'unhealthy', 'rubbish']
    predicted_labels = [class_labels[pred] for pred in all_preds]

    results_df = pd.DataFrame({
        'image_name': image_names,
        'label': predicted_labels
    })
    predictions_file = os.path.join(args.log_dir, 'predictions.csv')
    results_df.to_csv(predictions_file, index=False)

    logging.info(f"Predictions saved to {predictions_file}")
    print(f"Predictions saved to {predictions_file}")

    print("Evaluation complete.")

    # Predict labels for the test images
    try:
        image_names, all_preds = predict_labels(model, test_loader, device)
    except Exception as e:
        logging.error(f"Error during label prediction: {e}")
        return

    # Map predicted class indices to class labels
    class_labels = ['healthy', 'unhealthy', 'rubbish'] 
    predicted_labels = [class_labels[pred] for pred in all_preds]

    # Save image names and predicted labels to a CSV file
    results_df = pd.DataFrame({
        'image_name': image_names,
        'label': predicted_labels
    })
    predictions_file = os.path.join(args.log_dir, 'predictions.csv')
    results_df.to_csv(predictions_file, index=False)

    logging.info(f"Predictions saved to {predictions_file}")
    print(f"Predictions saved to {predictions_file}")

    print("Evaluation complete.")

if __name__ == "__main__":
    main()