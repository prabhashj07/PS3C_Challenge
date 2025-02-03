import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime
import os
import numpy as np
import logging  
from tqdm import tqdm

from src.models.factory import ModelFactory
from src.dataset import create_dataloaders
from src.early_stopping import EarlyStopping

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, num_classes=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply softmax to the inputs
        inputs = F.softmax(inputs, dim=1)
        
        # Get the probability of the correct class for each target
        targets = F.one_hot(targets, self.num_classes).float()
        
        # Calculate cross entropy
        cross_entropy = -targets * torch.log(inputs + 1e-8)
        
        # Calculate focal loss
        loss = self.alpha * (1 - inputs) ** self.gamma * cross_entropy
        
        # Reduce loss according to the specified reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Get the current timestamp in a readable format
def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Set the seed for reproducibility in training
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument("--train_csv", type=str, default='data/isbi2025-ps3c-train-dataset.csv', help='Path to the training CSV file.')
    parser.add_argument("--test_csv", type=str, default='data/isbi2025-ps3c-test-dataset.csv', help="Path to the test CSV file.")
    parser.add_argument("--data_dir", type=str, default='data', help="Directory containing the dataset images.")
    parser.add_argument("--model_name", type=str, default='resnet50', help="Model name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--use_scheduler", action='store_true', help="Use LR scheduler.")
    parser.add_argument("--save_dir", type=str, default='../artifacts', help="Save directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--gamma", type=float, default=2, help="Focal loss gamma.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes.")
    return parser.parse_args()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def log_metrics(log_file, data):
    with open(log_file, 'a') as f:
        f.write(", ".join(map(str, data)) + "\n")

def save_checkpoint(model, optimizer, epoch, loss, save_dir, is_best=False):
    ensure_dir(save_dir)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{get_timestamp()}.pth"))
    if is_best:
        torch.save(checkpoint, os.path.join(save_dir, "best_checkpoint.pth"))

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = get_timestamp().replace(" ", "_").replace(":", "-")
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
    
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
    with open(log_file, 'a') as f:
        f.write("Training Arguments:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write("\n")

# Main Function
def main():
    # Parse command-line arguments
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    artifacts_dir = './artifacts'
    ensure_dir(artifacts_dir)
    ensure_dir(os.path.join(artifacts_dir, 'checkpoints'))
    
    # Setup logging
    log_file = setup_logging(os.path.join(artifacts_dir, 'logs'))
    logging.info(f"Training started with model: {args.model_name}")

    # Log the arguments
    log_args(log_file, args)
    
    # Create data loaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
        args.train_csv, 
        args.test_csv, 
        args.data_dir, 
        args.batch_size
    )
    
    model = ModelFactory(args.model_name, num_classes=args.num_classes).get_model().to(device)
    
    # Initialize Focal Loss
    label_counts = torch.tensor([40265, 1894, 23146], dtype=torch.float32)
    alpha_values = label_counts.sum() / (args.num_classes * label_counts)
    alpha_normalized = alpha_values / alpha_values.sum()
    alpha_normalized = alpha_normalized.to(device)
    criterion = FocalLoss(gamma=args.gamma, alpha=alpha_normalized, reduction='mean')
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) if args.use_scheduler else None
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Training loop
    best_val_f1 = 0.0  
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct, train_preds, train_targets = 0, 0, [], []
        train_label_counts = np.zeros(args.num_classes, dtype=int)
        
        # Wrap train_loader with tqdm for progress bar
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False)
        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            train_loss += loss.item()
            for label in labels.cpu().numpy():
                train_label_counts[label] += 1
            
            # Update progress bar with current loss
            train_progress.set_postfix({"Loss": loss.item()})
        
        train_acc = 100 * train_correct / len(train_dataset)
        train_precision = precision_score(train_targets, train_preds, average='weighted')
        train_recall = recall_score(train_targets, train_preds, average='weighted')
        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_preds, val_targets = 0, 0, [], []
        val_label_counts = np.zeros(args.num_classes, dtype=int)
        
        # Wrap val_loader with tqdm for progress bar
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, labels in val_progress:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_loss += loss.item()
                for label in labels.cpu().numpy():
                    val_label_counts[label] += 1
                
                # Update progress bar with current loss
                val_progress.set_postfix({"Loss": loss.item()})
        
        val_acc = 100 * val_correct / len(val_dataset)
        val_precision = precision_score(val_targets, val_preds, average='weighted')
        val_recall = recall_score(val_targets, val_preds, average='weighted')
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        
        # Calculate class-wise F1 scores
        class_f1_scores = f1_score(val_targets, val_preds, average=None)  
        avg_f1_score = np.mean(class_f1_scores)

        # Log metrics
        log_metrics(log_file, [epoch, train_loss / len(train_loader), val_loss / len(val_loader),
                            train_acc, val_acc, train_precision, val_precision, train_recall, val_recall, train_f1, val_f1])

        # Log to console and file using logging
        logging.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")
        logging.info(f"Train Accuracy: {train_acc:.2f}, Val Accuracy: {val_acc:.2f}")
        logging.info(f"Train Precision: {train_precision:.2f}, Val Precision: {val_precision:.2f}")
        logging.info(f"Train Recall: {train_recall:.2f}, Val Recall: {val_recall:.2f}")
        logging.info(f"Train F1: {train_f1:.2f}, Val F1: {val_f1:.2f}")
        logging.info(f"Class-wise F1-Scores: {class_f1_scores}")
        logging.info(f"Average F1-Score: {avg_f1_score:.4f}")
        
        # Save checkpoint based on the best F1-Score
        if avg_f1_score > best_val_f1:
            best_val_f1 = avg_f1_score
            save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(artifacts_dir, 'checkpoints'), is_best=True)
        
        # Early stopping based on F1-Score
        early_stopping(-avg_f1_score, model)  
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

if __name__ == "__main__":
    main()