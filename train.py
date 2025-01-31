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

from src.utils import save_images_from_dataloader
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
    parser.add_argument("--model_name", type=str, default='resnet50', help="Model name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--use_scheduler", action='store_true', help="Use LR scheduler.")
    parser.add_argument("--save_dir", type=str, default='../artifacts', help="Save directory.")
    return parser.parse_args()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def log_metrics(log_file, data):
    with open(log_file, 'a') as f:
        f.write(", ".join(map(str, data)) + "\n")

def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    ensure_dir(save_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(save_dir, "checkpoint.pth"))

# Main function
# Main function
def main():
    # Parse command-line arguments
    args = parse_args()
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Update the save directory to the new location
    artifacts_dir = './artifacts'  # Assuming artifacts folder is at the root of the project
    ensure_dir(artifacts_dir)
    ensure_dir(os.path.join(artifacts_dir, 'checkpoints'))
    ensure_dir(os.path.join(artifacts_dir, 'logs'))
    
    # Create data loaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
        'data/isbi2025-ps3c-train-dataset.csv', 
        'data/isbi2025-ps3c-test-dataset.csv', 
        'data', 
        args.batch_size
    )
    
    model = ModelFactory(args.model_name, num_classes=3).get_model().to(device)

    # Define your label counts and calculate alpha values
    label_counts = torch.tensor([40265, 1894, 23146], dtype=torch.float32)
    total_samples = label_counts.sum()
    alpha_values = total_samples / (3 * label_counts) 
    alpha_normalized = alpha_values / alpha_values.sum()  
    alpha_normalized = alpha_normalized.to(device)

    # Initialize Focal Loss criterion
    criterion = FocalLoss(gamma=2, alpha=alpha_normalized, num_classes=3, reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) if args.use_scheduler else None
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    log_file = os.path.join(artifacts_dir, 'logs', 'training_log.csv')
    with open(log_file, 'w') as f:
        f.write("Epoch, Train_Loss, Val_Loss, Train_Acc, Val_Acc, Train_Precision, Val_Precision, Train_Recall, Val_Recall, Train_F1, Val_F1\n")
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct, train_preds, train_targets = 0, 0, [], []
        
        print(f"Epoch {epoch + 1}/{args.epochs} - Training")
        # Initialize label counters
        train_label_counts = np.zeros(3, dtype=int)  
        
        for images, labels in train_loader:
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

            # Count the labels
            for label in labels.cpu().numpy():
                train_label_counts[label] += 1
        
        train_acc = 100 * train_correct / len(train_dataset)
        train_precision = precision_score(train_targets, train_preds, average='macro')
        train_recall = recall_score(train_targets, train_preds, average='macro')
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        
        print(f"Train Label Counts: {train_label_counts}")
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_preds, val_targets = 0, 0, [], []
        
        print(f"Epoch {epoch + 1}/{args.epochs} - Validation")
        # Initialize label counters for validation
        val_label_counts = np.zeros(3, dtype=int)  
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_loss += loss.item()

                # Count the labels
                for label in labels.cpu().numpy():
                    val_label_counts[label] += 1
        
        val_acc = 100 * val_correct / len(val_dataset)
        val_precision = precision_score(val_targets, val_preds, average='macro')
        val_recall = recall_score(val_targets, val_preds, average='macro')
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        print(f"Val Label Counts: {val_label_counts}")
        
        log_metrics(log_file, [epoch, train_loss / len(train_loader), val_loss / len(val_loader),
                                train_acc, val_acc, train_precision, val_precision, train_recall, val_recall, train_f1, val_f1])
        
        print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"Train Accuracy: {train_acc:.2f}, Val Accuracy: {val_acc:.2f}")
        print(f"Train Precision: {train_precision:.2f}, Val Precision: {val_precision:.2f}")
        print(f"Train Recall: {train_recall:.2f}, Val Recall: {val_recall:.2f}")
        print(f"Train F1: {train_f1:.2f}, Val F1: {val_f1:.2f}")
        
        save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(artifacts_dir, 'checkpoints'))
        if scheduler:
            scheduler.step()
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    main()