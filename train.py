import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms  
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime
import os
import numpy as np

from src.models.factory import ModelFactory  
from src.dataset import create_dataloaders, LabeledDataset, transform 
from src.early_stopping import EarlyStopping
from src.utils import calculate_sensitivity_specivity

# Function to get current timestamp in a desired format
def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Set random seed for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU support
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument("--model_name", type=str, default='resnet50', help="Name of the model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--use_scheduler", action='store_true', help="Use learning rate scheduler.")
    parser.add_argument("--save_dir", type=str, default='../artifacts', help="Directory to save logs and models.")
    
    return parser.parse_args()

# Function to initialize directories for saving outputs
def create_output_dirs(save_dir):
    dirs = ['checkpoints', 'logs', 'visualizations', 'metrics']
    for dir_name in dirs:
        dir_path = os.path.join(save_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# Function to log training metrics (accuracy, loss, etc.)
def log_metrics(log_file, epoch, train_loss, val_loss, train_acc, val_acc):
    with open(log_file, 'a') as f:
        f.write(f"{epoch}, {train_loss}, {val_loss}, {train_acc}, {val_acc}\n")

# Function to save model checkpoints
def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename="checkpoint.pth"):
    checkpoint_path = os.path.join(save_dir, "checkpoints", filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

# Function to visualize and save metrics or images during training
def save_visualizations(images, labels, save_dir, epoch, prefix="train"):
    # Save example images during training or validation
    visualization_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(visualization_dir, exist_ok=True)
    for i, img in enumerate(images[:5]):  
        img_path = os.path.join(visualization_dir, f"{prefix}_epoch{epoch}_img{i+1}.png")
        img = transforms.ToPILImage()(img.cpu())  
        img.save(img_path)

# Add this function to ensure the CSV exists before loading
def check_file_exists(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path

# Main training loop
def main():
    # Parse arguments
    args = parse_args()
    set_seed(42)  # Set a fixed seed for reproducibility
    
    # Create output directories for saving results
    create_output_dirs(args.save_dir)
    
    # Set device (CUDA if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_dir = 'data'

    data_dir = os.path.join(os.path.dirname(__file__), 'data')  
    train_csv = os.path.join(data_dir, 'isbi2025-ps3c-train-dataset.csv')
    test_csv = os.path.join(data_dir, 'isbi2025-ps3c-test-dataset.csv')
    
    # Ensure the CSV files exist before proceeding
    check_file_exists(train_csv)
    check_file_exists(test_csv)
    
    # Initialize data loaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
        train_csv, test_csv, image_dir, batch_size=args.batch_size
    )
    
    # Initialize model
    model_factory = ModelFactory(args.model_name, num_classes=3)  
    model = model_factory.get_model()
    model.to(device)  
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Set up learning rate scheduler if specified
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Early stopping setup
    early_stopping = EarlyStopping(patience=10, verbose=True)

    # Initialize logging
    log_file = os.path.join(args.save_dir, 'logs', 'training_log.csv')
    with open(log_file, 'w') as f:
        f.write("Epoch, Train_Loss, Val_Loss, Train_Accuracy, Val_Accuracy\n")
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) 
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_loss += loss.item()
        
        # Validation step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device) 
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_loss += loss.item()

        # Compute metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Log the metrics to file
        log_metrics(log_file, epoch, train_loss / len(train_loader), val_loss / len(val_loader), train_acc, val_acc)
        
        # Save visualizations and model checkpoints
        save_visualizations(images, labels, args.save_dir, epoch)
        save_checkpoint(model, optimizer, epoch, train_loss / len(train_loader), args.save_dir)
        
        # Check early stopping criteria
        early_stopping(val_loss / len(val_loader), model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()

    print(f"Training completed. Checkpoints and logs saved to {args.save_dir}")

if __name__ == "__main__":
    main()