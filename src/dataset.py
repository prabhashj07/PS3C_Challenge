import os
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

# Get the number of CPU cores available
num_workers = os.cpu_count()

# Transformation pipeline
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Define LabeledDataset class
class LabeledDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {label: idx for idx, label in enumerate(self.data['label'].unique())}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Class for unlabeled test data
class UnlabeledDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Function to create dataset and dataloaders
def create_dataloaders(train_csv, test_csv, image_dir, batch_size=16, num_workers=4):
    # Load full dataset
    full_dataset = LabeledDataset(train_csv, image_dir, transform)

    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Load test dataset
    test_dataset = UnlabeledDataset(test_csv, image_dir, transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

# Main block
if __name__ == "__main__":
    root_dir = '../data/'  
    train_csv = os.path.abspath('../data/isbi2025-ps3c-train-dataset.csv')
    test_csv = os.path.abspath('../data/isbi2025-ps3c-test-dataset.csv')
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(train_csv, test_csv, root_dir)

    # Verification
    print("\nDataset Splits:")
    print(f"Total dataset size: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
