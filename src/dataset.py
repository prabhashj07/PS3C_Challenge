import os
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
import random

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

class LabeledDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, treat_bothcells_as_unhealthy=False):
        # Read CSV file containing image names and labels
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.treat_bothcells_as_unhealthy = treat_bothcells_as_unhealthy

        # Map labels to folder names
        self.label_to_folder = {
            'healthy': 'healthy',
            'unhealthy': 'unhealthy',
            'rubbish': 'rubbish',
            'bothcells': 'bothcells'
        }

        # If treating "bothcells" as "unhealthy", update the mapping
        if self.treat_bothcells_as_unhealthy:
            self.label_to_folder['bothcells'] = 'unhealthy'

        # Create a mapping from class labels to class indices
        self.class_to_idx = {label: idx for idx, label in enumerate(self.data['label'].unique())}

        # Filter out missing images initially
        self._remove_missing_images()
        
    def _remove_missing_images(self):
        """Removes entries from self.data where images are missing."""
        valid_data = []
        for idx in range(len(self.data)):
            img_name = self.data.iloc[idx, 0] 
            label = self.data.iloc[idx, 1] 
            folder_name = self.label_to_folder.get(label, 'rubbish')
            img_path = os.path.join(self.root_dir, folder_name, img_name)
            
            if os.path.exists(img_path):
                valid_data.append(self.data.iloc[idx])

        # Update dataset to only contain valid data
        self.data = pd.DataFrame(valid_data).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # Image file name
        label = self.data.iloc[idx, 1]  # Image label
        folder_name = self.label_to_folder.get(label, 'rubbish')
        img_path = os.path.join(self.root_dir, folder_name, img_name)

        if not os.path.exists(img_path):
            print(f"[WARNING] Missing image: {img_path}. Replacing with a random {label} image.")

            # Select a random replacement image from the same class
            same_class_images = self.data[self.data['label'] == label]['image_name'].tolist()
            if same_class_images:
                img_name = random.choice(same_class_images)
                img_path = os.path.join(self.root_dir, folder_name, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.class_to_idx[label]

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

# Function to count class distribution
def count_classes(dataset):
    if isinstance(dataset, LabeledDataset):
        labels = dataset.data['label'].tolist()
    elif isinstance(dataset, UnlabeledDataset):
        return {}  # No labels for unlabeled data
    else:
        # For datasets created by random_split
        labels = [dataset.dataset.data.iloc[idx, 1] for idx in dataset.indices]
    
    return Counter(labels)

# Function to visualize and save test images 
def visualize_and_save_images(train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, num_images=4, save_dir='../artifacts/saved_images'):
    # Create the directory to save the images
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Visualize and save training images
    print("\nVisualizing and Saving Training Images:")
    images, labels = next(iter(train_loader))
    idx_to_class_train = {v: k for k, v in train_dataset.dataset.class_to_idx.items()}

    for i in range(num_images):
        plt.figure(figsize=(5, 5))
        image = images[i].numpy().transpose((1, 2, 0))  
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean 
        image = np.clip(image, 0, 1)   
        plt.imshow(image)
        plt.title(f"Label: {idx_to_class_train[labels[i].item()]}")
        plt.axis('off')
        save_path = os.path.join(save_dir, f'train_image_{i+1}.png')
        plt.savefig(save_path)  
        plt.close()  

    # Visualize and save validation images
    print("\nVisualizing and Saving Validation Images:")
    images, labels = next(iter(val_loader))
    idx_to_class_val = {v: k for k, v in val_dataset.dataset.class_to_idx.items()}

    for i in range(num_images):
        plt.figure(figsize=(5, 5))
        image = images[i].numpy().transpose((1, 2, 0))  
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean 
        image = np.clip(image, 0, 1)   
        plt.imshow(image)
        plt.title(f"Label: {idx_to_class_val[labels[i].item()]}")
        plt.axis('off')
        save_path = os.path.join(save_dir, f'val_image_{i+1}.png')
        plt.savefig(save_path)  
        plt.close()  # Close the plot to free memory

    # Visualize and save test images
    print("\nVisualizing and Saving Test Images:")
    images = next(iter(test_loader))

    for i in range(num_images):
        plt.figure(figsize=(5, 5))
        image = images[i].numpy().transpose((1, 2, 0))  
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean 
        image = np.clip(image, 0, 1)   
        plt.imshow(image)
        plt.title(f"Test Image {i+1}")
        plt.axis('off')
        save_path = os.path.join(save_dir, f'test_image_{i+1}.png')
        plt.savefig(save_path)  
        plt.close() 

    print(f"Images have been saved to: {save_dir}")

# Function to create dataset and dataloaders
def create_dataloaders(train_csv, test_csv, image_dir, batch_size=16, num_workers=4):
    # Load full dataset
    full_dataset = train_dataset = LabeledDataset(train_csv, image_dir, transform['train'], treat_bothcells_as_unhealthy=True)

    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Load test dataset
    test_dataset = UnlabeledDataset(test_csv, image_dir, transform['test'])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

# Main block
if __name__ == "__main__":
    root_dir = '../data'
    train_csv = os.path.abspath('../data/isbi2025-ps3c-train-dataset.csv')
    test_csv = os.path.abspath('../data/isbi2025-ps3c-test-dataset.csv')

    # Print out the paths to verify correctness
    print(f"Root dir: {root_dir}")
    print(f"Train CSV path: {train_csv}")
    print(f"Test CSV path: {test_csv}")

    # Check if the files exist
    if not os.path.exists(train_csv):
        print(f"Error: Train CSV file not found at {train_csv}")
    if not os.path.exists(test_csv):
        print(f"Error: Test CSV file not found at {test_csv}")

    # Proceed to create dataloaders 
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(train_csv, test_csv, root_dir)

    # Verification
    print("\nDataset Splits:")
    print(f"Total dataset size: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Check class distribution
    train_class_counts = count_classes(train_dataset)
    val_class_counts = count_classes(val_dataset)
    test_class_counts = count_classes(test_dataset)

    print("\nClass Distribution:")
    print(f"Train set: {train_class_counts}")
    print(f"Validation set: {val_class_counts}")
    print(f"Test set: {test_class_counts}")

    # Visualization after splitting
    visualize_and_save_images(train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)