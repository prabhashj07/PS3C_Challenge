import os
import torch
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


class Padding:
    def __init__(self, padding=20, fill=(255, 255, 255)):
        self.padding = [padding] * 4
        self.fill = fill

    def __call__(self, img):
        return F.pad(img, self.padding, fill=self.fill)

# Transformation pipeline
transform = {
    'train': transforms.Compose([
        Padding(padding=20, fill=(255, 255, 255)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        Padding(padding=20, fill=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        Padding(padding=20, fill=(255, 255, 255)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def calculate_class_weights(dataset):
    if isinstance(dataset, LabeledDataset):
        labels = dataset.data['label'].tolist()
    else:
        labels = [dataset.dataset.data.iloc[idx, 1] for idx in dataset.indices]

    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    
    weight_sum = sum(class_weights.values())
    class_weights = {cls: weight / weight_sum for cls, weight in class_weights.items()}
    
    if isinstance(dataset, LabeledDataset):
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    else:
        idx_to_class = {v: k for k, v in dataset.dataset.class_to_idx.items()}
    
    weights = [class_weights[idx_to_class[idx]] for idx in sorted(idx_to_class.keys())]
    
    return torch.tensor(weights, dtype=torch.float32)

class LabeledDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, treat_bothcells_as_unhealthy=False, balance_classes=False):
        # Read CSV file containing image names and labels
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.treat_bothcells_as_unhealthy = treat_bothcells_as_unhealthy
        self.balance_classes = balance_classes

        # Map labels to folder names
        self.label_to_folder = {
            'rubbish': 'rubbish',
            'unhealthy': 'unhealthy',
            'healthy': 'healthy',
            'bothcells': 'bothcells'
        }

        # If treating "bothcells" as "unhealthy", update the mapping
        if self.treat_bothcells_as_unhealthy:
            self.label_to_folder['bothcells'] = 'unhealthy'

        # Define class-to-index mapping
        self.class_to_idx = {'rubbish': 0, 'unhealthy': 1, 'healthy': 2}

        # Filter out missing images initially
        self._remove_missing_images()

        # Balance classes 
        if self.balance_classes:
            self._balance_classes()
        
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

    def _balance_classes(self):
        """Balances the dataset by reducing the number of 'rubbish' samples to match 'healthy' samples."""
        healthy_count = self.data[self.data['label'] == 'healthy'].shape[0]
        rubbish_count = self.data[self.data['label'] == 'rubbish'].shape[0]

        if rubbish_count > healthy_count:
            # Reduce the number of 'rubbish' samples to match 'healthy' samples
            rubbish_samples = self.data[self.data['label'] == 'rubbish']
            rubbish_samples = rubbish_samples.sample(n=healthy_count, random_state=42)  # Randomly sample to match healthy count
            other_samples = self.data[self.data['label'] != 'rubbish']
            self.data = pd.concat([other_samples, rubbish_samples]).reset_index(drop=True)

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
        
        label = self.class_to_idx[label]

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
        # Get the image name from the CSV file
        img_name = self.data.iloc[idx, 0]  
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            # print(f"Image {img_name} shape: {image.shape}")
        
        return image, img_name

# Function to count class distribution
def count_classes(dataset):
    if isinstance(dataset, LabeledDataset):
        labels = ['rubbish', 'unhealthy', 'healthy']
    elif isinstance(dataset, UnlabeledDataset):
        return {}  # No labels for unlabeled data
    else:
        # For datasets created by random_split
        labels = [dataset.dataset.data.iloc[idx, 1] for idx in dataset.indices]
    
    return Counter(labels)

def visualize_and_save_images(train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, num_images=4, save_dir='../artifacts/saved_images'):
    # Create the directory to save the images
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Function to denormalize an image
    def denormalize(image, mean, std):
        if image.ndim == 3:  # (C, H, W)
            image = image.transpose((1, 2, 0))  
        image = std * image + mean  
        image = np.clip(image, 0, 1)  
        return image

    # Visualize and save training images
    print("\nVisualizing and Saving Training Images:")
    images, labels = next(iter(train_loader))
    idx_to_class_train = {v: k for k, v in train_dataset.dataset.class_to_idx.items()}

    for i in range(min(num_images, len(images))): 
        plt.figure(figsize=(5, 5))
        image = images[i].numpy() 
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = denormalize(image, mean, std)
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

    for i in range(min(num_images, len(images))): 
        plt.figure(figsize=(5, 5))
        image = images[i].numpy()  
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = denormalize(image, mean, std)
        plt.imshow(image)
        plt.title(f"Label: {idx_to_class_val[labels[i].item()]}")
        plt.axis('off')
        save_path = os.path.join(save_dir, f'val_image_{i+1}.png')
        plt.savefig(save_path)
        plt.close()

    # Visualize and save test images
    print("\nVisualizing and Saving Test Images:")
    test_data = next(iter(test_loader))
    if isinstance(test_data, (list, tuple)):  
        images = test_data[0]
    else:
        images = test_data

    for i in range(min(num_images, len(images))):  
        plt.figure(figsize=(5, 5))
        image = images[i].numpy() 
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = denormalize(image, mean, std)
        plt.imshow(image)
        plt.title(f"Test Image {i+1}")
        plt.axis('off')
        save_path = os.path.join(save_dir, f'test_image_{i+1}.png')
        plt.savefig(save_path)
        plt.close()

    print(f"Images have been saved to: {save_dir}")

# Function to create dataset and dataloaders
def create_dataloaders(train_csv, test_csv, image_dir, batch_size=16, num_workers=4, balance_classes=False):
    # Load full dataset
    full_dataset = train_dataset = LabeledDataset(train_csv, image_dir, transform['train'], treat_bothcells_as_unhealthy=True, balance_classes=balance_classes)

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

    # Proceed to create dataloaders with balancing
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(train_csv, test_csv, root_dir, balance_classes=True)

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

    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    print(f"\nClass Weights: {class_weights}")

    # Visualization after splitting
    visualize_and_save_images(train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
