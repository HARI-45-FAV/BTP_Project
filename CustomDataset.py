#CustomDataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, npy_file, labels_file, train_ratio=0.8, transform=None):
        self.data = npy_file
        self.labels = labels_file
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

def load_custom_dataset(batch_size, npy_file, labels_file, train_ratio=0.8):
    

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])

    custom_dataset = CustomDataset(npy_file, labels_file, transform=None)

    # Split the custom dataset into training and testing sets
    dataset_size = len(custom_dataset)
    print(dataset_size)

    if train_ratio < 1:
        train_size = int(train_ratio * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader
    else:
        train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, None



# Usage example:
# train_loader, test_loader = load_custom_dataset(batch_size=64, npy_file='your_data.npy', labels_file='your_labels.npy')
