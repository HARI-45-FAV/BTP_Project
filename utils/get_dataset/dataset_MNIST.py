import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_MNIST_full(batch_size=32, train_subset=10000, test_subset=None):
    transform = transforms.Compose([
        transforms.Resize((30, 30)),
        transforms.ToTensor(),
        # NOTE: Normalize intentionally omitted — quantum layer requires values in [0, 1]
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Subset the training set for faster quantum preprocessing
    if train_subset is not None and train_subset < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:train_subset]
        train_dataset = Subset(train_dataset, indices)

    # Subset the test set to match original dataset scale
    if test_subset is not None and test_subset < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:test_subset]
        test_dataset = Subset(test_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, None
