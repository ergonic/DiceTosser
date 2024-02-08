import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Path to your image folder
data_dir = 'D:\\dicetoss_grayscale'

# Transform to convert images to PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# DataLoader to iterate through the dataset
dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)

def compute_mean_std(dataloader):
    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in dataloader:
        # Rearrange batch to be the shape of [B, C, W * H]
        images = images.view(images.size(0), images.size(1), -1)
        # Update total_images_count
        total_images_count += images.size(0)
        # Compute mean and std here
        mean += images.mean(2).sum(0) 
        std += images.std(2).sum(0)

    # Final mean and std
    mean /= total_images_count
    std /= total_images_count

    return mean, std

mean, std = compute_mean_std(dataloader)

print(f"Mean: {mean}")
print(f"Std: {std}")
