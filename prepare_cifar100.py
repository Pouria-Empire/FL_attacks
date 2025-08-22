import os
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import random

def create_cifar10_subset(root_dir="./data/cifar10", out_dir="./cifar10_subset", samples_per_class=500):
    """
    Creates a smaller, balanced subset of the CIFAR-10 dataset in an ImageFolder structure.
    """
    print(f"Creating a balanced CIFAR-10 subset with {samples_per_class} samples per class...")
    
    # Download the full dataset
    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True)

    # Create directories
    for split in ['train', 'test']:
        for class_name in trainset.classes:
            os.makedirs(os.path.join(out_dir, split, class_name), exist_ok=True)

    # --- Process and save a balanced subset of the training data ---
    train_class_counts = {c: 0 for c in trainset.classes}
    for img, label in tqdm(trainset, desc="Processing Train Set"):
        class_name = trainset.classes[label]
        if train_class_counts[class_name] < samples_per_class:
            # Randomly assign to train (80%) or test (20%) split
            split = 'train' if random.random() < 0.8 else 'test'
            img_path = os.path.join(out_dir, split, class_name, f"{train_class_counts[class_name]}.png")
            img.save(img_path)
            train_class_counts[class_name] += 1
            
    print("✅ Balanced CIFAR-10 subset created successfully.")

if __name__ == "__main__":
    create_cifar10_subset()