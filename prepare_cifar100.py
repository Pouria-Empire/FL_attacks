import os
from PIL import Image
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

# Transform: CIFAR100 -> grayscale 128x128
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128))
])

def save_dataset(dataset, classes, split, out_dir="./data/cifar10_dataset"):
    base_dir = os.path.join(out_dir, split)
    os.makedirs(base_dir, exist_ok=True)

    for class_name in classes:
        os.makedirs(os.path.join(base_dir, class_name), exist_ok=True)

    print(f"Saving {split} dataset...")
    for idx, (img, label) in enumerate(tqdm(dataset)):
        class_name = classes[label]
        img_path = os.path.join(base_dir, class_name, f"{idx}.png")
        img.save(img_path)

if __name__ == "__main__":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    classes = trainset.classes

    save_dataset(trainset, classes, "train")
    save_dataset(testset, classes, "test")

    print("\n✅ CIFAR-100 prepared successfully in ImageFolder format!")
