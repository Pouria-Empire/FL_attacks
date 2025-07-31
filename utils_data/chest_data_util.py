import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
from typing import Tuple, List

FINDINGS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
]

class ChestXRayDataset(Dataset):
    def __init__(self, data_path: str, df: pd.DataFrame, transform=None):
        self.img_dir = os.path.join(data_path, 'images', 'images')
        self.df = df
        self.transform = transform #<-- Expose transform
        self.labels = FINDINGS

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image Index'])
        image = Image.open(img_path).convert('L')
        label_vector = torch.zeros(len(self.labels), dtype=torch.float32)
        for finding in row['Finding Labels'].split('|'):
            if finding in self.labels:
                label_vector[self.labels.index(finding)] = 1.0
        if self.transform:
            image = self.transform(image)
        return image, label_vector

def load_data(data_path: str, train_list_file: str, test_list_file: str) -> Tuple[Dataset, Dataset]:
    df = pd.read_csv(os.path.join(data_path, 'Data_Entry_2017_v2020.csv'))
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    with open(os.path.join(data_path, train_list_file), 'r') as f:
        train_files = [os.path.basename(line.strip()) for line in f]
    with open(os.path.join(data_path, test_list_file), 'r') as f:
        test_files = [os.path.basename(line.strip()) for line in f]
    df_train = df[df['Image Index'].isin(train_files)]
    df_test = df[df['Image Index'].isin(test_files)]
    train_set = ChestXRayDataset(data_path, df_train.reset_index(drop=True), transform=transform)
    test_set = ChestXRayDataset(data_path, df_test.reset_index(drop=True), transform=transform)
    return train_set, test_set

def get_client_data(cid: str, total_clients: int, data_path: str, train_list_file: str, test_list_file: str) -> Tuple[Subset, Dataset]:
    train_data_full, test_data = load_data(data_path, train_list_file, test_list_file)
    client_id_numeric = int(cid.replace("client", "")); client_idx = client_id_numeric - 1
    len_train = len(train_data_full)
    indices = list(range(
        client_idx * (len_train // total_clients),
        (client_idx + 1) * (len_train // total_clients)
    ))
    return Subset(train_data_full, indices), test_data