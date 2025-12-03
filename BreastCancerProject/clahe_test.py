import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# =====================================
# 1) Custom CLAHE Transform
# =====================================
class CLAHETransform:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def __call__(self, img_tensor):
        # img_tensor → tensor with shape [1, H, W]
        img_np = img_tensor.numpy().squeeze(0)  # [H, W]

        # convert to uint8 0-255
        img_np = (img_np * 255).astype(np.uint8)

        # apply CLAHE
        clahe_img = self.clahe.apply(img_np)

        # back to float32 tensor with shape [1, H, W]
        clahe_img = clahe_img.astype(np.float32) / 255.0
        return torch.tensor(clahe_img).unsqueeze(0)  # add channel back


# =====================================
# 2) Dataset Class
# =====================================
class CalcDataset(Dataset):
    def __init__(self, csv_file, transform=None, use_clahe=False):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.use_clahe = use_clahe
        self.clahe = CLAHETransform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label = int(self.df.iloc[idx, 1])

        img = Image.open(img_path).convert("L")   # grayscale
        img = transforms.ToTensor()(img)         # [1, H, W] float32

        # Apply CLAHE if enabled
        if self.use_clahe:
            img = self.clahe(img)

        # Apply PyTorch transforms (resize, normalize, etc.)
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


# =====================================
# 3) Create train & test loaders
# =====================================
train_csv = os.path.join(os.getcwd(), "calc_train_paths.csv")
test_csv = os.path.join(os.getcwd(), "calc_test_paths.csv")

# Standard transforms AFTER CLAHE
common_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Datasets
train_dataset = CalcDataset(train_csv, transform=common_transform, use_clahe=True)
test_dataset = CalcDataset(test_csv, transform=common_transform, use_clahe=True)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# =====================================
# 4) Test one batch
# =====================================
if __name__ == "__main__":
    imgs, labels = next(iter(train_loader))
    print("Batch image shape:", imgs.shape)
    print("Batch labels:", labels[:10])
