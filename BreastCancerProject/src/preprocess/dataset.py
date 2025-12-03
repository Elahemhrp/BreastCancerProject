import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CalcDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        csv_path: مسیر فایل calc_train_paths.csv یا calc_test_paths.csv
        transform: تابع/کامپوزیشن transformهای torchvision
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        # تعداد نمونه‌ها
        return len(self.df)

    def __getitem__(self, idx):
        # یک ردیف از CSV
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        label = int(row["label"])

        # خواندن تصویر (ماموگرافی‌ها خاکستری هستند)
        img = Image.open(img_path).convert("L")  # L = grayscale

        # اگر transform تعریف شده بود، اعمال کن
        if self.transform is not None:
            img = self.transform(img)

        # تبدیل label به tensor
        label = torch.tensor(label, dtype=torch.long)

        return img, label


def get_transforms():
    # Transform برای train (کمی augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),        # همه را یک اندازه می‌کنیم
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),                # تبدیل به Tensor: شکل [1, H, W]
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Transform برای test/validation (بدون augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return train_transform, test_transform


def build_dataloaders(
    train_csv=os.path.join(os.getcwd(),"calc_train_paths.csv"),
    test_csv=os.path.join(os.getcwd(),"calc_test_paths.csv"),
    batch_size=16,
):
    train_transform, test_transform = get_transforms()

    train_dataset = CalcDataset(train_csv, transform=train_transform)
    test_dataset = CalcDataset(test_csv, transform=test_transform)

    # num_workers روی ویندوز بهتره 0 باشه
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # تست اولیه
    train_loader, test_loader = build_dataloaders()

    # یک batch از train بگیریم
    images, labels = next(iter(train_loader))
    print("Train batch - images shape:", images.shape)
    print("Train batch - labels shape:", labels.shape)
    print("Label sample:", labels[:10])
