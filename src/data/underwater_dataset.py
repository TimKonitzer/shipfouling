import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from pathlib import Path
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

class UnderwaterImageDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Name', 'Label'])
        
        self.classes = sorted(df['Label'].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        grouped = df.groupby('Name')['Label'].apply(lambda x: x.mode()[0]).reset_index()
        self.image_names = grouped['Name'].tolist()
        self.image_labels = grouped['Label'].tolist()
        
    def __len__(self):
        return len(self.image_names)
        
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label = self.image_labels[idx]
        
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        target = torch.tensor(self.class_to_idx[label], dtype=torch.long)
        return image, target


class UnderwaterPatchDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None, patch_size=224):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.patch_size = patch_size
        
        df = pd.read_csv(csv_path)
        self.df = df.dropna(subset=['Name', 'Label', 'Row', 'Column']).reset_index(drop=True)
        
        self.classes = sorted(self.df['Label'].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.class_counts = self.df['Label'].value_counts().to_dict()
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row_data = self.df.iloc[idx]
        img_name = row_data['Name']
        r = int(row_data['Row'])
        c = int(row_data['Column'])
        label = row_data['Label']
        
        img_path = self.images_dir / img_name
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (self.patch_size, self.patch_size), (0, 0, 0))
            print(f"Warning: Could not load {img_path}, using dummy. Error: {e}")

        half = self.patch_size // 2
        left = c - half
        top = r - half
        right = c + half
        bottom = r + half
        
        patch = image.crop((left, top, right, bottom))
        
        if patch.size[0] < self.patch_size or patch.size[1] < self.patch_size:
            pad_left = max(0, -left)
            pad_top = max(0, -top)
            pad_right = max(0, right - image.width)
            pad_bottom = max(0, bottom - image.height)
            
            from torchvision.transforms.functional import pad as tv_pad
            patch = tv_pad(patch, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
            if patch.size != (self.patch_size, self.patch_size):
                patch = patch.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        
        if self.transform is not None:
            patch = self.transform(patch)
            
        target = torch.tensor(self.class_to_idx[label], dtype=torch.long)
        return patch, target
