"""Image and label loading for the six-class TA-DOTCNet benchmark."""

import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


class RemoteCLIPDataset(Dataset):
    """Load image-label pairs from alphabetically ordered class folders."""

    def __init__(self, root_dir, sub_dir, transform=None):
        self.root_dir = os.path.join(root_dir, sub_dir)
        self.transform = transform if transform else default_transform
        self.data = []
        self.class_to_idx = {}
        self.class_names = []
        self._prepare_data()

    def _prepare_data(self):
        if not os.path.isdir(self.root_dir):
            raise ValueError(
                f"[ERROR] Dataset root '{self.root_dir}' not found or is not a directory."
            )

        self.class_names = sorted(
            name for name in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, name))
        )
        self.class_to_idx = {
            name: index for index, name in enumerate(self.class_names)
        }

        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            if not os.path.isdir(class_dir):
                continue
            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    self.data.append((os.path.join(class_dir, filename), class_idx))

        if not self.data:
            raise ValueError(f"[ERROR] Dataset '{self.root_dir}' has no valid image files.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_training(root_dir, sub_dir, batch_size, kwargs, transform=None):
    dataset = RemoteCLIPDataset(root_dir, sub_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return dataset, loader


def load_testing(root_dir, sub_dir, batch_size, kwargs, transform=None):
    dataset = RemoteCLIPDataset(root_dir, sub_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return dataset, loader


__all__ = ["RemoteCLIPDataset", "load_training", "load_testing"]
