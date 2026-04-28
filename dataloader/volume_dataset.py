from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyVolumeDataset(Dataset):
    """Dataset for paired 3D seismic volumes and voxel-wise labels."""

    def __init__(self, image_dir, label_dir, normalize=True):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.normalize = normalize
        self.image_paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in {".npy", ".npz"}]
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No .npy or .npz files found in {self.image_dir}.")

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _load_array(path):
        if path.suffix.lower() == ".npz":
            data = np.load(path)
            return data[data.files[0]]
        return np.load(path)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_dir / image_path.name
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {image_path.name}: {label_path}")

        image = self._load_array(image_path).astype(np.float32)
        label = self._load_array(label_path).astype(np.int64)
        if image.shape != label.shape:
            raise ValueError(f"Image and label shape mismatch for {image_path.name}: {image.shape} versus {label.shape}.")

        if self.normalize:
            std = float(image.std())
            image = image - float(image.mean())
            if std > 0:
                image = image / std

        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label).long()
        return image_tensor, label_tensor
