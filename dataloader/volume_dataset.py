from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyVolumeDataset(Dataset):
    """Dataset for paired 3D seismic volumes and voxel-wise labels."""

    def __init__(self, image_dir, label_dir, volume_shape=(128, 128, 128), normalize=True, transpose=True):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.volume_shape = tuple(volume_shape)
        self.normalize = normalize
        self.transpose = transpose
        self.image_paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in {".dat", ".npy", ".npz"}]
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No .dat, .npy, or .npz files found in {self.image_dir}.")

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _load_array(path, dtype, volume_shape):
        if path.suffix.lower() == ".npz":
            data = np.load(path)
            return data[data.files[0]].astype(dtype, copy=False)
        if path.suffix.lower() == ".npy":
            return np.load(path).astype(dtype, copy=False)
        array = np.fromfile(path, dtype=dtype)
        return array.reshape(volume_shape)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_dir / image_path.name
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {image_path.name}: {label_path}")

        image = self._load_array(image_path, np.float32, self.volume_shape)
        label = self._load_array(label_path, np.float32, self.volume_shape).astype(np.int64, copy=False)
        if image.shape != label.shape:
            raise ValueError(f"Image and label shape mismatch for {image_path.name}: {image.shape} versus {label.shape}.")

        if self.transpose:
            image = np.transpose(image)
            label = np.transpose(label)

        if self.normalize:
            std = float(image.std())
            image = image - float(image.mean())
            if std > 0:
                image = image / std

        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label).long()
        return image_tensor, label_tensor
