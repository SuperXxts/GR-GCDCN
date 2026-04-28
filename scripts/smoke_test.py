import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.GCDCNet01 import GCDCNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCDCNet(num_channels=1, num_classes=2, geometric_loss_weight=0.1).to(device)
    model.eval()

    x = torch.randn(1, 1, 32, 32, 32, device=device)
    with torch.no_grad():
        y = model(x)

    params = sum(p.numel() for p in model.parameters())
    print(f"device={device}")
    print(f"input_shape={tuple(x.shape)}")
    print(f"output_shape={tuple(y.shape)}")
    print(f"parameters={params}")


if __name__ == "__main__":
    main()
