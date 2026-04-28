# GR-GCDCN

This repository contains the source code for GR-GCDCN, a structure tensor guided three-dimensional deformable convolutional network for seismic fault segmentation.

The repository intentionally contains code only. It does not include seismic data, trained weights, prediction volumes, logs, figures, or experiment result files.

## Repository Structure

```text
models/
  GCDCNet01.py              GR-GCDCN network definition.
  blocks/
    DCNv3_3D.py             Three-dimensional deformable convolution.
    GCDCNv3.py              Structure tensor guidance and geometric constraint modules.
dataloader/
  volume_dataset.py         Minimal NumPy volume dataset.
scripts/
  train.py                  Minimal training and validation entry point.
  smoke_test.py             Data-free import and forward-pass check.
requirements.txt            Python dependencies.
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Smoke Test

Run a data-free smoke test before launching any experiment:

```bash
python scripts/smoke_test.py
```

The script instantiates GR-GCDCN, runs one forward pass on a dummy 3D tensor, and reports the output tensor shape and parameter count.

## Data Layout

The training script expects paired NumPy volumes. Each image volume and label volume should use the same file name under separate directories.

```text
dataset/
  train/images/sample_000.npy
  train/labels/sample_000.npy
  val/images/sample_100.npy
  val/labels/sample_100.npy
```

Supported file formats are `.npy` and `.npz`. Label values are expected to be integer class indices.

## Minimal Training Example

```bash
python scripts/train.py \
  --train-images /path/to/train/images \
  --train-labels /path/to/train/labels \
  --val-images /path/to/val/images \
  --val-labels /path/to/val/labels \
  --epochs 50 \
  --batch-size 1 \
  --output-dir outputs/example
```

The `outputs/` directory is ignored by Git. Do not commit generated checkpoints, logs, metrics, prediction volumes, or figures.

## License

This project is released under the MIT License.
