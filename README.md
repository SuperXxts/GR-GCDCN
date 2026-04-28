# GR-GCDCN

This repository contains the source code for GR-GCDCN, a structure tensor guided three-dimensional deformable convolutional network for seismic fault segmentation.

## Repository Structure

```text
models/
  GCDCNet01.py              GR-GCDCN network definition.
  blocks/
    DCNv3_3D.py             Three-dimensional deformable convolution.
    GCDCNv3.py              Structure tensor guidance and geometric constraint modules.
dataloader/
  volume_dataset.py         Minimal NumPy volume dataset.
data/
  faultseg3d_sample/        Small processed FaultSeg3D sample for direct testing.
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

## Datasets

The repository includes a small processed FaultSeg3D sample so that the training script can be executed directly after cloning. Full-scale experiments can be reproduced using public seismic fault segmentation datasets.

Thebe seismic data:

```text
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YBYGBK
```

FaultSeg3D:

```text
https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8
```

## Included Sample Data

The included sample data are stored under:

```text
data/faultseg3d_sample/
  train/images/sample_000.npz
  train/labels/sample_000.npz
  val/images/sample_100.npz
  val/labels/sample_100.npz
```

These files are processed from the FaultSeg3D benchmark and are intended for code verification, debugging, and demonstration. They are not a substitute for the full dataset used in the paper.

## Data Layout

The training script expects paired volumes. Each image volume and label volume should use the same file name under separate directories.

```text
dataset/
  train/images/sample_000.npy
  train/labels/sample_000.npy
  val/images/sample_100.npy
  val/labels/sample_100.npy
```

Supported file formats are `.dat`, `.npy`, and `.npz`. Label values are expected to be integer class indices.

## Direct Training Check

```bash
python scripts/train.py \
  --train-images data/faultseg3d_sample/train/images \
  --train-labels data/faultseg3d_sample/train/labels \
  --val-images data/faultseg3d_sample/val/images \
  --val-labels data/faultseg3d_sample/val/labels \
  --epochs 1 \
  --batch-size 1 \
  --workers 0 \
  --output-dir outputs/sample_run
```

The command runs one training epoch and one validation pass on the included processed sample. The `outputs/` directory is ignored by Git.

## Full Dataset Training

After downloading and preprocessing the full dataset into the same paired directory layout, replace the four data paths in the command above with the full training and validation directories. For raw `.dat` volumes, the default volume shape is `128 128 128`.

## License

This project is released under the MIT License.
