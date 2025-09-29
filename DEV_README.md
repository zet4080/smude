# Dev setup (modernized)

## Quick start

- Python 3.11 or 3.12
- Install package (editable):

```powershell
python -m pip install -e .[dev]
```

- Install PyTorch/torchvision suitable for your platform (CPU-only example):

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install torchvision --index-url https://download.pytorch.org/whl/cpu
```

- Optional (training):

```powershell
python -m pip install .[train]
```

## Useful scripts

- Convert model checkpoint to pure weights (already done in repo):
```powershell
python .\scripts\extract_model_weights.py
```

- Loader-only smoke test:
```powershell
python .\scripts\smoke_loader_only.py
```

- Package smoke loader (requires torchvision):
```powershell
python .\scripts\smoke_test.py
```

## Notes

- The app uses pure PyTorch for inference; Lightning is optional and recommended for training.
- Runtime-heavy deps (torch/torchvision/opencv) are intentionally not tightly pinned in pyproject to allow platform-specific wheel selection.
