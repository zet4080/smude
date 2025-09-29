"""
Loader-only smoke test that bypasses importing the smude package (and thus avoids
executing smude/__init__.py). It:
- Dynamically loads UNet from smude/model.py via importlib
- Loads pure PyTorch weights from smude/model_weights.pth
- Runs a forward pass on a dummy tensor and prints the output shape
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_PY = ROOT / "smude" / "model.py"
WEIGHTS = ROOT / "smude" / "model_weights.pth"
CONFIG = ROOT / "smude" / "model_config.pth"


def load_unet_class():
    spec = importlib.util.spec_from_file_location("smude_model", str(MODEL_PY))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load smude/model.py module spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.UNet  # type: ignore[attr-defined]


def main() -> None:
    assert WEIGHTS.exists(), f"Weights not found: {WEIGHTS}"
    assert CONFIG.exists(), f"Config not found: {CONFIG}"

    UNet = load_unet_class()
    cfg = torch.load(CONFIG, map_location="cpu")

    print("Instantiating UNet with config:", cfg)
    model = UNet(**cfg)
    state = torch.load(WEIGHTS, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("Warning: Key mismatches while loading state dict")
        if missing:
            print("  Missing:", missing)
        if unexpected:
            print("  Unexpected:", unexpected)

    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)

    print("Output shape:", tuple(y.shape))
    assert y.ndim == 4, f"Unexpected ndim: {y.ndim}"
    assert y.shape[1] == 4, f"Expected 4 channels, got {y.shape[1]}"
    print("Loader-only smoke test passed.")


if __name__ == "__main__":
    main()
