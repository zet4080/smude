"""Model loading utilities to support both legacy checkpoint and modern pure PyTorch weights.

Usage during transition:
- Prefer loading "model_weights.pth" (pure PyTorch) if present.
- Fallback to "model.ckpt" using the legacy Lightning loader if still needed.

After upgrading dependencies, remove the legacy fallback to avoid requiring old Lightning.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch

from .model import UNet


DEFAULT_CONFIG = {
    "num_classes": 4,
    "num_layers": 5,
    "features_start": 64,
    "bilinear": True,
}


def find_model_artifacts(root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Return paths to (weights_path, checkpoint_path) under smude/ if found."""
    weights = root / "smude" / "model_weights.pth"
    ckpt = root / "smude" / "model.ckpt"
    return (weights if weights.exists() else None, ckpt if ckpt.exists() else None)


def load_unet(config: Optional[dict] = None) -> UNet:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    return UNet(**cfg)


def load_model_flexible(
    weights_path: Optional[str | os.PathLike] = None,
    checkpoint_path: Optional[str | os.PathLike] = None,
    config: Optional[dict] = None,
) -> torch.nn.Module:
    """Load the UNet using either pure weights or legacy checkpoint.

    Order of precedence:
    1) weights_path (pure PyTorch)
    2) checkpoint_path (legacy Lightning) if available and legacy support present
    3) auto-discover under package directory
    """
    root = Path(__file__).resolve().parents[1]

    if not weights_path and not checkpoint_path:
        w_auto, c_auto = find_model_artifacts(root)
        weights_path = str(w_auto) if w_auto else None
        checkpoint_path = str(c_auto) if c_auto else None

    model = load_unet(config)

    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # Non-fatal: still proceed, but surface info to the caller
            print("[loader] Warning: key mismatches when loading weights")
            if missing:
                print("  Missing keys:", missing)
            if unexpected:
                print("  Unexpected keys:", unexpected)
        return model

    if checkpoint_path and Path(checkpoint_path).exists():
        # Lazy import to avoid requiring Lightning unless absolutely necessary
        try:
            from .model import SegModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Legacy checkpoint found but legacy Lightning loader is unavailable. "
                "Run scripts/extract_model_weights.py with the old environment first."
            ) from e
        legacy = SegModel.load_from_checkpoint(str(checkpoint_path))  # type: ignore
        model.load_state_dict(legacy.net.state_dict())
        return model

    raise FileNotFoundError(
        "No model artifacts found. Expected 'smude/model_weights.pth' or 'smude/model.ckpt'."
    )
