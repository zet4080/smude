"""
Extract PyTorch weights from the legacy PyTorch Lightning 0.9.0 checkpoint (model.ckpt)
into a pure PyTorch state_dict so it can be loaded without Lightning.

Lightweight path (default):
- Uses torch.load to read the checkpoint dict and pulls out the 'state_dict'
- Strips the legacy 'net.' prefix used by the Lightning module
- Saves pure UNet weights to smude/model_weights.pth

Fallback path (optional):
- If direct parsing fails, it will attempt the legacy loader via smude.model.load_model,
  which requires the old PyTorch Lightning environment.

Outputs:
- smude/model_weights.pth (pure PyTorch weights of the UNet)
- smude/model_config.pth  (architecture config)

Run this BEFORE dependency upgrades so you keep a portable copy of weights.
"""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

import torch

# Add project root to path so `smude` imports resolve when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UNET_IMPORTABLE = False  # will try to import lazily only for verification


def main():
    ckpt_path = ROOT / "smude" / "model.ckpt"
    if not ckpt_path.exists():
        url = (
            "https://github.com/sonovice/smude/releases/download/v0.1.0/model.ckpt"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint not found. Downloading from:\n{url}\n→ {ckpt_path}")
        try:
            urlretrieve(url, str(ckpt_path))
        except Exception as e:
            print("Download failed. You can also download manually and place it at the path above.")
            raise

    print(f"Reading checkpoint (no Lightning required): {ckpt_path}")
    # Try a safe load first; if it fails due to disallowed globals, retry with a stubbed PL module
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    except Exception as e:
        # Handle PyTorch 2.6+ 'weights_only' default path and missing Lightning types
        import types

        print("Initial load failed; retrying with stubbed PyTorch Lightning and weights_only=False…")
        # Create stub modules so pickle can resolve references
        pl_name = "pytorch_lightning"
        utils_name = "pytorch_lightning.utilities"
        parsing_name = "pytorch_lightning.utilities.parsing"
        if pl_name not in sys.modules:
            sys.modules[pl_name] = types.ModuleType(pl_name)
        if utils_name not in sys.modules:
            sys.modules[utils_name] = types.ModuleType(utils_name)
        if parsing_name not in sys.modules:
            parsing_mod = types.ModuleType(parsing_name)
            # Minimal stand-in for AttributeDict used in older Lightning
            class AttributeDict(dict):
                pass

            parsing_mod.AttributeDict = AttributeDict
            sys.modules[parsing_name] = parsing_mod
        # Retry with full unpickle (trusted source assumption)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Lightning checkpoints store a dict with a 'state_dict' key
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip the leading module prefix used by the LightningModule, typically 'net.'
    # If keys look like 'net.xxx', drop the 'net.'; otherwise try to auto-detect a
    # single leading prefix segment and remove it if it unwraps to UNet-compatible keys.
    def strip_prefix(d: dict, prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}

    cleaned = strip_prefix(state_dict, "net.")

    if not cleaned:
        # Try a more generic approach: remove the first segment and a dot
        # e.g., 'model.net.layers.0...' -> 'net.layers.0...'
        tentative = {}
        for k, v in state_dict.items():
            if "." in k:
                tentative[k.split(".", 1)[1]] = v
        # If that produced 'net.'-prefixed keys, strip again
        cleaned = strip_prefix(tentative, "net.") or tentative

    if not cleaned:
        # Fallback to legacy Lightning loader if available (lazy import)
        print("Direct parse failed; attempting legacy Lightning loader…")
        try:
            from smude.model import load_model  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Could not parse checkpoint state_dict keys and legacy loader is unavailable.\n"
                "Please install a temporary environment with old PyTorch Lightning to proceed."
            ) from e
        legacy_model = load_model(str(ckpt_path))  # type: ignore
        cleaned = legacy_model.net.state_dict()

    weights_path = ROOT / "smude" / "model_weights.pth"
    torch.save(cleaned, str(weights_path))
    print(f"Saved pure PyTorch weights to: {weights_path}")

    # Save the model architecture config we know from the source
    config = {
        "num_classes": 4,
        "num_layers": 5,
        "features_start": 64,
        "bilinear": True,
    }
    config_path = ROOT / "smude" / "model_config.pth"
    torch.save(config, str(config_path))
    print(f"Saved model config to: {config_path}")

    # Quick self-check: attempt to load into a raw UNet
    # Optional verification: try importing UNet; skip if unavailable due to Lightning dep
    try:
        from smude.model import UNet  # noqa: WPS433

        print("Verifying weights load into raw UNet…")
        unet = UNet(**config)
        missing, unexpected = unet.load_state_dict(
            torch.load(weights_path, map_location="cpu"), strict=False
        )
        if missing or unexpected:
            print("Warning: Key mismatches when loading into UNet:")
            if missing:
                print("  Missing keys:", missing)
            if unexpected:
                print("  Unexpected keys:", unexpected)
        else:
            print("Weights loaded into UNet successfully.")
    except Exception:
        print(
            "Skipped UNet verification (could not import smude.model). The weights were still saved."
        )

    print("Done. You can now upgrade dependencies and load with pure PyTorch.")


if __name__ == "__main__":
    main()
