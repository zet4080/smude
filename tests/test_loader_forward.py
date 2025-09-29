from pathlib import Path
import importlib.util
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_PY = ROOT / "smude" / "model.py"
WEIGHTS = ROOT / "smude" / "model_weights.pth"
CONFIG = ROOT / "smude" / "model_config.pth"


def _load_unet_class():
    spec = importlib.util.spec_from_file_location("smude_model", str(MODEL_PY))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.UNet  # type: ignore[attr-defined]


def test_forward_shape():
    assert WEIGHTS.exists() and CONFIG.exists()
    UNet = _load_unet_class()
    cfg = torch.load(CONFIG, map_location="cpu")
    model = UNet(**cfg)
    state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape[:2] == (1, 4)
