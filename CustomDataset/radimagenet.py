"""Utilities for working with RadImageNet pretrained checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from timm.models import create_model


@dataclass(frozen=True)
class RadImageNetSpec:
    """Metadata describing a RadImageNet checkpoint."""

    arch: str
    filename: str
    url: Optional[str] = None
    input_size: int = 224


# Known checkpoint naming conventions from the official RadImageNet release.
RADIMAGENET_SPECS: Dict[str, RadImageNetSpec] = {
    "densenet121": RadImageNetSpec("densenet121", "radimagenet_densenet121.pth"),
    "efficientnet_b0": RadImageNetSpec("efficientnet_b0", "radimagenet_efficientnet_b0.pth", input_size=224),
    "efficientnet_b1": RadImageNetSpec("efficientnet_b1", "radimagenet_efficientnet_b1.pth", input_size=240),
    "efficientnet_b2": RadImageNetSpec("efficientnet_b2", "radimagenet_efficientnet_b2.pth", input_size=260),
    "efficientnet_b3": RadImageNetSpec("efficientnet_b3", "radimagenet_efficientnet_b3.pth", input_size=300),
    "efficientnet_b4": RadImageNetSpec("efficientnet_b4", "radimagenet_efficientnet_b4.pth", input_size=380),
    "efficientnet_b5": RadImageNetSpec("efficientnet_b5", "radimagenet_efficientnet_b5.pth", input_size=456),
    "inception_resnet_v2": RadImageNetSpec("inception_resnet_v2", "radimagenet_inception_resnet_v2.pth"),
    "inception_v3": RadImageNetSpec("inception_v3", "radimagenet_inception_v3.pth"),
    "resnet50": RadImageNetSpec("resnet50", "radimagenet_resnet50.pth"),
    "resnet101": RadImageNetSpec("resnet101", "radimagenet_resnet101.pth"),
    "resnet152": RadImageNetSpec("resnet152", "radimagenet_resnet152.pth"),
    "seresnext50_32x4d": RadImageNetSpec("seresnext50_32x4d", "radimagenet_seresnext50_32x4d.pth"),
    "seresnext101_32x4d": RadImageNetSpec("seresnext101_32x4d", "radimagenet_seresnext101_32x4d.pth"),
    "wide_resnet50_2": RadImageNetSpec("wide_resnet50_2", "radimagenet_wide_resnet50_2.pth"),
}

_PREFIXES = ("radimagenet_", "radimagenet.")


def _parse_radimagenet_arch(model_name: str) -> Optional[str]:
    canonical = model_name.lower().replace("-", "_")
    for prefix in _PREFIXES:
        if canonical.startswith(prefix):
            arch = canonical[len(prefix) :]
            if arch:
                return arch
    return None


def is_radimagenet_model(model_name: str) -> bool:
    arch = _parse_radimagenet_arch(model_name)
    return arch in RADIMAGENET_SPECS if arch else False


def _cleanup_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned[new_key] = value
    return cleaned


def _filter_for_model(
    state_dict: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
    skip_classifier: bool = True,
) -> Dict[str, torch.Tensor]:
    filtered: Dict[str, torch.Tensor] = {}
    for key, weight in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if skip_classifier and ("classifier" in key or key.endswith(".fc.weight") or key.endswith(".fc.bias")):
            continue
        if target.shape != weight.shape:
            continue
        filtered[key] = weight
    return filtered


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "net", "module_state_dict"):
            nested = raw.get(key)
            if isinstance(nested, dict):
                return _cleanup_state_dict(nested)
    if isinstance(raw, dict):
        return _cleanup_state_dict(raw)
    raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}")


def _resolve_checkpoint(
    spec: RadImageNetSpec,
    weights_root: Optional[Path],
    checkpoint_override: Optional[Path],
    auto_download: bool,
) -> Optional[Path]:
    if checkpoint_override:
        resolved = checkpoint_override.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"RadImageNet checkpoint not found at {resolved}")
        return resolved

    if weights_root is None:
        return None

    weights_root = weights_root.expanduser().resolve()
    if not weights_root.exists():
        if auto_download and spec.url:
            weights_root.mkdir(parents=True, exist_ok=True)
        else:
            return None

    candidates = [
        weights_root / spec.filename,
        weights_root / f"{spec.arch}.pth",
        weights_root / f"{spec.arch}.pt",
        weights_root / f"radimagenet_{spec.arch}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if auto_download and spec.url:
        weights_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = weights_root / spec.filename
        state_dict = torch.hub.load_state_dict_from_url(spec.url, map_location="cpu")
        torch.save(state_dict, checkpoint_path)
        return checkpoint_path

    return None


def create_radimagenet_model(
    model_name: str,
    num_classes: int,
    *,
    weights_root: Optional[Path] = None,
    checkpoint_override: Optional[Path] = None,
    auto_download: bool = False,
) -> torch.nn.Module:
    arch = _parse_radimagenet_arch(model_name)
    if arch is None:
        raise ValueError(f"Model name {model_name} is not a RadImageNet variant")
    if arch not in RADIMAGENET_SPECS:
        supported = ", ".join(sorted(RADIMAGENET_SPECS))
        raise ValueError(f"Unsupported RadImageNet backbone '{arch}'. Supported: {supported}")

    spec = RADIMAGENET_SPECS[arch]
    model = create_model(spec.arch, pretrained=False, num_classes=num_classes)

    checkpoint_path = _resolve_checkpoint(spec, weights_root, checkpoint_override, auto_download)
    if checkpoint_path is None:
        print(
            f"Warning: RadImageNet weights for '{arch}' were not located. Proceeding with random initialization."
        )
        return model

    state_dict = _load_checkpoint(checkpoint_path)
    filtered_state = _filter_for_model(state_dict, model.state_dict())
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing:
        print(f"Warning: missing keys when loading RadImageNet weights: {sorted(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys when loading RadImageNet weights: {sorted(unexpected)}")

    return model


def radimagenet_input_resolution(model_name: str) -> Optional[int]:
    arch = _parse_radimagenet_arch(model_name)
    if arch is None:
        return None
    spec = RADIMAGENET_SPECS.get(arch)
    if spec is None:
        return None
    return spec.input_size
