"""Convert SBStrokeCT NIfTI volumes into 2D PNG slices per class.

Each source class folder (for example ``Stroke`` or ``NoStroke``) is scanned for NIfTI
files (``.nii`` or ``.nii.gz``). The script normalizes voxel intensities using configurable
percentile clipping, extracts axial slices, optionally skips blank slices, converts them into
3-channel PNG images, and writes them into a class-matching directory under the output root.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

VOLUME_EXTENSIONS = {".nii", ".nii.gz"}
HU_WINDOW_PRESETS = {
    "brain": (40.0, 80.0),
    "subdural": (50.0, 130.0),
    "stroke": (35.0, 40.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NIfTI CT volumes into PNG slices for classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="Directory containing class subfolders.")
    parser.add_argument(
        "output",
        type=Path,
        help="Destination directory where class subfolders of PNG images will be created.",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        help="Axis index (0, 1, or 2) representing the slice dimension in the volume.",
    )
    parser.add_argument(
        "--slice-step",
        type=int,
        default=1,
        help="Take every Nth slice to reduce dataset size.",
    )
    parser.add_argument(
        "--min-std",
        type=float,
        default=5.0,
        help="Skip slices whose standard deviation is below this threshold (filters blank slices).",
    )
    parser.add_argument(
        "--percentile-range",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Lower and upper percentiles for intensity clipping before scaling to 8-bit.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional spatial resize applied to every output PNG.",
    )
    parser.add_argument(
        "--hu-window",
        type=float,
        nargs=2,
        metavar=("CENTER", "WIDTH"),
        default=None,
        help="Clip intensities using an explicit HU window defined by center and width.",
    )
    parser.add_argument(
        "--hu-window-preset",
        choices=sorted(HU_WINDOW_PRESETS.keys()),
        default=None,
        help="Use a predefined HU window (brain, subdural, stroke) instead of percentiles.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many slices would be written without producing files.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_volumes(class_dir: Path) -> Iterable[Path]:
    for extension in VOLUME_EXTENSIONS:
        yield from class_dir.glob(f"*{extension}")


def load_volume(path: Path) -> np.ndarray:
    volume = nib.load(str(path))
    data = volume.get_fdata(dtype=np.float32)
    return np.nan_to_num(data)


def normalize_slice(slice_data: np.ndarray, low: float, high: float) -> np.ndarray:
    slice_clipped = np.clip(slice_data, low, high)
    denominator = high - low
    if denominator <= 0:
        return np.zeros_like(slice_clipped, dtype=np.uint8)
    scaled = (slice_clipped - low) / denominator
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def iter_slices(volume: np.ndarray, axis: int, step: int) -> Iterable[Tuple[int, np.ndarray]]:
    axis = axis % volume.ndim
    volume = np.moveaxis(volume, axis, -1)
    total = volume.shape[-1]
    for index in range(0, total, step):
        yield index, volume[..., index]


def volume_stem(path: Path) -> str:
    name = path.name
    for ext in sorted(VOLUME_EXTENSIONS, key=len, reverse=True):
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def resolve_hu_window(args_window: Tuple[float, float] | None, preset: str | None) -> Tuple[float, float] | None:
    if preset:
        return HU_WINDOW_PRESETS[preset]
    return args_window


def process_volume(
    volume_path: Path,
    output_dir: Path,
    axis: int,
    slice_step: int,
    min_std: float,
    percentile_low: float,
    percentile_high: float,
    hu_window: Tuple[float, float] | None,
    resize: Tuple[int, int] | None,
    dry_run: bool,
) -> int:
    data = load_volume(volume_path)
    if hu_window is not None:
        center, width = hu_window
        half_width = max(width / 2.0, 1e-3)
        low = center - half_width
        high = center + half_width
    else:
        low = np.percentile(data, percentile_low)
        high = np.percentile(data, percentile_high)
    if high <= low:
        high = low + 1e-3
    base_name = volume_stem(volume_path)
    written = 0

    for slice_index, slice_data in iter_slices(data, axis=axis, step=slice_step):
        if float(np.std(slice_data)) < min_std:
            continue
        slice_uint8 = normalize_slice(slice_data, low, high)
        slice_rgb = np.stack([slice_uint8] * 3, axis=-1)
        image = Image.fromarray(slice_rgb)
        if resize is not None:
            image = image.resize(resize, resample=Image.BILINEAR)
        filename = f"{base_name}_s{slice_index:03d}.png"
        if not dry_run:
            image.save(output_dir / filename)
        written += 1
    return written


def process_class(
    class_dir: Path,
    output_root: Path,
    axis: int,
    slice_step: int,
    min_std: float,
    percentile_low: float,
    percentile_high: float,
    hu_window: Tuple[float, float] | None,
    resize: Tuple[int, int] | None,
    dry_run: bool,
) -> Tuple[int, int]:
    volumes = list(find_volumes(class_dir))
    if not volumes:
        raise ValueError(f"No NIfTI files found in {class_dir}")

    target_dir = output_root / class_dir.name
    if not dry_run:
        ensure_dir(target_dir)

    slice_total = 0
    for volume_path in volumes:
        count = process_volume(
            volume_path,
            target_dir,
            axis,
            slice_step,
            min_std,
            percentile_low,
            percentile_high,
            hu_window,
            resize,
            dry_run,
        )
        slice_total += count
        print(f"{class_dir.name}/{volume_path.name}: {count} slices")
    return len(volumes), slice_total


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()

    hu_window = resolve_hu_window(args.hu_window, args.hu_window_preset)

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    if not args.dry_run:
        ensure_dir(output)

    class_dirs = [path for path in source.iterdir() if path.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class directories found in {source}")

    total_volumes = 0
    total_slices = 0
    for class_dir in class_dirs:
        volumes, slices = process_class(
            class_dir,
            output,
            args.axis,
            args.slice_step,
            args.min_std,
            args.percentile_range[0],
            args.percentile_range[1],
            hu_window,
            tuple(args.resize) if args.resize else None,
            args.dry_run,
        )
        total_volumes += volumes
        total_slices += slices
        print(f"Class {class_dir.name}: {volumes} volumes -> {slices} slices")

    summary = f"Converted {total_volumes} volumes into {total_slices} slices"
    if args.dry_run:
        summary += " (dry run, no files written)"
    print(summary)


if __name__ == "__main__":
    main()
