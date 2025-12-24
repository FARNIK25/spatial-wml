from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image

PNG_ROOT = Path("data/ SBStrokeCT_png".replace(" ", "").replace("//", "/"))
SUBJECT_SPLIT_ROOT = Path("data/SBStrokeCT_splits_subject")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUBJECT_PATTERN = re.compile(r"^(?P<subject>.+?_CT)_s\d+.*$", re.IGNORECASE)


def extract_subject_id(path: Path) -> str:
    stem = path.stem
    match = SUBJECT_PATTERN.match(stem)
    if match:
        return match.group("subject")
    if "_s" in stem:
        return stem.split("_s", 1)[0]
    return stem


def iter_images(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def update_running_stats(
    *,
    mean: np.ndarray,
    m2: np.ndarray,
    count: int,
    block_mean: np.ndarray,
    block_m2: np.ndarray,
    block_count: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if count == 0:
        return block_mean, block_m2, block_count
    total = count + block_count
    delta = block_mean - mean
    mean = mean + delta * (block_count / total)
    m2 = m2 + block_m2 + (delta * delta) * (count * block_count / total)
    return mean, m2, total


def analyze_png_dataset(root: Path) -> Dict[str, Dict[str, float]]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    results: Dict[str, Dict[str, float]] = {}
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        mean = np.zeros(3, dtype=np.float64)
        m2 = np.zeros(3, dtype=np.float64)
        count = 0
        files = list(iter_images(class_dir))
        if not files:
            raise ValueError(f"No images found under {class_dir}")
        pixel_min = math.inf
        pixel_max = -math.inf
        blank_slices = 0
        for path in files:
            with Image.open(path) as img:
                arr = np.array(img, dtype=np.float32)
            pixel_min = min(pixel_min, float(arr.min()))
            pixel_max = max(pixel_max, float(arr.max()))
            if float(arr.std()) < 1e-3:
                blank_slices += 1
            arr = arr.reshape(-1, arr.shape[-1]) / 255.0
            block_mean = arr.mean(axis=0)
            block_count = arr.shape[0]
            block_m2 = ((arr - block_mean) ** 2).sum(axis=0)
            mean, m2, count = update_running_stats(
                mean=mean,
                m2=m2,
                count=count,
                block_mean=block_mean,
                block_m2=block_m2,
                block_count=block_count,
            )
        if count < 2:
            raise ValueError(f"Insufficient pixels for stats in {class_dir}")
        variance = m2 / (count - 1)
        std = np.sqrt(variance)
        results[class_dir.name] = {
            "images": len(files),
            "pixel_min": pixel_min,
            "pixel_max": pixel_max,
            "channel_mean": mean.tolist(),
            "channel_std": std.tolist(),
            "blank_slices": blank_slices,
        }
    return results


def check_subject_splits(root: Path) -> Dict[str, Dict[str, int]]:
    if not root.exists():
        raise FileNotFoundError(f"Split root not found: {root}")
    split_counts: Dict[str, Dict[str, int]] = {}
    subject_presence: Dict[str, Counter] = {}
    for split_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        split_name = split_dir.name
        class_counts: Dict[str, int] = {}
        subjects_in_split: set[str] = set()
        for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            files = list(iter_images(class_dir))
            class_counts[class_dir.name] = len(files)
            for file_path in files:
                subject_id = extract_subject_id(file_path)
                subjects_in_split.add(subject_id)
                subject_presence.setdefault(subject_id, Counter())[split_name] += 1
        split_counts[split_name] = class_counts
        print(f"Split {split_name}: {sum(class_counts.values())} images across {len(class_counts)} classes")
    leaked_subjects = {subject: counter for subject, counter in subject_presence.items() if sum(1 for count in counter.values() if count > 0) > 1}
    if leaked_subjects:
        print("WARNING: subjects found in multiple splits")
        for subject, counter in leaked_subjects.items():
            print(f"  {subject}: {dict(counter)}")
    else:
        print("No subject leakage detected between splits.")
    return split_counts


def main() -> None:
    print("Analyzing PNG dataset stats...")
    png_stats = analyze_png_dataset(PNG_ROOT)
    for class_name, stats in png_stats.items():
        mean = ", ".join(f"{value:.4f}" for value in stats["channel_mean"])
        std = ", ".join(f"{value:.4f}" for value in stats["channel_std"])
        print(
            f"Class {class_name}: images={stats['images']}, blank_slices={stats['blank_slices']}, "
            f"pixel_range=[{stats['pixel_min']:.1f}, {stats['pixel_max']:.1f}], "
            f"mean=({mean}), std=({std})"
        )
    print("\nChecking subject-based splits...")
    split_counts = check_subject_splits(SUBJECT_SPLIT_ROOT)
    for split_name, counts in split_counts.items():
        details = ", ".join(f"{cls}:{count}" for cls, count in sorted(counts.items()))
        print(f"  {split_name}: {details}")


if __name__ == "__main__":
    main()
