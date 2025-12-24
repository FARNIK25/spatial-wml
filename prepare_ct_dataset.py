"""Utility to create MedViT-compatible splits for the SBStrokeCT_classifier dataset.

The source directory is expected to contain class folders (for example, ``Strok`` and
``NonStrock``). The script copies or links images into ``train``/``val``/``test``
subdirectories that MedViT's ``image_folder`` dataset loader can consume. Splits can be
generated either slice-by-slice or by grouping entire subjects/volumes together.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DEFAULT_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
}

SUBJECT_PATTERN = re.compile(r"^(?P<subject>.+?_CT)_s\d+.*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for SBStrokeCT_classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Directory containing class folders (e.g., Strok/ and NonStrock/).",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Destination directory that will receive train/val[/test] structure.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of images per class reserved for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Optional fraction of images per class reserved for a held-out test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling images before splitting.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="How to materialize files in the output directory.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=sorted(DEFAULT_IMAGE_EXTENSIONS),
        help="Image extensions (case-insensitive) to include when scanning the source tree.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the planned split sizes without creating any files.",
    )
    parser.add_argument(
        "--group-mode",
        choices=["slice", "subject"],
        default="slice",
        help="Decide whether to split files individually (slice) or keep full subjects together.",
    )
    return parser.parse_args()


def collect_class_folders(root: Path) -> List[Path]:
    class_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found in {root}")
    return class_dirs


def list_images(folder: Path, extensions: Iterable[str]) -> List[Path]:
    lower_exts = {ext.lower() for ext in extensions}
    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in lower_exts
    ]


def split_counts(total: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in the range [0, 1)")
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must be in the range [0, 1)")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    val_count = int(round(total * val_ratio))
    test_count = int(round(total * test_ratio))
    train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError(
            f"Computed train_count={train_count}. Reduce val/test ratios for class with {total} items."
        )
    # Adjust counts if rounding produced an over-allocation.
    while train_count + val_count + test_count > total:
        if val_count > test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            train_count -= 1
    return train_count, val_count, test_count


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def materialize(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def relative_to(src: Path, start: Path) -> Path:
    try:
        return src.relative_to(start)
    except ValueError:
        return Path(src.name)


def extract_subject_id(file_path: Path) -> str:
    stem = file_path.stem
    match = SUBJECT_PATTERN.match(stem)
    if match:
        return match.group("subject")
    if "_s" in stem:
        return stem.split("_s", 1)[0]
    return stem


def group_by_subject(files: Sequence[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for file_path in files:
        subject_id = extract_subject_id(file_path)
        groups.setdefault(subject_id, []).append(file_path)
    return groups


def process_class(
    class_dir: Path,
    output_root: Path,
    extensions: Iterable[str],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    mode: str,
    dry_run: bool,
    group_mode: str,
) -> None:
    files = list_images(class_dir, extensions)
    if not files:
        raise ValueError(f"No images found in {class_dir}")

    rng = random.Random(seed)

    if group_mode == "slice":
        rng.shuffle(files)
        train_count, val_count, test_count = split_counts(
            len(files), val_ratio, test_ratio
        )
        splits = [
            ("train", files[:train_count]),
            ("val", files[train_count : train_count + val_count]),
            (
                "test",
                files[
                    train_count
                    + val_count : train_count
                    + val_count
                    + test_count
                ],
            ),
        ]
        print(
            f"Class {class_dir.name}: total_slices={len(files)} train={train_count} val={val_count} test={test_count}"
        )
    else:
        subject_groups = group_by_subject(files)
        subject_ids = list(subject_groups)
        rng.shuffle(subject_ids)
        train_subj, val_subj, test_subj = split_counts(
            len(subject_ids), val_ratio, test_ratio
        )
        split_subjects = {
            "train": subject_ids[:train_subj],
            "val": subject_ids[train_subj : train_subj + val_subj],
            "test": subject_ids[train_subj + val_subj : train_subj + val_subj + test_subj],
        }
        splits = []
        for split_name, subject_list in split_subjects.items():
            if not subject_list:
                splits.append((split_name, []))
                continue
            split_files = [
                file_path
                for subject_id in subject_list
                for file_path in subject_groups[subject_id]
            ]
            splits.append((split_name, split_files))
        slice_counts = {name: len(items) for name, items in splits}
        print(
            "Class {name}: subjects={subjects} train={train_subj} ({train_slices} slices) "
            "val={val_subj} ({val_slices} slices) test={test_subj} ({test_slices} slices)".format(
                name=class_dir.name,
                subjects=len(subject_ids),
                train_subj=train_subj,
                train_slices=slice_counts["train"],
                val_subj=val_subj,
                val_slices=slice_counts["val"],
                test_subj=test_subj,
                test_slices=slice_counts["test"],
            )
        )

    if dry_run:
        return

    for split_name, split_files in splits:
        if not split_files:
            continue
        split_dir = output_root / split_name / class_dir.name
        ensure_dir(split_dir)
        for file_path in split_files:
            rel_path = relative_to(file_path, class_dir)
            destination = split_dir / rel_path
            ensure_dir(destination.parent)
            materialize(file_path, destination, mode)


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
    ensure_dir(output)

    class_dirs = collect_class_folders(source)
    for class_dir in class_dirs:
        process_class(
            class_dir,
            output,
            args.extensions,
            args.seed,
            args.val_ratio,
            args.test_ratio,
            args.link_mode,
            args.dry_run,
            args.group_mode,
        )

    if args.dry_run:
        print("Dry run complete. No files were created.")
    else:
        print(f"Prepared dataset at {output}")


if __name__ == "__main__":
    main()
