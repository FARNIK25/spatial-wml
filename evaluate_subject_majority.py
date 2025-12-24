"""Subject-level evaluation with slice-wise inference and majority voting."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model

PROJECT_ROOT = Path(__file__).resolve().parent
CUSTOM_DATASET_ROOT = PROJECT_ROOT / "CustomDataset"
if str(CUSTOM_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(CUSTOM_DATASET_ROOT))

import MedViT  # noqa: F401  # Registers custom architectures with timm

DEFAULT_SUBJECT_SPLIT_ROOTS = [
    PROJECT_ROOT / "data" / "SBStrokeCT_splits_subject",
    PROJECT_ROOT / "data" / "SBStrokeCT_sculpt_splits_subject",
]


@dataclass
class SubjectStats:
    label_idx: int
    label_name: str
    vote_counts: List[int]
    prob_sums: List[float]
    slice_count: int = 0

    def register_prediction(self, pred_idx: int, probs: Sequence[float]) -> None:
        self.slice_count += 1
        self.vote_counts[pred_idx] += 1
        for idx, value in enumerate(probs):
            self.prob_sums[idx] += float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MedViT checkpoints with subject-level majority voting")
    parser.add_argument("--run-dir", type=Path, help="Directory containing checkpoint_best.pth and log.txt")
    parser.add_argument("--checkpoint", type=Path, help="Explicit checkpoint path (overrides --run-dir)")
    parser.add_argument("--model", type=str, default="MedViT_large", help="Model architecture to instantiate")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Dataset root containing split subfolders (train/val/test). If omitted the script tries "
            "common locations under data/SBStrokeCT_splits_subject and data/SBStrokeCT_sculpt_splits_subject."
        ),
    )
    parser.add_argument("--split", type=str, default="val", help="Split name under data-root (e.g., val or test)")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--input-size", type=int, default=224, help="Model input resolution")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis_outputs" / "subject_evaluation",
        help="Directory where CSV/JSON artifacts will be written",
    )
    return parser.parse_args()


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        path = self.samples[index][0]
        return sample, target, path


def resolve_eval_split(args: argparse.Namespace) -> Tuple[Path, Path]:
    candidates: List[Path] = []
    if args.data_root is not None:
        candidates.append(Path(args.data_root).expanduser())
    for default_root in DEFAULT_SUBJECT_SPLIT_ROOTS:
        expanded = default_root.expanduser()
        if expanded not in candidates:
            candidates.append(expanded)

    tried_paths: List[Path] = []
    for root in candidates:
        eval_dir = (root / args.split).expanduser()
        tried_paths.append(eval_dir)
        if eval_dir.exists():
            return root.resolve(), eval_dir.resolve()

    tried_render = "\n".join(f" - {path}" for path in tried_paths)
    raise FileNotFoundError(
        "Evaluation split '{split}' not found. Checked the following locations:\n{paths}".format(
            split=args.split,
            paths=tried_render if tried_render else "(no candidate directories found)",
        )
    )


def build_loader(args: argparse.Namespace) -> Tuple[ImageFolderWithPaths, DataLoader, List[str]]:
    resolved_root, eval_dir = resolve_eval_split(args)
    args.resolved_data_root = resolved_root

    transform = transforms.Compose([
        transforms.Resize(int((256 / 224) * args.input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    dataset = ImageFolderWithPaths(eval_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    class_names = [name for name, _ in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])]
    return dataset, loader, class_names


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        return checkpoint_path
    if not args.run_dir:
        raise ValueError("Either --checkpoint or --run-dir must be provided")
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")
    for candidate in ("checkpoint_best.pth", "checkpoint.pth"):
        path = run_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint_best.pth or checkpoint.pth found inside {run_dir}")


def load_model(model_name: str, checkpoint_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = create_model(model_name, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint: {unexpected}")
    model.to(device)
    model.eval()
    return model


def extract_subject_and_slice(path: Path) -> Tuple[str, int]:
    stem = path.stem
    tokens = stem.split("_")
    if len(tokens) >= 2 and tokens[0].lower().startswith("sbstroke"):
        subject_id = "_".join(tokens[:2])
    else:
        subject_id = tokens[0]
    slice_token = tokens[-1]
    digits = "".join(ch for ch in slice_token if ch.isdigit())
    try:
        slice_idx = int(digits)
    except ValueError:
        slice_idx = -1
    return subject_id, slice_idx


def ensure_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_csv(rows: List[Dict[str, object]], fieldnames: List[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset, loader, class_names = build_loader(args)
    checkpoint_path = resolve_checkpoint(args)
    model = load_model(args.model, checkpoint_path, num_classes=len(class_names), device=device)

    slice_logs: List[Dict[str, object]] = []
    subject_stats: Dict[str, SubjectStats] = {}

    with torch.no_grad():
        for images, targets, paths in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            pred_confidences, pred_indices = torch.max(probabilities, dim=1)

            for idx in range(len(paths)):
                path = Path(paths[idx])
                subject_id, slice_idx = extract_subject_and_slice(path)
                target_idx = int(targets[idx].item())
                pred_idx = int(pred_indices[idx].item())
                probs = probabilities[idx].cpu().tolist()
                confidence = float(pred_confidences[idx].item())

                if subject_id not in subject_stats:
                    subject_stats[subject_id] = SubjectStats(
                        label_idx=target_idx,
                        label_name=class_names[target_idx],
                        vote_counts=[0] * len(class_names),
                        prob_sums=[0.0] * len(class_names),
                    )
                else:
                    existing_label = subject_stats[subject_id].label_idx
                    if existing_label != target_idx:
                        raise ValueError(
                            f"Subject {subject_id} has inconsistent labels ({existing_label} vs {target_idx})."
                        )

                subject_stats[subject_id].register_prediction(pred_idx, probs)

                record = {
                    "subject_id": subject_id,
                    "slice_index": slice_idx,
                    "file_name": path.name,
                    "file_path": str(path),
                    "true_label": class_names[target_idx],
                    "pred_label": class_names[pred_idx],
                    "pred_confidence": confidence,
                    "correct": class_names[pred_idx] == class_names[target_idx],
                }
                for class_idx, class_name in enumerate(class_names):
                    record[f"prob_{class_name}"] = probs[class_idx]
                slice_logs.append(record)

    subject_rows: List[Dict[str, object]] = []
    confusion = defaultdict(lambda: defaultdict(int))
    correct_subjects = 0

    for subject_id, stats in subject_stats.items():
        max_votes = max(stats.vote_counts)
        tied_indices = [idx for idx, votes in enumerate(stats.vote_counts) if votes == max_votes]
        if len(tied_indices) == 1:
            winner_idx = tied_indices[0]
        else:
            winner_idx = max(tied_indices, key=lambda idx: stats.prob_sums[idx])

        true_label = class_names[stats.label_idx]
        pred_label = class_names[winner_idx]
        confusion[true_label][pred_label] += 1
        is_correct = true_label == pred_label
        if is_correct:
            correct_subjects += 1

        row = {
            "subject_id": subject_id,
            "num_slices": stats.slice_count,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": is_correct,
        }
        for class_idx, class_name in enumerate(class_names):
            row[f"votes_{class_name}"] = stats.vote_counts[class_idx]
            row[f"mean_prob_{class_name}"] = (
                stats.prob_sums[class_idx] / stats.slice_count if stats.slice_count else 0.0
            )
        subject_rows.append(row)

    subject_accuracy = correct_subjects / max(len(subject_rows), 1)

    output_dir = ensure_output_dir(args.output_dir)
    slice_csv = output_dir / "slice_predictions.csv"
    subject_csv = output_dir / "subject_majority.csv"
    summary_json = output_dir / "subject_metrics.json"

    slice_fieldnames = list(slice_logs[0].keys()) if slice_logs else []
    subject_fieldnames = list(subject_rows[0].keys()) if subject_rows else []

    if slice_fieldnames:
        save_csv(slice_logs, slice_fieldnames, slice_csv)
    if subject_fieldnames:
        save_csv(subject_rows, subject_fieldnames, subject_csv)

    summary_payload = {
        "checkpoint": str(checkpoint_path),
        "model": args.model,
        "split": args.split,
        "num_subjects": len(subject_rows),
        "correct_subjects": correct_subjects,
        "subject_accuracy": subject_accuracy,
        "class_names": class_names,
        "confusion": {true: dict(preds) for true, preds in confusion.items()},
    }

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(
        f"Subject-level accuracy: {subject_accuracy:.4f} "
        f"({correct_subjects}/{len(subject_rows)})"
    )
    print(f"Slice-level log saved to {slice_csv}")
    print(f"Subject-level log saved to {subject_csv}")
    print(f"Metrics JSON saved to {summary_json}")


if __name__ == "__main__":
    main()
