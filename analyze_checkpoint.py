"""Compute detailed evaluation metrics for MedViT checkpoints."""
import argparse
import sys
import types
from pathlib import Path

import torch
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model

PROJECT_ROOT = Path(__file__).resolve().parent
CUSTOM_DATASET_ROOT = PROJECT_ROOT / "CustomDataset"
if str(CUSTOM_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(CUSTOM_DATASET_ROOT))

import MedViT  # noqa: F401  # registers MedViT variants with timm

# Allow common custom objects stored in checkpoints (e.g., training args, pathlib paths) to load safely
_SAFE_GLOBALS = [argparse.Namespace, Path]
try:  # pragma: no cover - platform-specific
    from pathlib import WindowsPath

    _SAFE_GLOBALS.append(WindowsPath)
except ImportError:  # pragma: no cover
    WindowsPath = None

try:  # pragma: no cover - optional module
    from pathlib import _local as pathlib_local

    for attr in ("WindowsPath", "Path"):
        cls = getattr(pathlib_local, attr, None)
        if cls is not None:
            _SAFE_GLOBALS.append(cls)
except (ImportError, AttributeError):
    pathlib_local = None

add_safe_globals([cls for cls in _SAFE_GLOBALS if cls is not None])

# Provide a stub for pathlib._local so that unpickling checkpoints referencing it succeeds
if "pathlib._local" not in sys.modules:  # pragma: no cover
    stub = types.ModuleType("pathlib._local")
    stub.WindowsPath = Path
    stub.Path = Path
    sys.modules["pathlib._local"] = stub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with extended metrics")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the .pth checkpoint file")
    parser.add_argument("--data-root", type=Path, default=Path("data/SBStrokeCT_splits"), help="Root folder with class subdirectories")
    parser.add_argument("--split", type=str, default="val", help="Subdirectory under data-root to evaluate")
    parser.add_argument("--model", type=str, default="MedViT_large", help="Model name to instantiate")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of images per evaluation batch")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--device", type=str, default="cuda", help="Device identifier (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--input-size", type=int, default=224, help="Input image size")
    parser.add_argument("--save-json", type=Path, help="Optional path to save metrics as JSON")
    return parser.parse_args()


def build_loader(args: argparse.Namespace) -> tuple[datasets.ImageFolder, DataLoader]:
    eval_dir = (args.data_root / args.split).resolve()
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation split not found at {eval_dir}")

    transform = transforms.Compose([
        transforms.Resize(int((256 / 224) * args.input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    dataset = datasets.ImageFolder(eval_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return dataset, loader


def load_model(args: argparse.Namespace, num_classes: int) -> torch.nn.Module:
    device = torch.device(args.device)
    model = create_model(args.model, num_classes=num_classes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return model


def accumulate_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> torch.Tensor:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1
    return confusion


def compute_metrics(confusion: torch.Tensor) -> dict:
    tp = confusion.diag().to(torch.float64)
    support = confusion.sum(dim=1).to(torch.float64)
    pred_counts = confusion.sum(dim=0).to(torch.float64)
    fp = pred_counts - tp
    fn = support - tp

    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(tp))
    iou = torch.where(tp + fp + fn > 0, tp / (tp + fp + fn), torch.zeros_like(tp))

    accuracy = tp.sum() / confusion.sum().clamp(min=1)

    metrics = {
        "confusion_matrix": confusion.tolist(),
        "overall_accuracy": accuracy.item(),
        "per_class": [],
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "macro_iou": iou.mean().item(),
    }

    for idx in range(confusion.size(0)):
        metrics["per_class"].append({
            "index": idx,
            "support": support[idx].item(),
            "precision": precision[idx].item(),
            "recall": recall[idx].item(),
            "f1": f1[idx].item(),
            "iou": iou[idx].item(),
        })
    return metrics


def display_metrics(metrics: dict, class_names: list[str]) -> None:
    print("Confusion matrix (rows=true, cols=pred):")
    cm = metrics["confusion_matrix"]
    header = "          " + " ".join(f"{name:>12}" for name in class_names)
    print(header)
    for row_name, row in zip(class_names, cm):
        values = " ".join(f"{value:12d}" for value in row)
        print(f"{row_name:>10} {values}")

    print("\nPer-class metrics:")
    for entry, class_name in zip(metrics["per_class"], class_names):
        print(
            f"{class_name:>10} | support={entry['support']:4.0f} | "
            f"precision={entry['precision']:.4f} | recall={entry['recall']:.4f} | "
            f"f1={entry['f1']:.4f} | iou={entry['iou']:.4f}"
        )

    print(
        "\nOverall: accuracy={:.4f} | macro_precision={:.4f} | "
        "macro_recall={:.4f} | macro_f1={:.4f} | macro_iou={:.4f}".format(
            metrics["overall_accuracy"],
            metrics["macro_precision"],
            metrics["macro_recall"],
            metrics["macro_f1"],
            metrics["macro_iou"],
        )
    )


def maybe_save_json(metrics: dict, path: Path | None) -> None:
    if path is None:
        return
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nSaved metrics to {path}")


def main() -> None:
    args = parse_args()
    dataset, loader = build_loader(args)
    class_names = [name for name, _ in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])]
    model = load_model(args, num_classes=len(class_names))
    device = torch.device(args.device)
    confusion = accumulate_predictions(model, loader, device, num_classes=len(class_names))
    metrics = compute_metrics(confusion)
    metrics["class_names"] = class_names
    display_metrics(metrics, class_names)
    maybe_save_json(metrics, args.save_json)


if __name__ == "__main__":
    main()
