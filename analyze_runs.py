#!/usr/bin/env python3
"""Aggregate historical MedViT runs and visualize the best checkpoints.

The script scans timestamped experiment folders (``runs/<experiment>/<timestamp>``)
and extracts validation metrics from their ``log.txt`` files.  It identifies the best
``test_acc1`` per experiment/model, merges optional confusion-matrix JSON artifacts,
and produces publication-friendly tables/plots.

Example:
    python analyze_runs.py \
        --runs-root E:\\Farzan\\Projects\\Stroke\\runs \
        --metrics-root metrics \
        --output-dir analysis_outputs
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Keep fonts readable in slides.
plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 160})

MODEL_LABELS: List[tuple[str, str]] = [
    ("convnext_tiny", "ConvNeXt-Tiny"),
    ("medvit_large", "MedViT-Large"),
    ("medvit_base", "MedViT-Base"),
    ("medvit_small", "MedViT-Small"),
    ("swin", "Swin-T"),
    ("efficientnet", "EfficientNet-B2"),
    ("vgg16", "VGG16"),
    ("inception", "Inception V3"),
    ("resnet50", "ResNet50"),
    ("densenet", "DenseNet121"),
    ("radimagenet", "RadImageNet"),
]

MODEL_NAME_LOOKUP: List[tuple[str, str]] = [
    ("medvit_large", "MedViT_large"),
    ("medvit_base", "MedViT_base"),
    ("medvit_small", "MedViT_small"),
    ("convnext_tiny", "convnext_tiny"),
    ("swin", "swin_tiny_patch4_window7_224"),
    ("efficientnetb2", "efficientnet_b2"),
    ("efficientnet", "efficientnet_b2"),
    ("inception", "inception_v3"),
    ("resnet50", "resnet50"),
    ("densenet", "densenet121"),
    ("vgg16", "vgg16"),
    ("radimagenet", "radimagenet_densenet121"),
]

PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZE_CHECKPOINT = PROJECT_ROOT / "analyze_checkpoint.py"

DATA_ROOT_MAP: Dict[tuple[str, str], Path] = {
    ("brainext", "slice"): PROJECT_ROOT / "data/SBStrokeCT_sculpt_splits_slice",
    ("brainext", "subject"): PROJECT_ROOT / "data/SBStrokeCT_sculpt_splits_subject",
    ("nonbrainext", "slice"): PROJECT_ROOT / "data/SBStrokeCT_splits",
    ("nonbrainext", "subject"): PROJECT_ROOT / "data/SBStrokeCT_splits_subject",
}


@dataclass
class RunRecord:
    experiment: str
    timestamp: str
    model: str
    dataset_type: str
    split_type: str
    regime: str
    best_epoch: int
    best_acc1: float
    best_acc5: float
    best_loss: float
    log_path: Path
    confusion_path: Optional[Path]
    confusion_matrix: Optional[List[List[int]]]
    class_names: Optional[List[str]]

    @property
    def condition(self) -> str:
        return f"{self.dataset_type}/{self.split_type}"


def infer_model(experiment: str) -> str:
    lower = experiment.lower()
    for key, label in MODEL_LABELS:
        if key in lower:
            return label
    # fall back to the raw experiment suffix (after leading ct_stroke_ prefix)
    return experiment.replace("ct_stroke_", "").replace("_", " ").title()


def infer_model_name_for_load(experiment: str) -> Optional[str]:
    lower = experiment.lower()
    for key, name in MODEL_NAME_LOOKUP:
        if key in lower:
            return name
    return None


def infer_dataset(experiment: str) -> str:
    lower = experiment.lower()
    if "sculpt" in lower or "nonbrain" in lower:
        return "nonbrainext"
    if "brainext" in lower:
        return "brainext"
    # default: skull-stripped (brain-extracted) volumes
    return "brainext"


def infer_split(experiment: str) -> str:
    lower = experiment.lower()
    if "subject" in lower:
        return "subject"
    if "slice" in lower:
        return "slice"
    return "unknown"


def infer_regime(experiment: str) -> str:
    lower = experiment.lower()
    if "_lp_" in lower or "linear" in lower:
        return "linear-probe"
    if "_ft_" in lower or "finetune" in lower or "_ft" in lower:
        return "finetune"
    return "unspecified"


def dataset_root_for(record: RunRecord) -> Optional[Path]:
    key = (record.dataset_type, record.split_type)
    path = DATA_ROOT_MAP.get(key)
    return path.resolve() if path else None


def parse_log(log_path: Path) -> Optional[Dict[str, float]]:
    best: Optional[Dict[str, float]] = None
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "test_acc1" not in payload:
                    continue
                if best is None or payload["test_acc1"] > best["test_acc1"]:
                    best = payload
    except FileNotFoundError:
        return None
    return best


def find_confusion(
    metrics_root: Path,
    experiment: str,
    timestamp: str,
    run_dir: Optional[Path] = None,
) -> Optional[Path]:
    candidates: List[Path] = []
    if metrics_root.exists():
        exact = metrics_root / f"{experiment}_val_{timestamp}.json"
        if exact.exists():
            return exact
        generic = metrics_root / f"{experiment}_val.json"
        if generic.exists():
            candidates.append(generic)
        pattern = metrics_root.glob(f"{experiment}_val*.json")
        candidates.extend(sorted(pattern))

        subdir = metrics_root / experiment
        if subdir.exists():
            sub_matches = sorted(subdir.glob(f"*{timestamp}*.json"))
            candidates.extend(sub_matches)
            sub_conf = subdir / "confusion.json"
            if sub_conf.exists():
                candidates.append(sub_conf)

    if run_dir and run_dir.exists():
        run_matches = sorted(run_dir.glob("**/*confusion*.json"))
        candidates.extend(run_matches)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_confusion(path: Optional[Path]) -> tuple[Optional[List[List[int]]], Optional[List[str]]]:
    if path is None:
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("confusion_matrix"), data.get("class_names")
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None


def ensure_confusion(
    record: RunRecord,
    metrics_root: Path,
    checkpoint_name: str,
    eval_batch_size: int,
    eval_device: str,
    eval_split: str,
) -> bool:
    if record.confusion_matrix:
        return True
    run_dir = record.log_path.parent
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        print(f"[warn] Missing checkpoint for {record.experiment}/{record.timestamp}: {checkpoint_path}")
        return False
    data_root = dataset_root_for(record)
    if not data_root or not data_root.exists():
        print(f"[warn] Data root not found for {record.experiment}: expected {data_root}")
        return False
    model_name = infer_model_name_for_load(record.experiment)
    if not model_name:
        print(f"[warn] Unable to infer model name for {record.experiment}")
        return False
    metrics_root.mkdir(parents=True, exist_ok=True)
    target_dir = metrics_root / record.experiment
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / f"{record.timestamp}_confusion.json"

    cmd = [
        sys.executable,
        str(ANALYZE_CHECKPOINT),
        "--checkpoint",
        str(checkpoint_path),
        "--model",
        model_name,
        "--data-root",
        str(data_root),
        "--split",
        eval_split,
        "--batch-size",
        str(eval_batch_size),
        "--device",
        eval_device,
        "--save-json",
        str(destination),
    ]
    print(f"[info] Generating confusion for {record.experiment}/{record.timestamp} -> {destination}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"[warn] Confusion generation failed for {record.experiment}/{record.timestamp}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
        return False
    record.confusion_path = destination
    record.confusion_matrix, record.class_names = load_confusion(destination)
    return record.confusion_matrix is not None


def scan_runs(runs_root: Path, metrics_root: Path) -> List[RunRecord]:
    records: List[RunRecord] = []
    for experiment_dir in sorted(runs_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        experiment = experiment_dir.name
        best_record: Optional[RunRecord] = None
        for run_dir in sorted(experiment_dir.iterdir()):
            log_path = run_dir / "log.txt"
            best_metrics = parse_log(log_path)
            if not best_metrics:
                continue
            confusion_path = find_confusion(metrics_root, experiment, run_dir.name, run_dir)
            confusion_matrix, class_names = load_confusion(confusion_path)
            record = RunRecord(
                experiment=experiment,
                timestamp=run_dir.name,
                model=infer_model(experiment),
                dataset_type=infer_dataset(experiment),
                split_type=infer_split(experiment),
                regime=infer_regime(experiment),
                best_epoch=int(best_metrics.get("epoch", -1)),
                best_acc1=float(best_metrics.get("test_acc1", 0.0)),
                best_acc5=float(best_metrics.get("test_acc5", 0.0)),
                best_loss=float(best_metrics.get("test_loss", 0.0)),
                log_path=log_path,
                confusion_path=confusion_path,
                confusion_matrix=confusion_matrix,
                class_names=class_names,
            )
            if (
                best_record is None
                or record.best_acc1 > best_record.best_acc1
            ):
                best_record = record
        if best_record:
            records.append(best_record)
    return records


def topk_records(records: Iterable[RunRecord], top_k: int = 5) -> List[RunRecord]:
    return sorted(records, key=lambda r: r.best_acc1, reverse=True)[:top_k]


def unique_records(records: Iterable[RunRecord]) -> List[RunRecord]:
    seen: Dict[tuple[str, str], RunRecord] = {}
    for record in records:
        key = (record.experiment, record.timestamp)
        seen[key] = record
    return list(seen.values())


def maybe_generate_confusions(
    records: Iterable[RunRecord],
    metrics_root: Path,
    checkpoint_name: str,
    eval_batch_size: int,
    eval_device: str,
    eval_split: str,
    enabled: bool,
) -> int:
    if not enabled:
        return 0
    generated = 0
    for record in unique_records(records):
        if ensure_confusion(record, metrics_root, checkpoint_name, eval_batch_size, eval_device, eval_split):
            generated += 1
    if generated:
        print(f"[info] Generated {generated} confusion matrices via on-demand evaluation.")
    return generated


def save_summary(records: Iterable[RunRecord], output_dir: Path) -> Path:
    rows = []
    for r in records:
        data = asdict(r)
        data["log_path"] = str(r.log_path)
        data["confusion_path"] = str(r.confusion_path) if r.confusion_path else ""
        rows.append(data)
    df = pd.DataFrame(rows)
    summary_path = output_dir / "run_summary.csv"
    df.to_csv(summary_path, index=False)
    return summary_path


def save_topk(records: List[RunRecord], output_dir: Path, filename: str = "top5_models.csv") -> Path:
    ordered = records
    df = pd.DataFrame(
        {
            "Rank": list(range(1, len(ordered) + 1)),
            "Model": [r.model for r in ordered],
            "Condition": [r.condition for r in ordered],
            "Regime": [r.regime for r in ordered],
            "Acc@1": [r.best_acc1 for r in ordered],
            "Acc@5": [r.best_acc5 for r in ordered],
            "Loss": [r.best_loss for r in ordered],
            "Best Epoch": [r.best_epoch for r in ordered],
            "Experiment": [r.experiment for r in ordered],
            "Timestamp": [r.timestamp for r in ordered],
        }
    )
    topk_path = output_dir / filename
    df.to_csv(topk_path, index=False)
    return topk_path


def plot_topk(
    records: List[RunRecord],
    output_dir: Path,
    filename: str = "top5_models.png",
    title: str = "Top-5 runs by validation accuracy",
) -> Path:
    if not records:
        raise RuntimeError("No runs available to plot top-5 results.")
    df = pd.DataFrame(
        {
            "Label": [f"{r.model} ({r.condition})" for r in records],
            "Acc@1": [r.best_acc1 for r in records],
            "Regime": [r.regime for r in records],
        }
    )
    plt.figure(figsize=(9, 4 + 0.6 * len(records)))
    ax = sns.barplot(data=df, x="Acc@1", y="Label", hue="Regime", dodge=False)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Best Acc@1 (%)")
    ax.set_ylabel("Model (Condition)")
    ax.set_title(title)
    for bar, acc in zip(ax.patches, df["Acc@1"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x + 0.5, y, f"{acc:.1f}%", va="center")
    plt.legend(title="Regime", loc="lower right")
    plt.tight_layout()
    plot_path = output_dir / filename
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def plot_accuracy_bars(records: Iterable[RunRecord], output_dir: Path) -> Path:
    df = pd.DataFrame(
        {
            "Model": [r.model for r in records],
            "Accuracy": [r.best_acc1 for r in records],
            "Condition": [r.condition for r in records],
        }
    )
    if df.empty:
        raise RuntimeError("No runs found to visualize.")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="Accuracy", hue="Condition")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("Best Acc@1 (%)")
    plt.title("Best validation accuracy per model and data condition")
    plt.tight_layout()
    plot_path = output_dir / "accuracy_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def plot_confusions(records: Iterable[RunRecord], output_dir: Path, prefix: str = "") -> List[Path]:
    paths: List[Path] = []
    for record in records:
        if not record.confusion_matrix:
            continue
        mat = record.confusion_matrix
        labels = record.class_names or ["NoStroke", "Stroke"]
        pred_labels = [f"Pred {label}" for label in labels]
        df = pd.DataFrame(mat, index=labels, columns=pred_labels)
        plt.figure(figsize=(4, 4))
        sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{record.model}\n{record.condition}")
        plt.ylabel("Ground Truth")
        plt.xlabel("Prediction")
        plt.tight_layout()
        suffix = f"{record.experiment}_{record.timestamp}"
        plot_path = output_dir / f"{prefix}confusion_{suffix}.png"
        plt.savefig(plot_path)
        plt.close()
        paths.append(plot_path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize MedViT stroke runs.")
    parser.add_argument("--runs-root", type=Path, required=True, help="Path to runs directory (contains experiment subfolders).")
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("metrics"),
        help="Directory that stores confusion-matrix JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Destination folder for summary artifacts.",
    )
    parser.add_argument(
        "--generate-confusions",
        action="store_true",
        help="If set, run analyze_checkpoint.py to create confusion matrices when missing.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="checkpoint_best.pth",
        help="Checkpoint filename to evaluate when generating confusion matrices.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Batch size used while generating confusion matrices.",
    )
    parser.add_argument(
        "--eval-device",
        type=str,
        default="cuda",
        help="Device identifier passed to analyze_checkpoint.py (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="val",
        help="Dataset split name evaluated when building confusion matrices.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.runs_root = args.runs_root.expanduser()
    args.metrics_root = args.metrics_root.expanduser()
    args.output_dir = args.output_dir.expanduser()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = scan_runs(args.runs_root, args.metrics_root)
    if not records:
        raise SystemExit("No completed runs discovered. Check --runs-root path.")

    summary_csv = save_summary(records, output_dir)
    top_records = topk_records(records)
    topk_csv = save_topk(top_records, output_dir)
    accuracy_plot = plot_accuracy_bars(records, output_dir)
    topk_plot = plot_topk(top_records, output_dir)

    subject_records = [r for r in records if r.split_type == "subject"]
    subject_top: List[RunRecord] = topk_records(subject_records) if subject_records else []
    subject_csv: Optional[Path] = None
    subject_plot: Optional[Path] = None
    subject_confusions: List[Path] = []
    if subject_top:
        subject_csv = save_topk(subject_top, output_dir, filename="top5_subject_models.csv")
        subject_plot = plot_topk(
            subject_top,
            output_dir,
            filename="top5_subject_models.png",
            title="Top-5 subject runs by validation accuracy",
        )

    slice_records = [r for r in records if r.split_type == "slice"]
    slice_top: List[RunRecord] = topk_records(slice_records) if slice_records else []
    slice_csv: Optional[Path] = None
    slice_plot: Optional[Path] = None
    slice_confusions: List[Path] = []
    if slice_top:
        slice_csv = save_topk(slice_top, output_dir, filename="top5_slice_models.csv")
        slice_plot = plot_topk(
            slice_top,
            output_dir,
            filename="top5_slice_models.png",
            title="Top-5 slice runs by validation accuracy",
        )

    target_records = top_records + subject_top + slice_top
    maybe_generate_confusions(
        target_records,
        args.metrics_root,
        args.checkpoint_name,
        args.eval_batch_size,
        args.eval_device,
        args.eval_split,
        args.generate_confusions,
    )

    confusion_plots = plot_confusions(records, output_dir)
    if subject_top:
        subject_confusions = plot_confusions(subject_top, output_dir, prefix="subject_top5_")
    if slice_top:
        slice_confusions = plot_confusions(slice_top, output_dir, prefix="slice_top5_")

    print(f"Saved summary table: {summary_csv}")
    print(f"Saved top-5 table: {topk_csv}")
    if subject_csv:
        print(f"Saved subject-only top-5 table: {subject_csv}")
    if slice_csv:
        print(f"Saved slice-only top-5 table: {slice_csv}")
    print(f"Saved accuracy plot: {accuracy_plot}")
    print(f"Saved top-5 plot: {topk_plot}")
    if subject_plot:
        print(f"Saved subject-only top-5 plot: {subject_plot}")
    if slice_plot:
        print(f"Saved slice-only top-5 plot: {slice_plot}")

    if confusion_plots:
        print("Saved confusion heatmaps:")
        for path in confusion_plots:
            print(f"  - {path}")
    else:
        print("No confusion matrices were available for plotting.")

    if subject_top:
        if subject_confusions:
            print("Saved subject top-5 confusion heatmaps:")
            for path in subject_confusions:
                print(f"  - {path}")
        else:
            print("Subject top-5 runs did not have confusion matrices available.")

    if slice_top:
        if slice_confusions:
            print("Saved slice top-5 confusion heatmaps:")
            for path in slice_confusions:
                print(f"  - {path}")
        else:
            print("Slice top-5 runs did not have confusion matrices available.")


if __name__ == "__main__":
    main()
