"""Entry point for training MedViT on the SBStrokeCT_classifier dataset."""
import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CUSTOM_DATASET_ROOT = PROJECT_ROOT / "CustomDataset"
if str(CUSTOM_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(CUSTOM_DATASET_ROOT))

from CustomDataset.main import get_args_parser, main as medvit_main

DEFAULT_DATA_ROOT = Path(r"G:\.shortcut-targets-by-id\1OCRaLxrUFclmW2_RKXTema7sPbSC1rOT\SBStrokeCT_classifier")


def build_parser() -> argparse.ArgumentParser:
    parent_parser = get_args_parser()
    parser = argparse.ArgumentParser(
        description="Train MedViT for stroke vs non-stroke CT classification",
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory that contains the train/val folders for SBStrokeCT_classifier.",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Subdirectory name under data-root that holds training images.",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="val",
        help="Subdirectory name under data-root that holds validation images.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of diagnostic classes (stroke vs non-stroke).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Directory used to store model checkpoints and logs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ct_stroke",
        help="Name of the experiment subfolder inside output-root.",
    )
    parser.add_argument(
        "--timestamp-output",
        action="store_true",
        help="Append a timestamp to the experiment directory for isolated runs.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force training on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--ct-optim-preset",
        action="store_true",
        help=(
            "Apply a tuned optimization preset (longer schedule, layer-wise decay, "
            "gentle mixup, and lighter weight decay) tailored for SBStrokeCT."
        ),
    )
    parser.set_defaults(
        data_set="image_folder",
        batch_size=16,
        epochs=60,
        lr=3e-4,
        model="MedViT_base",
        disable_lr_scaling=True,
        weight_decay=0.05,
        mixup=0.0,
        cutmix=0.0,
        smoothing=0.0,
        color_jitter=0.0,
        aa="",
        drop_path=0.1,
        warmup_epochs=5,
        num_workers=4,
        pin_mem=True,
        output_dir="",
    )
    return parser


def prepare_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()

    if args.ct_optim_preset:
        preset_values = dict(
            epochs=90,
            lr=1.5e-4,
            min_lr=5e-6,
            warmup_epochs=10,
            layer_decay=0.5,
            weight_decay=0.02,
            drop_path=0.2,
            mixup=0.1,
            cutmix=0.0,
            mixup_prob=0.7,
            color_jitter=0.1,
            aa="rand-m5-mstd0.5",
            reprob=0.05,
            smoothing=0.05,
        )
        for key, value in preset_values.items():
            setattr(args, key, value)

    data_root = args.data_root.expanduser().resolve()
    train_dir = (data_root / args.train_split).resolve()
    val_dir = (data_root / args.val_split).resolve()
    if not train_dir.exists():
        raise FileNotFoundError(f"Training split not found at {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation split not found at {val_dir}")

    output_root = args.output_root.expanduser().resolve()
    experiment_dir = output_root / args.experiment_name
    if args.timestamp_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = experiment_dir / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if not args.output_dir:
        args.output_dir = str(experiment_dir)
    else:
        explicit_dir = Path(args.output_dir).expanduser().resolve()
        explicit_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(explicit_dir)

    args.data_path = str(train_dir)
    args.eval_data_path = str(val_dir)
    args.nb_classes = args.num_classes

    radimagenet_root = getattr(args, "radimagenet_weights_root", None)
    if radimagenet_root:
        args.radimagenet_weights_root = Path(radimagenet_root).expanduser()
    else:
        args.radimagenet_weights_root = None

    radimagenet_checkpoint = getattr(args, "radimagenet_checkpoint", None)
    if radimagenet_checkpoint:
        args.radimagenet_checkpoint = Path(radimagenet_checkpoint).expanduser()
    else:
        args.radimagenet_checkpoint = None

    if args.cpu:
        args.device = "cpu"
    return args


def main() -> None:
    args = prepare_args()
    medvit_main(args)


if __name__ == "__main__":
    main()
