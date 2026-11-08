#!/usr/bin/env python3
"""Export the paper's cross-split and per-category tables as JSON and CSV."""

import argparse
import csv
import json
from pathlib import Path

from torch.utils.data import DataLoader

from d4rnet.data import CLASS_NAMES, DriverDataset, dataset_specs
from d4rnet.metrics import evaluate
from d4rnet.runtime import add_data_arguments, load_student, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument(
        "--weights-map", required=True, help="JSON: training split -> checkpoint path"
    )
    parser.add_argument("--datasets", nargs="+", help="Test splits; defaults to all 15")
    parser.add_argument("--backbone", choices=["mobilenet_v2", "mobilenet_v3_large"])
    parser.add_argument("--output", default="results/cross_evaluation")
    args = parser.parse_args()
    mapping_path = Path(args.weights_map).resolve()
    weights = json.loads(mapping_path.read_text())
    if not weights:
        parser.error("The weight map is empty; add the models to evaluate")
    datasets = args.datasets or list(dataset_specs())
    device = select_device(args.device)
    loaders = {
        name: DataLoader(
            DriverDataset(
                args.data_root,
                name,
                "test",
                splits_root=args.splits_root,
                image_size=args.image_size,
            ),
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        for name in datasets
    }
    results = {}
    for training_split, path in weights.items():
        checkpoint = Path(path)
        if not checkpoint.is_absolute():
            checkpoint = mapping_path.parent / checkpoint
        model = load_student(checkpoint, args.backbone).to(device).eval()
        results[training_split] = {}
        for name, loader in loaders.items():
            result = evaluate(model, loader, device)
            results[training_split][name] = result
            print(f"{training_split} -> {name}: {result['top1']:.2f}%", flush=True)
        del model
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", results)
    with (output / "cross_split_top1.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trained_on", *datasets])
        for training_split, result in results.items():
            writer.writerow([training_split, *[result[name]["top1"] for name in datasets]])
    # Fig. 6 uses the DN-all student across 15 test splits (15 x 22).
    for training_split, result in results.items():
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in training_split)
        with (output / f"{safe_name}_per_category.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["tested_on", *[f"C{i + 1}: {name}" for i, name in enumerate(CLASS_NAMES)]]
            )
            for name in datasets:
                writer.writerow([name, *result[name]["per_class_accuracy"]])


if __name__ == "__main__":
    main()
