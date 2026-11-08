#!/usr/bin/env python3
"""Evaluate a student or Swin-B teacher on one or more complete test splits."""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from d4rnet.data import CLASS_NAMES, DriverDataset
from d4rnet.metrics import evaluate
from d4rnet.models import create_teacher
from d4rnet.runtime import (
    add_data_arguments,
    load_student,
    load_torch,
    load_weights,
    select_device,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--kind", choices=["student", "teacher"], default="student")
    parser.add_argument("--backbone", choices=["mobilenet_v2", "mobilenet_v3_large"])
    parser.add_argument("--datasets", nargs="+", help="Overrides --dataset for cross-split testing")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", help="Output JSON report")
    args = parser.parse_args()
    device = select_device(args.device)
    if args.kind == "student":
        model = load_student(args.weights, args.backbone)
    else:
        if args.image_size != 224:
            parser.error("The Swin-B teacher requires 224x224 inputs")
        model = create_teacher()
        load_weights(model, load_torch(args.weights))
    model.to(device).eval()
    results = {}
    for name in args.datasets or [args.dataset]:
        dataset = DriverDataset(
            args.data_root,
            name,
            args.split,
            splits_root=args.splits_root,
            image_size=args.image_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        result = evaluate(model, loader, device)
        result["sample_fingerprint"] = dataset.fingerprint
        results[name] = result
        print(
            f"{name:6s}: Top-1 {result['top1']:.2f}%  Top-5 {result['top5']:.2f}%  "
            f"Precision {result['precision_macro']:.2f}%  Recall {result['recall_macro']:.2f}%  "
            f"F1 {result['f1_macro']:.2f}%  n={result['samples']}",
            flush=True,
        )
    if args.output:
        mean = sum(value["top1"] for value in results.values()) / len(results)
        write_json(
            args.output,
            dict(
                weights=Path(args.weights).name,
                split=args.split,
                image_size=args.image_size,
                class_names=CLASS_NAMES,
                mean_top1_across_splits=mean,
                results=results,
            ),
        )
    if len(results) > 1:
        mean = sum(value["top1"] for value in results.values()) / len(results)
        print(f"Unweighted mean Top-1 over {len(results)} test splits: {mean:.2f}%")


if __name__ == "__main__":
    main()
