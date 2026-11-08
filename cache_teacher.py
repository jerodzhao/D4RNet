#!/usr/bin/env python3
"""Cache deterministic teacher logits with explicit sample-order metadata."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from d4rnet.data import DriverDataset
from d4rnet.models import create_teacher
from d4rnet.runtime import (
    add_data_arguments,
    load_torch,
    load_weights,
    save_checkpoint,
    select_device,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    args = parser.parse_args()
    if args.image_size != 224:
        parser.error("The Swin-B teacher requires --image-size 224")
    if Path(args.output).exists():
        raise FileExistsError(f"Cache already exists: {args.output}; choose a new output")
    device = select_device(args.device)
    dataset = DriverDataset(args.data_root, args.dataset, args.split, splits_root=args.splits_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = create_teacher()
    checkpoint = load_torch(args.weights)
    configured = checkpoint.get("config", {}).get("dataset")
    if configured is not None and configured != args.dataset:
        raise ValueError("Teacher checkpoint was trained on a different dataset")
    load_weights(model, checkpoint)
    model.to(device).eval()
    logits = torch.empty(len(dataset), 22)
    labels = torch.tensor([sample.label for sample in dataset.samples])
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            prediction = model(batch["image"].to(device, non_blocking=True)).cpu()
            logits[batch["index"]] = prediction
            if step % 100 == 0 or step == len(loader):
                print(
                    f"Cached {min(step * args.batch_size, len(dataset))}/{len(dataset)}", flush=True
                )
    save_checkpoint(
        args.output,
        dict(
            logits=logits,
            labels=labels,
            indices=torch.arange(len(dataset)),
            sample_fingerprint=dataset.fingerprint,
            image_size=224,
            dataset=args.dataset,
            split=args.split,
            teacher_checkpoint=Path(args.weights).name,
        ),
    )
    print(f"Saved {args.output} ({tuple(logits.shape)})")


if __name__ == "__main__":
    main()
