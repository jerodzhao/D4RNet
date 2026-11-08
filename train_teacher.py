#!/usr/bin/env python3
"""Fine-tune a Swin-B teacher on one 100-Driver split, optionally with torchrun."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from d4rnet.data import DriverDataset
from d4rnet.models import create_teacher
from d4rnet.runtime import (
    add_data_arguments,
    parse_config,
    save_checkpoint,
    seed_everything,
    seed_worker,
    select_device,
    validate_training_arguments,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="runs/teacher/D4")
    args = parse_config(parser, "teacher.json")
    validate_training_arguments(args)
    return args


def run_epoch(model, loader, device, optimizer=None, distributed=False):
    model.train(optimizer is not None)
    # Validation shards can differ in length. Calling the unwrapped module avoids
    # DDP buffer broadcasts on uneven numbers of validation batches.
    active = (
        model.module if optimizer is None and isinstance(model, DistributedDataParallel) else model
    )
    sums = torch.zeros(3, dtype=torch.float64, device=device)
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = active(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite teacher loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            sums[0] += loss.detach() * len(labels)
            sums[1] += logits.argmax(1).eq(labels).sum()
            sums[2] += len(labels)
    if distributed:
        dist.all_reduce(sums)
    if sums[2].item() == 0:
        raise ValueError("Empty data loader")
    return dict(
        loss=(sums[0] / sums[2]).item(),
        top1=(100 * sums[1] / sums[2]).item(),
        samples=int(sums[2].item()),
    )


def main():
    args = parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = select_device(args.device)
    if world > 1 and device.type == "cuda":
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        seed_everything(args.seed)
        output = Path(args.output)
        if (output / "last.pth").exists():
            raise FileExistsError(f"{output} already contains a run; use a new --output")
        train = DriverDataset(args.data_root, args.dataset, splits_root=args.splits_root)
        val = DriverDataset(args.data_root, args.dataset, "val", splits_root=args.splits_root)
        sampler = DistributedSampler(train, seed=args.seed) if world > 1 else None
        # No validation padding: each sample contributes exactly once globally.
        val_shard = Subset(val, range(rank, len(val), world)) if world > 1 else val
        options = dict(
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
        )
        train_loader = DataLoader(train, sampler=sampler, shuffle=sampler is None, **options)
        val_loader = DataLoader(val_shard, shuffle=False, **options)
        model = create_teacher(args.pretrained).to(device)
        if world > 1:
            model = DistributedDataParallel(
                model, device_ids=[local] if device.type == "cuda" else None
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "config.json", vars(args))
            print(
                f"Swin-B teacher: {args.dataset}; {world} process(es); "
                f"batch size per process: {args.batch_size}",
                flush=True,
            )
        best = -1.0
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            if sampler is not None:
                sampler.set_epoch(epoch)
            training = run_epoch(model, train_loader, device, optimizer, world > 1)
            validation = run_epoch(model, val_loader, device, distributed=world > 1)
            scheduler.step()
            if rank == 0:
                improved = validation["top1"] > best
                best = max(best, validation["top1"])
                unwrapped = model.module if world > 1 else model
                state = dict(
                    model=unwrapped.state_dict(),
                    epoch=epoch,
                    config=vars(args),
                    best_top1=best,
                    sample_fingerprint=train.fingerprint,
                )
                save_checkpoint(output / "last.pth", state)
                if improved:
                    save_checkpoint(output / "best.pth", state)
                record = dict(
                    epoch=epoch,
                    train=training,
                    val=validation,
                    lr=scheduler.get_last_lr()[0],
                    seconds=time.perf_counter() - started,
                )
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(
                    f"Epoch {epoch:02d}: train={training['top1']:.2f}% "
                    f"val={validation['top1']:.2f}% best={best:.2f}%",
                    flush=True,
                )
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
