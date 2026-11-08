#!/usr/bin/env python3
"""Train the D4RNet student with offline KD and optional domain adversarial loss."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from d4rnet.data import DriverDataset, TeacherLogits
from d4rnet.grl import GRLController
from d4rnet.losses import classification_loss, distillation_weight
from d4rnet.metrics import evaluate
from d4rnet.models import D4RNet
from d4rnet.runtime import (
    add_data_arguments,
    capture_rng,
    load_torch,
    load_weights,
    parse_config,
    restore_rng,
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
    parser.add_argument(
        "--backbone", choices=["mobilenet_v2", "mobilenet_v3_large"], default="mobilenet_v2"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--milestones", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--lr-gamma", type=float, default=0.2)
    parser.add_argument("--kd", dest="use_kd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--grl", dest="use_grl", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--beta1", type=float, default=0.7)
    parser.add_argument("--beta1-decay-start", type=int, default=10)
    parser.add_argument("--beta1-decay-step", type=float, default=0.04)
    parser.add_argument("--beta1-minimum", type=float, default=0.1)
    parser.add_argument("--beta2", type=float, default=0.2)
    parser.add_argument("--grl-mode", choices=["legacy", "algorithm1"], default="legacy")
    parser.add_argument("--teacher-logits")
    parser.add_argument("--allow-legacy-logits", action="store_true")
    parser.add_argument("--resume", help="Path to last.pth, including optimizer and RNG state")
    parser.add_argument("--output", default="runs/D4")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--tensorboard", action="store_true")
    args = parse_config(parser, "d4rnet.json")
    validate_training_arguments(args)
    if args.use_kd and not args.teacher_logits:
        parser.error("--teacher-logits is required with KD; generate it with cache_teacher.py")
    if args.warmup_epochs < 0 or args.temperature <= 0 or args.beta2 < 0:
        parser.error("Invalid warm-up, temperature or auxiliary loss weight")
    if not 0 <= args.beta1_minimum <= args.beta1 <= 1 or args.beta1_decay_step < 0:
        parser.error("Require 0 <= beta1_minimum <= beta1 <= 1 and nonnegative decay")
    return args


def learning_rate(args, epoch, batch_index, batches):
    if epoch <= args.warmup_epochs:
        # Matches the original WarmUpLR: the first optimizer step has lr=0.
        return args.lr * ((epoch - 1) * batches + batch_index) / (args.warmup_epochs * batches)
    return args.lr * args.lr_gamma ** sum(epoch >= step for step in args.milestones)


def train_epoch(model, loader, optimizer, teacher, controller, device, epoch, args):
    model.train()
    totals = dict(
        loss=0.0, hard=0.0, soft=0.0, auxiliary=0.0, correct=0, domain_correct=0, samples=0
    )
    beta1 = distillation_weight(
        epoch, args.beta1, args.beta1_decay_start, args.beta1_decay_step, args.beta1_minimum
    )
    if controller:
        controller.begin_epoch(epoch)
        controller.apply(model)
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(args, epoch, batch_index, len(loader))
        optimizer.zero_grad(set_to_none=True)
        if controller:
            logits, domains = model(images, return_aux=True)
            targets = batch["domain"].to(device, non_blocking=True)
            auxiliary = nn.functional.cross_entropy(domains, targets)
        else:
            logits = model(images)
            auxiliary = logits.new_zeros(())
        teacher_values = teacher[batch["index"]].to(device, non_blocking=True) if teacher else None
        hard, soft = classification_loss(logits, labels, teacher_values, args.temperature, beta1)
        loss = hard + soft + args.beta2 * auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
        loss.backward()
        optimizer.step()
        size = len(labels)
        for key, value in [
            ("loss", loss),
            ("hard", hard),
            ("soft", soft),
            ("auxiliary", auxiliary),
        ]:
            totals[key] += value.detach().item() * size
        totals["correct"] += logits.argmax(1).eq(labels).sum().item()
        totals["samples"] += size
        if controller:
            correct = domains.argmax(1).eq(targets).sum().item()
            totals["domain_correct"] += correct
            controller.update(correct / size, epoch)
            controller.apply(model)
    count = totals.pop("samples")
    result = {key: value / count for key, value in totals.items()}
    result["top1"] = 100 * result.pop("correct")
    result["domain_accuracy"] = result.pop("domain_correct")
    result["beta1"] = beta1 if teacher else 0.0
    result["lr"] = optimizer.param_groups[0]["lr"]
    if controller:
        result.update(grl_strength=controller.strength, grl_enabled=controller.enabled)
    return result


def main():
    args = parse_args()
    device = select_device(args.device)
    seed_everything(args.seed)
    output = Path(args.output)
    if not args.resume and (output / "last.pth").exists():
        raise FileExistsError(f"{output} already contains a run; choose a new --output or --resume")
    train = DriverDataset(
        args.data_root,
        args.dataset,
        auxiliary=args.use_grl,
        splits_root=args.splits_root,
        image_size=args.image_size,
    )
    val = DriverDataset(
        args.data_root,
        args.dataset,
        "val",
        splits_root=args.splits_root,
        image_size=args.image_size,
    )
    teacher = (
        TeacherLogits(args.teacher_logits, train, args.allow_legacy_logits) if args.use_kd else None
    )
    generator = torch.Generator().manual_seed(args.seed)
    options = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    loader = DataLoader(train, shuffle=True, generator=generator, **options)
    validation = DataLoader(val, shuffle=False, **options)
    model = D4RNet(
        args.backbone,
        num_domains=train.spec["num_domains"] if args.use_grl else 0,
        pretrained=args.pretrained and not args.resume,
    ).to(device)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    controller = (
        GRLController(
            train.spec["num_domains"], mode=args.grl_mode, warmup_epochs=args.warmup_epochs
        )
        if args.use_grl
        else None
    )
    start, best = 1, -1.0
    if args.resume:
        state = load_torch(args.resume)
        if state.get("sample_fingerprint") != train.fingerprint:
            raise ValueError("Resume checkpoint belongs to a different training manifest")
        mutable = {
            "config",
            "resume",
            "output",
            "data_root",
            "splits_root",
            "teacher_logits",
            "workers",
            "device",
            "tensorboard",
            "save_every",
            "epochs",
            "pretrained",
        }
        for name, value in state["config"].items():
            if name not in mutable and getattr(args, name) != value:
                raise ValueError(f"Resume configuration changed: {name}")
        load_weights(model, state)
        optimizer.load_state_dict(state["optimizer"])
        if state.get("grl"):
            controller = GRLController(**state["grl"])
        generator.set_state(state["loader_rng"])
        restore_rng(state["rng"])
        start, best = state["epoch"] + 1, state["best_top1"]
    if start > args.epochs:
        raise ValueError("Resume epoch is already at or beyond --epochs")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", vars(args))
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(output / "tensorboard"))
    print(
        f"Student: {args.backbone}; dataset: {args.dataset}; device: {device}; "
        f"train/val: {len(train)}/{len(val)}",
        flush=True,
    )
    try:
        for epoch in range(start, args.epochs + 1):
            started = time.perf_counter()
            training = train_epoch(
                model, loader, optimizer, teacher, controller, device, epoch, args
            )
            metrics = evaluate(model, validation, device)
            improved = metrics["top1"] > best
            best = max(best, metrics["top1"])
            record = dict(
                epoch=epoch,
                train=training,
                val=metrics,
                seconds=time.perf_counter() - started,
                best_top1=best,
            )
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            checkpoint = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                config=vars(args),
                best_top1=best,
                sample_fingerprint=train.fingerprint,
                grl=controller.state_dict() if controller else None,
                rng=capture_rng(),
                loader_rng=generator.get_state(),
            )
            save_checkpoint(output / "last.pth", checkpoint)
            if improved:
                save_checkpoint(output / "best.pth", checkpoint)
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(output / f"epoch_{epoch:03d}.pth", checkpoint)
            if writer:
                for key, value in training.items():
                    writer.add_scalar(f"train/{key}", value, epoch)
                writer.add_scalar("val/top1", metrics["top1"], epoch)
                writer.add_scalar("val/loss", metrics["loss"], epoch)
            print(
                f"Epoch {epoch:02d}/{args.epochs}: train={training['top1']:.2f}% "
                f"val={metrics['top1']:.2f}% best={best:.2f}% "
                f"loss={training['loss']:.4f} ({record['seconds']:.1f}s)",
                flush=True,
            )
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
