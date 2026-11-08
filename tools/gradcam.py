#!/usr/bin/env python3
"""Visualize the final student feature map with Grad-CAM (paper Fig. 7)."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from d4rnet.data import CLASS_NAMES, preprocessing
from d4rnet.runtime import load_student, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-index", type=int, choices=range(22))
    parser.add_argument("--backbone", choices=["mobilenet_v2", "mobilenet_v3_large"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if Path(args.output).resolve() == Path(args.image).resolve():
        raise ValueError("Output must differ from input")
    device = select_device(args.device)
    model = load_student(args.weights, args.backbone).to(device).eval()
    with Image.open(args.image) as image:
        original = image.convert("RGB")
    captured = {}

    def capture(module, inputs, output):
        captured["features"] = output
        output.retain_grad()

    handle = model.features[-1].register_forward_hook(capture)
    try:
        logits = model(preprocessing()(original).unsqueeze(0).to(device))
        target = args.class_index if args.class_index is not None else int(logits.argmax(1))
        logits[0, target].backward()
        features = captured["features"]
        weights = features.grad.mean((2, 3), keepdim=True)
        heat = (weights * features).sum(1).relu()
        heat = torch.nn.functional.interpolate(
            heat[:, None],
            size=(original.height, original.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        heat = heat.detach().cpu().numpy()
        heat = heat / max(float(heat.max()), 1e-12)
        # A simple red/yellow heatmap keeps matplotlib/OpenCV optional.
        colors = np.stack([np.ones_like(heat), heat, np.zeros_like(heat)], axis=2) * 255
        alpha = (heat * 0.5)[..., None]
        overlay = np.asarray(original) * (1 - alpha) + colors * alpha
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(args.output)
        print(f"C{target + 1}: {CLASS_NAMES[target]}; saved {args.output}")
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
