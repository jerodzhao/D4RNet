#!/usr/bin/env python3
"""Classify an image or annotate a video with PyTorch or ONNX Runtime."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from d4rnet.data import CLASS_NAMES, preprocessing
from d4rnet.runtime import load_student, select_device, write_json


class Predictor:
    def __init__(self, args):
        self.args = args
        self.onnx = Path(args.weights).suffix.lower() == ".onnx"
        if self.onnx:
            import onnxruntime as ort

            requested = args.providers or ["CPUExecutionProvider"]
            missing = set(requested) - set(ort.get_available_providers())
            if missing:
                raise RuntimeError(
                    f"ONNX providers unavailable: {sorted(missing)}; "
                    f"installed: {ort.get_available_providers()}"
                )
            self.session = ort.InferenceSession(args.weights, providers=requested)
            input_spec = self.session.get_inputs()[0]
            self.input_name = input_spec.name
            shape = input_spec.shape
            if len(shape) != 4 or shape[1] != 3:
                raise ValueError("Expected an NCHW RGB ONNX model")
            height = shape[2] if isinstance(shape[2], int) else args.image_size
            width = shape[3] if isinstance(shape[3], int) else args.image_size
            # torchvision Resize accepts (height, width), also for non-square exports.
            self.transform = preprocessing(args.image_size)
            self.transform.transforms[0].size = (height, width)
            print(f"ONNX providers: {self.session.get_providers()}")
        else:
            if args.providers:
                raise ValueError("--providers applies to ONNX models")
            self.device = select_device(args.device)
            self.model = load_student(args.weights, args.backbone).to(self.device).eval()
            self.transform = preprocessing(args.image_size)

    def __call__(self, image):
        inputs = self.transform(image.convert("RGB")).unsqueeze(0)
        if self.onnx:
            logits = self.session.run(None, {self.input_name: inputs.numpy()})[0][0]
        else:
            with torch.inference_mode():
                logits = self.model(inputs.to(self.device)).cpu().numpy()[0]
        shifted = logits - logits.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        return [
            dict(
                class_index=int(index),
                category=f"C{index + 1}",
                label=CLASS_NAMES[index],
                probability=float(probabilities[index]),
            )
            for index in probabilities.argsort()[::-1][: self.args.topk]
        ]


def annotate_video(args, predict):
    import cv2

    if not args.output:
        raise ValueError("Video inference requires --output annotated.mp4")
    if Path(args.output).resolve() == Path(args.input).resolve():
        raise ValueError("Video output must differ from input")
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")
    writer = None
    frames = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create video: {args.output}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            best = predict(Image.fromarray(rgb))[0]
            text = f"{best['category']} {best['label']}: {best['probability']:.1%}"
            cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            writer.write(frame)
            frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frames == 0:
        raise RuntimeError("No video frames could be read")
    print(f"Annotated {frames} frames: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="JSON for an image; MP4 for a video")
    parser.add_argument("--backbone", choices=["mobilenet_v2", "mobilenet_v3_large"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--providers", nargs="+", help="Ordered ONNX Runtime execution providers")
    parser.add_argument("--topk", type=int, choices=range(1, 23), default=5)
    args = parser.parse_args()
    predictor = Predictor(args)
    if Path(args.input).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        annotate_video(args, predictor)
    else:
        with Image.open(args.input) as image:
            predictions = predictor(image)
        print(json.dumps(predictions, indent=2))
        if args.output:
            if Path(args.output).resolve() == Path(args.input).resolve():
                raise ValueError("Output must differ from input")
            write_json(args.output, predictions)


if __name__ == "__main__":
    main()
