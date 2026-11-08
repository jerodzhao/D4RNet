#!/usr/bin/env python3
"""Export the deployment student without auxiliary heads, then check equivalence."""

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from d4rnet.data import CLASS_NAMES, MEAN, STD
from d4rnet.runtime import load_student, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backbone", choices=["mobilenet_v2", "mobilenet_v3_large"])
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare ONNX Runtime CPU outputs with PyTorch",
    )
    args = parser.parse_args()
    if min(args.height, args.width) < 32:
        parser.error("Input height and width must be at least 32")
    import onnx

    model = load_student(args.weights, args.backbone).cpu().eval()
    # Export a single-input inference graph so the training forward's optional
    # boolean flag cannot become an ONNX tracing input.
    deployment = torch.nn.Sequential(
        OrderedDict(
            features=model.features,
            avgpool=model.avgpool,
            flatten=torch.nn.Flatten(1),
            main_classifier=model.main_classifier,
        )
    ).eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    example = torch.randn(1, 3, args.height, args.width)
    axes = {"input": {0: "batch"}, "output": {0: "batch"}} if args.dynamic_batch else None
    with torch.inference_mode():
        torch.onnx.export(
            deployment,
            example,
            str(output),
            opset_version=args.opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=axes,
        )
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    names = [initializer.name for initializer in graph.graph.initializer]
    if any(name.startswith(("aux_projection.", "aux_classifier.")) for name in names):
        raise RuntimeError("Export unexpectedly contains an auxiliary branch")
    difference = None
    if args.check:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        for batch_size in [1, 2] if args.dynamic_batch else [1]:
            inputs = torch.randn(batch_size, 3, args.height, args.width)
            with torch.inference_mode():
                expected = model(inputs).numpy()
            actual = session.run(None, {"input": inputs.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)
            difference = max(difference or 0, float(np.max(np.abs(actual - expected))))
    write_json(
        output.with_suffix(".json"),
        dict(
            backbone=model.backbone_name,
            input_shape=[1, 3, args.height, args.width],
            layout="NCHW",
            color="RGB",
            input_dtype="float32",
            input_range="normalized",
            normalization=dict(mean=MEAN, std=STD),
            resize="bilinear",
            class_names=CLASS_NAMES,
            onnxruntime_max_abs_difference=difference,
        ),
    )
    print(f"Exported {output}; ONNX graph validated; max abs difference: {difference}")


if __name__ == "__main__":
    main()
