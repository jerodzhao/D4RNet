#!/usr/bin/env python3
"""Measure ONNX Runtime model latency; preprocessing and video I/O are excluded."""

import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("Require positive iterations and nonnegative warmup")
    import onnxruntime as ort

    missing = set(args.providers) - set(ort.get_available_providers())
    if missing:
        raise RuntimeError(f"Unavailable providers: {sorted(missing)}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.model, sess_options=options, providers=args.providers)
    inputs = session.get_inputs()[0]
    shape = [
        1 if i == 0 and not isinstance(size, int) else size for i, size in enumerate(inputs.shape)
    ]
    if any(not isinstance(size, int) for size in shape):
        raise ValueError("Benchmark requires fixed spatial dimensions")
    feed = {inputs.name: np.random.default_rng(42).standard_normal(shape).astype(np.float32)}
    for _ in range(args.warmup):
        session.run(None, feed)
    durations = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        session.run(None, feed)
        durations.append(time.perf_counter() - started)
    print(f"Providers: {session.get_providers()}; input: {shape}")
    print(
        f"Mean: {1000 * np.mean(durations):.3f} ms; "
        f"median: {1000 * np.median(durations):.3f} ms; "
        f"p95: {1000 * np.percentile(durations, 95):.3f} ms; "
        f"throughput: {shape[0] / np.mean(durations):.2f} images/s"
    )


if __name__ == "__main__":
    main()
