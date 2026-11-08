"""Shared configuration, checkpoint, and reproducibility utilities."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .data import PROJECT_ROOT


def parse_config(parser, default_config):
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / default_config))
    partial, _ = parser.parse_known_args()
    with Path(partial.config).open() as handle:
        values = json.load(handle)
    destinations = {action.dest for action in parser._actions}
    unknown = values.keys() - destinations
    if unknown:
        parser.error(f"Unknown configuration keys: {sorted(unknown)}")
    parser.set_defaults(**values)
    return parser.parse_args()


def add_data_arguments(parser):
    parser.add_argument("--dataset", default="D4")
    parser.add_argument("--data-root", default="data/100-driver")
    parser.add_argument("--splits-root", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")


def select_device(name="auto"):
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else torch.device(name)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def load_torch(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and state.get("d4rnet_format") == "1":
        tensors = state["training_tensors"]

        def decode(value):
            if isinstance(value, list):
                return [decode(item) for item in value]
            if isinstance(value, dict):
                if "tensor" in value:
                    return tensors[value["tensor"]]
                if "tuple" in value:
                    return tuple(decode(item) for item in value["tuple"])
                return {decode(key): decode(item) for key, item in value["dict"]}
            return value

        result = decode(json.loads(state["metadata_json"]))
        if "model" in state:
            result["model"] = state["model"]
        return result
    return state


def checkpoint_weights(checkpoint):
    weights = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Checkpoint does not contain a state dictionary")
    return {key.removeprefix("module."): value for key, value in weights.items()}


def load_weights(model, checkpoint, deployment=False):
    weights = checkpoint_weights(checkpoint)
    if deployment:
        prefixes = ("aux_projection.", "aux_classifier.", "grl.")
        weights = {key: value for key, value in weights.items() if not key.startswith(prefixes)}
    # Never hide missing backbone/head weights with strict=False.
    model.load_state_dict(weights, strict=True)


def load_student(path, backbone=None):
    from .models import D4RNet

    checkpoint = load_torch(path)
    configured = checkpoint.get("config", {}).get("backbone")
    if backbone is not None and configured is not None and backbone != configured:
        raise ValueError("Requested backbone differs from checkpoint metadata")
    if backbone is None:
        backbone = configured or "mobilenet_v2"
    model = D4RNet(backbone=backbone, num_domains=0)
    load_weights(model, checkpoint, deployment=True)
    return model


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # PyTorch 1.13's weights_only loader cannot decode pickle BINFLOAT. Keep
    # ordinary state_dict tensors readable, and serialize training metadata as
    # JSON plus tensor references to support safe loading on the paper's version.
    tensors = []

    def encode(value):
        if isinstance(value, torch.Tensor):
            tensors.append(value)
            return {"tensor": len(tensors) - 1}
        if isinstance(value, tuple):
            return {"tuple": [encode(item) for item in value]}
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            return {"dict": [[encode(key), encode(item)] for key, item in value.items()]}
        return value

    metadata = {key: value for key, value in checkpoint.items() if key != "model"}
    payload = {
        "d4rnet_format": "1",
        "metadata_json": json.dumps(encode(metadata), allow_nan=False),
        "training_tensors": tensors,
    }
    if "model" in checkpoint:
        payload["model"] = checkpoint["model"]
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    name, values, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state(
        (name, np.asarray(values, dtype=np.uint32), position, has_gauss, cached_gaussian)
    )
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_training_arguments(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        raise ValueError("epochs, batch_size and lr must be positive; workers nonnegative")
    if args.image_size != 224:
        raise ValueError("Published training uses 224x224 inputs; image_size must be 224")
