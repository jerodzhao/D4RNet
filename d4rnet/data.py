"""100-Driver manifests, deterministic preprocessing, and offline logit alignment.

Manifest loading and normalization are adapted from 100-Driver-Source
(Copyright 2022 Wenjing Li; MIT license, retained in THIRD_PARTY_NOTICES.md).
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASS_NAMES = (
    "Normal Driving",
    "Sleeping",
    "Yawning",
    "Calling (left)",
    "Calling (right)",
    "Texting (left)",
    "Texting (right)",
    "Make Up",
    "Looking Left",
    "Looking Right",
    "Looking Up",
    "Looking Down",
    "Smoking (left)",
    "Smoking (right)",
    "Smoking (mouth)",
    "Drinking / Eating (left)",
    "Drinking / Eating (right)",
    "Adjusting Radio",
    "Operating GPS",
    "Reaching Behind",
    "Hands off the Steering Wheel",
    "Talking to Passenger",
)


def dataset_specs():
    with (PROJECT_ROOT / "configs" / "datasets.json").open() as handle:
        return json.load(handle)


def preprocessing(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


@dataclass(frozen=True)
class Sample:
    manifest_id: int
    path: str
    label: int
    domain: int = -1


def read_manifest(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) not in {3, 4}:
                raise ValueError(f"{path}:{number}: expected id, path, class[, domain]")
            sample = Sample(
                int(fields[0]),
                fields[1],
                int(fields[2]),
                int(fields[3]) if len(fields) == 4 else -1,
            )
            relative = Path(sample.path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{path}:{number}: image paths must be relative")
            if not 0 <= sample.label < len(CLASS_NAMES):
                raise ValueError(f"{path}:{number}: class index outside [0, 21]")
            records.append(sample)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def sample_fingerprint(samples):
    """Ignore manifest IDs/domain columns; preserve sample order, path and label."""
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(f"{sample.path}\t{sample.label}\n".encode())
    return digest.hexdigest()


def domain_key(relative_path):
    """Combine available vehicle, scene, camera and modality, excluding behavior."""
    path = Path(relative_path)
    vehicle = re.search(r"_V(\d+)_", path.name)
    scene = re.search(r"_S(\d+)_", path.name)
    if vehicle is None or scene is None:
        raise ValueError(f"Cannot extract vehicle/scene from {relative_path}")
    parts = [f"V{vehicle.group(1)}", f"S{scene.group(1)}"]
    parts.extend(part for part in path.parts if re.fullmatch(r"Cam[1-4]", part))
    parts.extend(part for part in path.parts if part in {"Day", "Night"})
    return "_".join(parts)


class DriverDataset(Dataset):
    def __init__(
        self, data_root, dataset, split="train", auxiliary=False, splits_root=None, image_size=224
    ):
        specs = dataset_specs()
        if dataset not in specs:
            raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(specs)}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.spec = specs[dataset]
        self.dataset_id = dataset
        self.root = Path(data_root) / self.spec["image_subdir"]
        self.image_size = image_size
        key = "auxiliary_train" if auxiliary and split == "train" else split
        self.manifest = Path(splits_root or PROJECT_ROOT / "data-splits") / self.spec[key]
        self.samples = read_manifest(self.manifest)
        self.fingerprint = sample_fingerprint(self.samples)
        self.transform = preprocessing(image_size)
        if auxiliary:
            if split != "train":
                raise ValueError("Auxiliary labels are defined for the training split")
            domains = {sample.domain for sample in self.samples}
            if domains != set(range(self.spec["num_domains"])):
                raise ValueError(f"Domain labels do not match {dataset} metadata")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        with Image.open(self.root / record.path) as image:
            image = self.transform(image.convert("RGB"))
        return {"image": image, "label": record.label, "domain": record.domain, "index": index}


class TeacherLogits:
    """Validate a cache before indexing by zero-based dataset position.

    The first column of the split file is NOT the dataset position. Aggregated
    splits contain repeated IDs; original caches use Dataset.__getitem__ indices.
    """

    def __init__(self, path, dataset, allow_legacy=False):
        from .runtime import load_torch

        cache = load_torch(path)
        if not isinstance(cache, dict) or not {"logits", "labels", "indices"} <= cache.keys():
            raise ValueError("A logit cache must contain logits, labels and indices")
        indices = cache["indices"].long().cpu()
        labels = cache["labels"].long().cpu()
        logits = cache["logits"].float().cpu()
        count = len(dataset)
        if indices.shape != (count,) or labels.shape != (count,) or logits.shape != (count, 22):
            raise ValueError("Teacher cache shapes do not match this training split")
        if not torch.equal(indices.sort().values, torch.arange(count)):
            raise ValueError("Teacher indices must cover every dataset position exactly once")
        order = indices.argsort()
        if not torch.equal(labels[order], torch.tensor([s.label for s in dataset.samples])):
            raise ValueError("Teacher labels and training sample order do not match")
        fingerprint = cache.get("sample_fingerprint")
        if fingerprint is None:
            if not allow_legacy:
                raise ValueError(
                    "Legacy cache has no sample fingerprint; regenerate it or use "
                    "--allow-legacy-logits after verifying its provenance"
                )
        elif fingerprint != dataset.fingerprint:
            raise ValueError("Teacher cache belongs to a different manifest or sample order")
        if cache.get("image_size", dataset.image_size) != dataset.image_size:
            raise ValueError("Teacher and student image sizes must match")
        if not torch.isfinite(logits).all():
            raise ValueError("Teacher logits contain NaN or infinity")
        self.logits = logits[order].contiguous()

    def __getitem__(self, indices):
        return self.logits[indices.cpu()]
