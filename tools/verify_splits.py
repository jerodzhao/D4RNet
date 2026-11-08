#!/usr/bin/env python3
"""Verify manifest integrity, Table 2 counts, domain labels, and combined splits."""

import argparse
import hashlib
import json
from pathlib import Path

from d4rnet.data import (
    PROJECT_ROOT,
    Sample,
    dataset_specs,
    domain_key,
    read_manifest,
    sample_fingerprint,
)
from d4rnet.runtime import write_json


def audit(root, data_root=None, checksums=True):
    specs = dataset_specs()
    expected_hashes = json.loads((root / "checksums.json").read_text()) if checksums else {}
    records = {}
    report = {}
    for name, spec in specs.items():
        record = {
            "counts": {},
            "num_domains": spec["num_domains"],
            "fingerprints": {},
            "repeated_image_rows": {},
        }
        paths = []
        for split in ("train", "val", "test", "auxiliary_train"):
            path = root / spec[split]
            if (
                checksums
                and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hashes[spec[split]]
            ):
                raise ValueError(f"Checksum mismatch: {path}")
            samples = read_manifest(path)
            records[name, split] = samples
            canonical_split = "train" if split == "auxiliary_train" else split
            if len(samples) != spec["counts"][canonical_split]:
                raise ValueError(f"Table 2 count mismatch: {name}/{split}")
            if split != "auxiliary_train":
                canonical = {(Path(spec["image_subdir"]) / s.path).as_posix() for s in samples}
                # Preserve repeated rows in the research manifests: removing them
                # changes the published evaluation population and sample counts.
                record["repeated_image_rows"][split] = len(samples) - len(canonical)
                paths.append(canonical)
                record["counts"][split] = len(samples)
                record["fingerprints"][split] = sample_fingerprint(samples)
                if data_root:
                    missing = [
                        s.path
                        for s in samples
                        if not (data_root / spec["image_subdir"] / s.path).is_file()
                    ]
                    if missing:
                        raise FileNotFoundError(
                            f"{name}/{split}: {len(missing)} missing images; first: {missing[0]}"
                        )
        if any(paths[i] & paths[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError(f"Image overlap between splits: {name}")
        training, auxiliary = records[name, "train"], records[name, "auxiliary_train"]
        if sample_fingerprint(training) != sample_fingerprint(auxiliary):
            raise ValueError(f"Auxiliary labels changed sample membership/order: {name}")
        domain_to_key, key_to_domain = {}, {}
        for sample in auxiliary:
            key = domain_key(sample.path)
            if (
                domain_to_key.setdefault(sample.domain, key) != key
                or key_to_domain.setdefault(key, sample.domain) != sample.domain
            ):
                raise ValueError(f"Inconsistent domain mapping: {name}")
        if set(domain_to_key) != set(range(spec["num_domains"])):
            raise ValueError(f"Domain count mismatch: {name}")
        record["domain_mapping"] = key_to_domain
        report[name] = record
    for split in ("train", "val", "test"):

        def merged(parts):
            return [
                Sample(0, (Path(prefix) / s.path).as_posix(), s.label)
                for prefix, name in parts
                for s in records[name, split]
            ]

        combinations = {
            "D-all": [(f"Cam{i}", f"D{i}") for i in range(1, 5)],
            "N-all": [(f"Cam{i}", f"N{i}") for i in range(1, 5)],
            "DN-all": [("Day", "D-all"), ("Night", "N-all")],
            **{
                f"DN{i}": [(f"Day/Cam{i}", f"D{i}"), (f"Night/Cam{i}", f"N{i}")]
                for i in range(1, 5)
            },
        }
        for name, parts in combinations.items():
            if sample_fingerprint(merged(parts)) != sample_fingerprint(records[name, split]):
                raise ValueError(
                    f"Combined split does not preserve source sequence: {name}/{split}"
                )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-root", type=Path, default=PROJECT_ROOT / "data-splits")
    parser.add_argument(
        "--data-root", type=Path, help="Also check image file existence, without opening images"
    )
    parser.add_argument("--checksums", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", help="Optional JSON report")
    args = parser.parse_args()
    report = audit(args.splits_root, args.data_root, args.checksums)
    for name, value in report.items():
        print(f"{name:6s}: {value['counts']}; domains={value['num_domains']}; OK")
    if args.output:
        write_json(args.output, report)
    print("Verified all 60 manifests, domain mappings and aggregate sample sequences.")


if __name__ == "__main__":
    main()
