#!/usr/bin/env python3
"""Rebuild the paper splits and deterministic domain labels from upstream lists.

The supplied data-splits directory already contains the exact research manifests.
This tool offers an alternative that starts with the upstream traditional splits.
It never samples, shuffles, filters images, or reassigns split membership.
"""

import argparse
import hashlib
from pathlib import Path

from d4rnet.data import Sample, dataset_specs, domain_key, read_manifest, sample_fingerprint
from d4rnet.runtime import write_json


def write_manifest(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for sample in samples:
            domain = f"\t{sample.domain}" if sample.domain >= 0 else ""
            handle.write(f"{sample.manifest_id}\t{sample.path}\t{sample.label}{domain}\n")


def merge(parts):
    records = []
    for prefix, samples in parts:
        for sample in samples:
            records.append(
                Sample(len(records), (Path(prefix) / sample.path).as_posix(), sample.label)
            )
    return records


def build(source, output):
    specs = dataset_specs()
    if (output / "Traditional-setting").exists():
        raise FileExistsError(
            f"Refusing to overwrite {output}/Traditional-setting; choose an empty output"
        )
    # Read and validate every source file before creating output manifests.
    all_samples = {}
    for name, spec in specs.items():
        if len(name) == 2 and name[1].isdigit():
            for split in ("train", "val", "test"):
                records = read_manifest(source / spec[split])
                all_samples[name, split] = [Sample(s.manifest_id, s.path, s.label) for s in records]
    for split in ("train", "val", "test"):
        for modality, prefix in [("D", "Day"), ("N", "Night")]:
            all_samples[f"{modality}-all", split] = merge(
                [
                    (f"Cam{camera}", all_samples[f"{modality}{camera}", split])
                    for camera in range(1, 5)
                ]
            )
        for camera in range(1, 5):
            all_samples[f"DN{camera}", split] = merge(
                [
                    (f"Day/Cam{camera}", all_samples[f"D{camera}", split]),
                    (f"Night/Cam{camera}", all_samples[f"N{camera}", split]),
                ]
            )
        all_samples["DN-all", split] = merge(
            [
                ("Day", all_samples["D-all", split]),
                ("Night", all_samples["N-all", split]),
            ]
        )
    report, checksums = {}, {}
    for name, spec in specs.items():
        report[name] = {"counts": {}, "fingerprints": {}}
        for split in ("train", "val", "test"):
            records = all_samples[name, split]
            write_manifest(output / spec[split], records)
            report[name]["counts"][split] = len(records)
            report[name]["fingerprints"][split] = sample_fingerprint(records)
        training = all_samples[name, "train"]
        keys = sorted({domain_key(sample.path) for sample in training})
        mapping = {key: index for index, key in enumerate(keys)}
        auxiliary = [
            Sample(s.manifest_id, s.path, s.label, mapping[domain_key(s.path)]) for s in training
        ]
        write_manifest(output / spec["auxiliary_train"], auxiliary)
        report[name]["domain_mapping"] = mapping
        for field in ("train", "val", "test", "auxiliary_train"):
            path = output / spec[field]
            checksums[spec[field]] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "checksums.json", checksums)
    write_json(output / "preparation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-splits",
        required=True,
        help="Upstream data-splits directory, containing Traditional-setting/",
    )
    parser.add_argument("--output", required=True, help="New empty output directory")
    args = parser.parse_args()
    report = build(Path(args.source_splits), Path(args.output))
    for name, value in report.items():
        print(f"{name:6s}: {value['counts']}; domains={len(value['domain_mapping'])}")
    print("Prepared 15 splits without modifying source membership or order.")


if __name__ == "__main__":
    main()
