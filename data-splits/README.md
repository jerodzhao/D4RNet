# Paper split manifests

`Traditional-setting/` contains the exact 60 research manifests for the 15
dataset IDs in `configs/datasets.json`. `checksums.json` records each archived
file's SHA-256 digest. `domain_mappings.json` records the original auxiliary-ID
mapping extracted and checked against the archived training labels.

Each line is whitespace-separated:

```text
manifest_id    image_path    class_index    [domain_index]
```

`class_index` is in 0–21. The fourth column exists in auxiliary training lists.
Image paths are relative to the split's `image_subdir` in `datasets.json`,
which is itself relative to the common 100-Driver image root.

The first column is retained for provenance. It is not guaranteed to be unique
in aggregate splits. Teacher caches use zero-based positions in the full list.
Original repeated rows and source assignments are retained.

Traditional manifests originate from
[100-Driver-Source](https://github.com/Shenqishaonv/100-Driver-Source).
D4RNet supplies day/night combinations and auxiliary domain columns. See
`THIRD_PARTY_NOTICES.md` for the retained upstream license and acknowledgment.

Run `python -m tools.verify_splits` from the repository root to check the
archived lists. Rebuild into a separate directory using
`python -m tools.prepare_splits --help`.
