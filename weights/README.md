# Model downloads

Google Drive download links will be added by the authors after uploading the
selected teacher and student checkpoints. No checkpoints are bundled in Git.

| Artifact | Google Drive | Local location |
| --- | --- | --- |
| MobileNetV2 students for 15 splits | Pending author upload | `weights/student/` |
| MobileNetV3 ablation student | Pending author upload | `weights/student/` |
| Swin-B teachers | Pending author upload | `weights/teacher/` |

After upload, replace each `null` URL in [downloads.json](downloads.json) with
the corresponding Google Drive share link and update the table above. Files can
be downloaded manually; no access token or private author path is required.

Original student state dictionaries are supported. For the original
`mobilenet_v3_large` checkpoints, pass `--backbone mobilenet_v3_large`.
Legacy filenames containing `mobilenetv2_100` can still contain the torchvision
MobileNetV2 architecture used by this release; the filename does not select the
architecture. Checkpoint compatibility does not certify which published metric a
particular checkpoint reproduces; the author-uploaded model list should identify
that mapping.

New training checkpoints expose their inference state dictionary in the `model`
field and include resumable training metadata. Use
`d4rnet.runtime.load_torch()` to read the complete state on PyTorch 1.13.
