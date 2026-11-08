# Release validation

Validation was performed on CPU using the author's existing `100driver`
environment: Python 3.10.16, PyTorch 1.13.0, torchvision 0.14.0, timm 1.0.13,
NumPy 1.25.0, Pillow 11.0.0, and ONNX 1.20.1. CUDA was unavailable to the
validation process. The existing environment was not modified.

ONNX Runtime was absent from that environment, so runtime equivalence was checked
separately in an isolated temporary environment using Python 3.11, PyTorch 2.0.1,
torchvision 0.15.2, ONNX 1.17.0, and ONNX Runtime 1.20.1. A model exported by
PyTorch 1.13 was also compared directly with its PyTorch 1.13 reference logits
using ONNX Runtime CPU inference.

## Completed checks

- **14 core behavioral tests:** gradient reversal sign and magnitude; legacy and
  Algorithm 1 adaptive policies; KD direction, temperature factor, and teacher
  detachment; loss-weight schedule; RGB conversion and deterministic resizing;
  cache reordering and rejection of mismatched labels, indices and fingerprints;
  macro metrics; safe RNG-state checkpoint round trip; model output shapes and
  auxiliary backward pass, including ordinary gradients when reversal is
  disabled and unchanged domain-classifier gradients when enabled; strict
  deployment checkpoint loading.
- **Archived data manifests:** all 60 SHA-256 checksums, all 15 Table 2 sample
  and domain counts, exact source-to-combination ordering, behavior-label
  alignment with auxiliary lists, one-to-one domain mappings, and image-path
  disjointness between train/validation/test lists passed. The generation tool
  rebuilt all combined splits and domain labels, and the regenerated lists
  passed the same semantic checks.
- **Existing checkpoint compatibility:** an original Swin-B teacher and original
  MobileNetV2/MobileNetV3 students loaded strictly and returned finite 22-class
  logits. MobileNetV2 and V3 inference logits were exactly equal, element by
  element, to the original final model implementation using the same weights
  and inputs. This checks architecture compatibility, not which checkpoint
  matches a published table entry.
- **Synthetic end-to-end training:** a Swin-B teacher completed one epoch and
  generated a validated cache. A MobileNetV2 student completed four epochs with
  KD and GRL. Training for two epochs, saving, and resuming through epoch four
  produced exactly the same model tensors, GRL state and loader RNG state as an
  uninterrupted four-epoch run. Both GRL policies, KD-only, GRL-only, neither
  loss, and the MobileNetV3 comparison completed training runs.
- **Distributed teacher validation:** two CPU/Gloo processes completed a
  synthetic teacher run. Three validation samples were counted exactly once
  globally despite uneven shard sizes. No padded validation samples were added.
- **Evaluation and presentation:** teacher/student metrics, PyTorch and ONNX
  image/video inference, three-frame video annotation, Grad-CAM, the ONNX latency
  benchmark, and cross-split/per-category CSV output completed on generated
  fixtures.
- **ONNX export:** the exported graph passed the ONNX checker, omitted auxiliary
  parameters, and supported dynamic batches of size one and two. ONNX Runtime
  CPU outputs matched PyTorch within `rtol=1e-3, atol=1e-5`; the maximum absolute
  error observed for the legacy MobileNetV2 checkpoint was below `9e-6`.
- **Source quality:** Ruff checks, formatting checks, Python compilation, and
  local documentation-link checks passed.
- **Standalone directory:** a copy outside the research workspace passed all
  entry-point help checks, core tests, and manifest verification with no parent
  project in its Python path.

## Scope

These checks establish that the cleaned implementation and its workflows run
and preserve the checked model behavior. They do not constitute a fresh
reproduction of every published experiment. The full 30-epoch dataset training,
all 15 final-checkpoint test evaluations, GPU execution, TensorRT execution,
and Jetson Nano frame-rate measurements were not repeated during cleanup.

Google Drive links remain intentionally unset until the authors upload the
selected model files. No dataset images, cached teacher outputs, checkpoints,
private paths, or experiment logs are bundled in this release.

## Repeatable checks

From the release root, with the reference dependencies installed:

```bash
python -m unittest discover -s tests -v
python -m tools.verify_splits
python -m tools.prepare_splits \
  --source-splits /path/to/upstream/data-splits \
  --output /tmp/d4rnet-splits-regenerated
python -m tools.verify_splits --splits-root /tmp/d4rnet-splits-regenerated
```

The tests generate their own temporary images and cache data. Dataset
verification reads text manifests; add `--data-root` only when image-file
existence should also be checked.
