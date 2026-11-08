# Reproducing D4RNet

## Architecture and objective

The primary implementation follows Sections 3.1–3.4 of the published paper and
the final research script `train_distill_final.py`.

| Component | Implementation |
| --- | --- |
| Student | torchvision MobileNetV2 |
| Auxiliary taps | Feature indices 3, 6, 13; pooled widths 24, 32, 96 |
| Behavior feature | Feature index 18; pooled width 1,280 |
| Behavior head | Linear(1280,128), ReLU, Dropout(0.3), Linear(128,22) |
| Domain projector | Linear(152,256), ReLU, Dropout(0.2), Linear(256,128), ReLU, Dropout(0.2) |
| Domain prediction | GRL after projector, followed by Linear(128,D_dom) |
| Teacher | timm Swin-B, `swin_base_patch4_window7_224` |
| KD | `(1-beta1)*CE(student,y) + beta1*T^2*KL(teacher || student)` |
| Total loss | KD loss plus `beta2*CE(domain_logits,domain_id)` |
| Domains | Available modality, camera, vehicle, and scene attributes; no behavior class |

The PyTorch `KLDivLoss(log_softmax(student/T), softmax(teacher/T))` direction is
retained from the experiment code. The ordering of the arguments in the paper's
Eq. (2) is schematic; this release documents the actual implemented direction.
The primary head receives the final feature only. Intermediate features serve
the auxiliary branch. The projector precedes GRL, so reversal affects the
projector and the backbone, while the domain classifier receives ordinary
classification gradients.

MobileNetV3 uses feature indices 3, 6, and 13 with widths 24, 40, and 160;
the final feature has width 960. Its primary head retains the same hidden width.

## Two GRL policies

The paper's Algorithm 1 and its accompanying prose do not describe identical
adaptive thresholds. To preserve the experiment behavior while allowing the
pseudocode to be studied, the policy is an explicit option.

| Behavior | `legacy` (default) | `algorithm1` |
| --- | --- | --- |
| First three epochs | Ordinary auxiliary backpropagation, no reversal | Same |
| Initial reversal strength | 1e-4, reset at every later epoch | 1e-4 when adversarial training starts |
| Accuracy statistic | Current batch accuracy | EMA: `0.9*previous + 0.1*batch`, initialized to zero |
| Enable/increase condition | Accuracy > 0.1 | EMA > 1/D_dom |
| Disable/decrease condition | Accuracy < 1/D_dom, after checking the first branch | EMA < 1/D_dom |
| Strength update | Increase ×1.05, capped at 3e-4; decrease ×0.1, floored at 1e-6 | Same |
| Across epochs | Reset enabled state and strength | Preserve state after warm-up |

Updates affect the following batch. When reversal is disabled, auxiliary
gradients propagate normally; a zero gradient multiplier would change the
method. The branch order in the legacy policy matters for subsets with fewer
than 10 domains and is intentionally preserved.

## Preprocessing and offline targets

Teacher training, teacher caching, and student training use RGB conversion,
bilinear resize to 224 × 224, and ImageNet normalization. This implements
Section 4.3's no-augmentation protocol. Some exploratory research data loaders
omitted resizing, or used random rotations/erasing for teachers and baseline
experiments. Those alternate transforms are not included in the release.

A new logit cache fingerprints every image path and behavior label, in order.
The auxiliary column and manifest ID are excluded so that ordinary and
auxiliary versions of the same training list can share logits. Cached indices
are zero-based dataset positions, not manifest IDs. The loader checks index
coverage, labels, image size, finite values, and the fingerprint.

Original caches containing only `logits`, `labels`, and `indices` require
`--allow-legacy-logits`. This checks shapes, positions and labels, but cannot
prove image-path or preprocessing equivalence. Regenerating caches with the
supplied script is preferred for a new run.

## Splits and labels

All 60 archived research manifests are supplied unchanged, including auxiliary
domain IDs. The 15 splits match Table 2 in train/validation/test row counts and
domain counts. Combined manifests preserve the source split assignments and
concatenation order. The preparation tool rebuilds these combinations without
resampling. It does not inspect or filter images.

Repeated rows in a source manifest are retained, including during evaluation.
Removing them would alter the published population and Table 2 counts.
SHA-256 checksums identify the archived files. The verification tool checks
image-path disjointness between the three splits, exact combined sequences,
and a one-to-one mapping between domain IDs and nuisance attribute combinations.

## Training and checkpoint selection

The student uses the supplied final script's AdamW settings, warm-up, LR
milestones and beta1 decay. Epochs are one-based: beta1 is 0.7 through epoch 10,
0.66 at epoch 11, and is clamped at 0.1. LR is updated before each optimizer
step. The original warm-up starts with LR zero; drops occur at epochs 10 and 20.

Checkpoint selection uses the best validation Top-1 across all epochs. The
research script restricted automatically named "best" checkpoints to epochs
after 20, while also saving intermediate regular checkpoints; the paper notes
that the best validation epoch often falls between 10 and 20. This release
always saves `best.pth` and `last.pth`, making the selection criterion explicit.
The test set is not used to select checkpoints.

The teacher preset is taken from the retained Swin-B launch script: 15 epochs,
batch size 32 per process, AdamW at 1e-4 with 1e-4 weight decay, and cosine LR.
The general teacher training source also supported 30 epochs; use `--epochs 30`
when appropriate for a particular teacher. This release does not infer an exact
teacher run from a checkpoint filename. In multi-process training, validation
uses unpadded disjoint shards and globally reduced totals. Training uses the
standard distributed sampler, which can pad the last shard for equal work.

Baseline switches reuse the same final model head and no-augmentation pipeline.
The original exploratory "baseline" identifier selected SGD with momentum 0.9
and weight decay 5e-4; those options remain available explicitly. This release
does not claim that changing optimizer and preprocessing preserves previously
reported baseline numbers without retraining.

Student continuation saves model, optimizer, GRL, loader generator, Python,
NumPy, and PyTorch random states. It supports resume at an epoch boundary.
Changing the training manifest or optimization settings is rejected. For an
exact continuation, retain batch size, device type, worker setup and dependency
versions. The teacher script saves inference checkpoints; it does not implement
resumable teacher optimization.

## Metrics and reported results

Top-1 is sample-level behavior accuracy. Top-5 counts whether the correct class
is among the five largest logits. Macro precision, recall, and F1 average over
the fixed 22-class taxonomy, assigning zero where a denominator is zero.
Per-category accuracy is recall within that behavior class. Confusion matrices
use ground-truth rows and predicted columns. Losses are sample-weighted.

The paper's 81.32% number is the arithmetic mean of the DN-all student's test
accuracies on D1–D4 and N1–N4 (Table 4). It is distinct from DN-all's overall
Top-1, which is reported as 81.33% in the ablation table. The release reports
computed metrics directly; it does not substitute published values.

## Deployment

The deployment model has no auxiliary parameters and produces raw 22-class
logits. Export defaults to opset 16 and fixed 224 × 224 spatial dimensions,
with optional dynamic batch size. ONNX Runtime verification compares actual
logits rather than only the predicted class.

The published Jetson Nano timing uses `TensorrtExecutionProvider`; CPU and CUDA
providers are different execution conditions. The benchmark excludes image
preprocessing and video I/O. No Jetson hardware performance is inferred from
CPU validation. Video prediction is frame-by-frame without an additional
temporal network or smoothing model.
