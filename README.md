# D4RNet

Official implementation of **D4RNet: A lightweight distracted driver detection
network with knowledge distillation and gradient reversal layer**.

Xiong Zhao, Zhuofan Yang, Sarina Sulaiman, Wong Yee Leng, and Xingyuan Li.  
*Expert Systems with Applications*, **334** (2027), 134357.  
[Paper](https://doi.org/10.1016/j.eswa.2026.134357) ·
[Code](https://github.com/jerodzhao/D4RNet) · [Model downloads](weights/README.md)

D4RNet trains a MobileNetV2 student using offline Swin-B logits and an auxiliary
domain classifier connected through a gradient reversal layer (GRL). Features
from layers 3, 6, and 13 supervise the domain branch; layer 18 supplies the
22-class behavior head. The teacher and auxiliary branch are unnecessary at
inference time. MobileNetV3 is included for the paper's ablation experiments.

The paper reports **81.32% mean Top-1 accuracy across eight camera/modality
subsets when using the DN-all model**, and **52 FPS on Jetson Nano** using ONNX
Runtime with TensorRT. These are published results; runtime depends on hardware,
input size, execution provider, and software versions. See
[reproduction details](docs/REPRODUCIBILITY.md) and [validation](docs/VALIDATION.md).

## Installation

The reference environment uses Python 3.10, PyTorch 1.13.0, torchvision 0.14.0,
and timm 1.0.13. Install a matching PyTorch/torchvision build for your platform
before installing the remaining dependencies.

```bash
git clone https://github.com/jerodzhao/D4RNet.git
cd D4RNet
conda create -n d4rnet python=3.10 -y
conda activate d4rnet
python -m pip install -r requirements.txt
```

For ONNX export, ONNX Runtime inference, and video annotation:

```bash
python -m pip install -r requirements-deploy.txt
```

For NVIDIA devices, install an ONNX Runtime build that supports your CUDA,
TensorRT, and JetPack versions. Install the appropriate GPU package instead of
the CPU `onnxruntime` package; avoid installing both in the same environment.
The CPU wheel does not provide TensorRT. TensorBoard is optional:
`python -m pip install tensorboard`.

All commands below run from the repository root. Core tests run on CPU; CUDA is
selected automatically when available, or use `--device cpu` explicitly. ImageNet
initialization is downloaded by torchvision/timm during training unless
`--no-pretrained` is supplied. Evaluation never downloads initialization weights.

## Dataset

We use [100-Driver](https://100-driver.github.io/), following the access and
preparation procedure in
[100-Driver-Source](https://github.com/Shenqishaonv/100-Driver-Source).
Complete the official dataset licensing form and request the download link from
the dataset providers. Download and extract the data into:

```text
data/100-driver/
├── Day/
│   ├── Cam1/
│   │   └── C1_Drive_Safe/...
│   ├── Cam2/...
│   ├── Cam3/...
│   └── Cam4/...
└── Night/
    ├── Cam1/...
    ├── Cam2/...
    ├── Cam3/...
    └── Cam4/...
```

Dataset images are not included. `--data-root /path/to/100-driver` can point to
an existing extraction without copying it. Images are converted to RGB, resized
to 224 × 224, and normalized with ImageNet mean/std. No data augmentation is used
in teacher training, logit caching, or student training.

The exact research manifests and auxiliary labels are included under
`data-splits/Traditional-setting/`. D1–D4 and N1–N4 follow the upstream traditional
splits. D-all and N-all combine camera views within each modality. DN1–DN4
combine day/night data for the corresponding camera, and DN-all combines all
views and modalities. The source train/validation/test assignments and row order
are retained. Class labels are zero-based (0–21), corresponding to C1–C22.

| Split | Train | Validation | Test | Auxiliary domains |
| --- | ---: | ---: | ---: | ---: |
| D1 | 45,388 | 4,334 | 11,196 | 9 |
| D2 | 42,959 | 3,810 | 10,890 | 10 |
| D3 | 46,131 | 3,953 | 11,373 | 9 |
| D4 | 48,606 | 4,937 | 11,689 | 14 |
| D-all | 183,084 | 17,034 | 45,148 | 42 |
| N1 | 45,239 | 5,563 | 10,351 | 6 |
| N2 | 41,897 | 5,561 | 10,598 | 5 |
| N3 | 31,902 | 4,200 | 7,484 | 6 |
| N4 | 45,305 | 6,146 | 10,696 | 6 |
| N-all | 164,343 | 21,470 | 39,129 | 23 |
| DN1 | 90,627 | 9,897 | 21,547 | 15 |
| DN2 | 84,856 | 9,371 | 21,488 | 15 |
| DN3 | 78,033 | 8,153 | 18,857 | 15 |
| DN4 | 93,911 | 11,083 | 22,385 | 20 |
| DN-all | 347,427 | 38,504 | 84,277 | 65 |

Verify the bundled manifests, including aggregate sample order and domain maps:

```bash
python -m tools.verify_splits
# Optionally check image file existence without opening images:
python -m tools.verify_splits --data-root /path/to/100-driver
```

To reconstruct the combined splits and domain labels from an upstream checkout:

```bash
python -m tools.prepare_splits \
  --source-splits /path/to/100-Driver-Source/data-splits \
  --output data-splits-regenerated
python -m tools.verify_splits --splits-root data-splits-regenerated
```

Regeneration preserves image membership and sequence. Newly generated aggregate
row IDs may differ from archived IDs; neither training nor caching uses that
column as a sample index. Pass `--splits-root data-splits-regenerated` to use the
regenerated lists.

## Training

Train a separate teacher/student pair for each desired split. Replace `D4` with
any split ID in the table; use separate output paths and logit files for each.

### 1. Fine-tune the teacher

```bash
python train_teacher.py --dataset D4 --output runs/teacher/D4
```

The teacher preset follows the retained Swin-B training script: AdamW,
learning rate 1e-4, weight decay 1e-4, 15 epochs, cosine schedule, and batch size
32 per process. Change `--epochs` or `--batch-size` as needed. For four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train_teacher.py \
  --dataset D4 --output runs/teacher/D4
```

### 2. Cache teacher logits

```bash
python cache_teacher.py --dataset D4 \
  --weights runs/teacher/D4/best.pth \
  --output teacher_logits/D4_train.pt
```

The cache records raw logits, labels, zero-based dataset positions, and a
fingerprint of image paths and labels in their exact order. Student training
checks this metadata before using the cache. Caching is deterministic and uses
the same preprocessing as student training.

### 3. Train the student

```bash
python train_student.py --dataset D4 \
  --teacher-logits teacher_logits/D4_train.pt \
  --output runs/student/D4
```

The default settings in [configs/d4rnet.json](configs/d4rnet.json) are:
MobileNetV2, batch size 128, 30 epochs, seed 42, AdamW with learning rate 0.002
and weight decay 1e-4, three warm-up epochs, and LR drops at epochs 10 and 20
(factor 0.2). The temperature is 4.0; the initial KD weight is 0.7 and the
auxiliary weight is 0.2. The KD weight decays by 0.04 per epoch after epoch 10,
down to 0.1. CLI options override JSON settings.

Two GRL policies are supported. **`--grl-mode legacy` is the default** and retains
the existing experiment code's batch-accuracy thresholds and per-epoch strength
reset. `--grl-mode algorithm1` implements the paper pseudocode's accuracy moving
average and chance-level threshold. Both disable reversal during the first three
epochs while allowing ordinary auxiliary gradients. Their difference is
documented in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

`best.pth` is selected using validation accuracy throughout the run;
`last.pth` includes optimizer, GRL, loader, and random states for continuation:

```bash
python train_student.py --dataset D4 \
  --teacher-logits teacher_logits/D4_train.pt \
  --output runs/student/D4 --resume runs/student/D4/last.pth
```

Add `--tensorboard` to record TensorBoard scalars. Configuration and per-epoch
metrics are always saved as `config.json` and `metrics.jsonl`.

### Ablations

Use the same split, data, preprocessing, and seed. These options are explicit;
the experiment directory name does not alter training behavior.

| Experiment | Additional student options |
| --- | --- |
| Full D4RNet | Default |
| KD only | `--no-grl` |
| GRL only | `--no-kd` |
| Neither KD nor GRL | `--no-kd --no-grl` |
| MobileNetV3 comparison | `--backbone mobilenet_v3_large` |
| Original SGD baseline optimizer | `--optimizer sgd --weight-decay 0.0005` |

With `--no-kd`, omit `--teacher-logits`. Use distinct `--output` directories for
each experiment. See the reproduction notes for differences between the
original exploratory baseline and the standardized ablation preprocessing.

## Evaluation

```bash
python evaluate.py --dataset D4 --weights runs/student/D4/best.pth \
  --output results/D4.json
python evaluate.py --kind teacher --dataset D4 \
  --weights runs/teacher/D4/best.pth
```

Reports include Top-1/Top-5 accuracy, macro precision/recall/F1, per-class recall,
support, and a 22 × 22 confusion matrix. Percentages use the complete selected
split. Image loading is ordered and unshuffled.

To evaluate one DN-all model on all eight camera/modality test subsets:

```bash
python evaluate.py --weights weights/student/DN-all.pth \
  --datasets D1 D2 D3 D4 N1 N2 N3 N4 --output results/DN-all_subsets.json
```

For the paper's cross-split tables, edit
[configs/weights_map.example.json](configs/weights_map.example.json) to list the
downloaded checkpoints. Paths in that file are relative to the JSON file:

```bash
python -m tools.cross_evaluate --weights-map configs/weights_map.example.json \
  --output results/cross_evaluation
```

This writes the cross-split Top-1 matrix and per-category accuracy tables as CSV,
plus complete metrics as JSON. Listing all 15 trained models yields a 15 × 15
cross-split matrix; the DN-all model yields its 15 × 22 per-category table.

## Inference and deployment

Download links will be added in [weights/README.md](weights/README.md) after the
authors upload the selected models to Google Drive. The repository reserves
`weights/student/` and `weights/teacher/` for local downloads.

```bash
python infer.py --weights weights/student/DN-all.pth --input example.jpg
python infer.py --weights weights/student/DN-all.pth \
  --input driving.mp4 --output results/annotated.mp4
python -m tools.gradcam --weights weights/student/DN-all.pth \
  --image example.jpg --output results/gradcam.png
```

Export a model with the training-only auxiliary branch removed:

```bash
python export_onnx.py --weights weights/student/DN-all.pth \
  --output exports/D4RNet.onnx --dynamic-batch
python infer.py --weights exports/D4RNet.onnx --input example.jpg
```

Export validates the ONNX graph and compares ONNX Runtime CPU logits with
PyTorch. Use `--no-check` only when ONNX Runtime is unavailable; graph validation
still runs. A JSON sidecar records class names and preprocessing. Alternate
spatial sizes can be exported with `--height 480 --width 856`.

On a device with a compatible TensorRT-enabled ONNX Runtime build:

```bash
python infer.py --weights exports/D4RNet.onnx --input driving.mp4 \
  --output results/annotated.mp4 \
  --providers TensorrtExecutionProvider CUDAExecutionProvider CPUExecutionProvider
python -m tools.benchmark --model exports/D4RNet.onnx \
  --providers TensorrtExecutionProvider CUDAExecutionProvider CPUExecutionProvider
```

The benchmark measures warmed-up ONNX Runtime calls, excluding preprocessing,
video decoding, and annotation. It prints the active providers, latency, and
throughput; compare runs only under matching settings.

## Tests

```bash
python -m unittest discover -s tests -v
python -m tools.verify_splits
```

Tests use generated fixtures and random initialization; no dataset or model
downloads are required. See [validation notes](docs/VALIDATION.md) for checks
completed during release preparation and their limits.

## Acknowledgments and license

We thank the **100-Driver authors and dataset providers** for their dataset,
evaluation splits, and public source code. Parts of this implementation were
adapted from [100-Driver-Source](https://github.com/Shenqishaonv/100-Driver-Source).
We also thank the developers of PyTorch, torchvision, timm, ONNX, and ONNX Runtime.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The code is released under the [MIT License](LICENSE), retaining the upstream
copyright notice. The 100-Driver dataset remains subject to its own access and
licensing agreement.

## Citation

The article appeared online in 2026 and is assigned to the journal's 2027 volume.
Please cite the final publication:

```bibtex
@article{zhao2027d4rnet,
  title   = {{D4RNet}: A lightweight distracted driver detection network with knowledge distillation and gradient reversal layer},
  author  = {Zhao, Xiong and Yang, Zhuofan and Sulaiman, Sarina and Wong, Yee Leng and Li, Xingyuan},
  journal = {Expert Systems with Applications},
  volume  = {334},
  pages   = {134357},
  year    = {2027},
  doi     = {10.1016/j.eswa.2026.134357}
}
```

Please also cite 100-Driver when using its data or splits (following the
upstream repository's citation):

```bibtex
@article{wang2023driver,
  title     = {{100-Driver}: A Large-scale, Diverse Dataset for Distracted Driver Classification},
  author    = {Wang, Jing and Li, Wenjing and Li, Fang and Zhang, Jun and Wu, Zhongcheng and Zhong, Zhun and Sebe, Nicu},
  journal   = {IEEE Transactions on Intelligent Transportation Systems},
  year      = {2023},
  publisher = {IEEE}
}
```
