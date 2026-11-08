# Third-party acknowledgments

## 100-Driver dataset and source code

This project builds on [100-Driver-Source](https://github.com/Shenqishaonv/100-Driver-Source),
provided by the authors of **100-Driver: A Large-scale, Diverse Dataset for
Distracted Driver Classification**. We thank Jing Wang, Wenjing Li, Fang Li,
Jun Zhang, Zhongcheng Wu, Zhun Zhong, and Nicu Sebe for their dataset and code.

The traditional split manifests, class ordering, dataset-loading conventions,
normalization pipeline, and parts of the training/evaluation workflow originate
from that repository. D4RNet adds combined day/night splits, auxiliary domain
labels, offline knowledge distillation, and domain-adversarial training.

The upstream software is distributed under the following MIT license. Dataset
access follows the separate licensing procedure on the
[100-Driver project website](https://100-driver.github.io/); this software license
does not replace the dataset agreement. Dataset images are not redistributed here.

```text
MIT License

Copyright (c) 2022 Wenjing Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Frameworks

- [PyTorch](https://github.com/pytorch/pytorch) and
  [torchvision](https://github.com/pytorch/vision) provide the student backbones
  and ImageNet initialization.
- [timm](https://github.com/huggingface/pytorch-image-models) provides Swin-B.
- [ONNX](https://github.com/onnx/onnx) and
  [ONNX Runtime](https://github.com/microsoft/onnxruntime) support deployment.

These dependencies retain their respective licenses. Pretrained initialization
weights and externally distributed trained weights may carry their own terms.
