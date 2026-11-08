"""Sample-weighted losses, Top-k accuracy and macro classification metrics."""

import torch
from torch.nn import functional as F


class ClassificationMetrics:
    def __init__(self, num_classes=22):
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.top5 = 0
        self.loss_sum = 0.0

    def update(self, logits, labels):
        logits, labels = logits.detach().cpu(), labels.detach().cpu().long()
        classes = self.confusion.shape[0]
        predictions = logits.argmax(1)
        self.confusion += torch.bincount(
            labels * classes + predictions, minlength=classes**2
        ).reshape(classes, classes)
        self.top5 += (
            logits.topk(min(5, classes), dim=1).indices.eq(labels[:, None]).any(1).sum().item()
        )
        self.loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()

    def compute(self):
        matrix = self.confusion.double()
        total = int(matrix.sum())
        if total == 0:
            raise ValueError("Cannot evaluate an empty dataset")
        correct = matrix.diag()
        support = matrix.sum(1)
        precision = correct / matrix.sum(0).clamp_min(1)
        recall = correct / support.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-15)
        return {
            "samples": total,
            "loss": self.loss_sum / total,
            "top1": 100 * correct.sum().item() / total,
            "top5": 100 * self.top5 / total,
            "precision_macro": 100 * precision.mean().item(),
            "recall_macro": 100 * recall.mean().item(),
            "f1_macro": 100 * f1.mean().item(),
            "per_class_accuracy": [
                100 * recall[i].item() if support[i] else None for i in range(len(support))
            ],
            "support": support.long().tolist(),
            "confusion_matrix": self.confusion.tolist(),
        }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    metrics = ClassificationMetrics()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        metrics.update(model(images), batch["label"])
    return metrics.compute()
