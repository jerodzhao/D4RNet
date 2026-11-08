"""Offline knowledge distillation, Eqs. (2) and (5)."""

from torch.nn import functional as F


def distillation_weight(epoch, initial=0.7, decay_start=10, decay_step=0.04, minimum=0.1):
    """Use one-based epochs, matching the released training script."""
    return max(minimum, initial - decay_step * max(0, epoch - decay_start))


def classification_loss(student, labels, teacher=None, temperature=4.0, beta=0.7):
    """Return weighted hard and soft losses (teacher distribution is the target).

    PyTorch's kl_div(log_softmax(student), softmax(teacher)) computes
    KL(teacher || student), preserving the original training implementation.
    """
    hard = F.cross_entropy(student, labels)
    if teacher is None:
        return hard, student.new_zeros(())
    if temperature <= 0 or not 0 <= beta <= 1:
        raise ValueError("temperature must be positive and beta must be in [0, 1]")
    soft = (
        F.kl_div(
            F.log_softmax(student / temperature, dim=1),
            F.softmax(teacher.detach() / temperature, dim=1),
            reduction="batchmean",
        )
        * temperature**2
    )
    return (1 - beta) * hard, beta * soft
