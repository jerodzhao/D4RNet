"""Adaptive GRL policies: original experiments and paper Algorithm 1."""

from dataclasses import asdict, dataclass


@dataclass
class GRLController:
    num_domains: int
    mode: str = "legacy"
    warmup_epochs: int = 3
    initial: float = 1e-4
    maximum: float = 3e-4
    minimum: float = 1e-6
    strength: float = 0.0
    enabled: bool = False
    accuracy_ema: float = 0.0

    def __post_init__(self):
        if self.num_domains < 1 or self.mode not in {"legacy", "algorithm1"}:
            raise ValueError("Invalid domain count or GRL mode")

    def begin_epoch(self, epoch):
        if epoch <= self.warmup_epochs:
            self.enabled, self.strength = False, 0.0
        elif self.mode == "legacy" or epoch == self.warmup_epochs + 1:
            self.enabled, self.strength = True, self.initial

    def update(self, accuracy, epoch):
        if epoch <= self.warmup_epochs:
            return
        chance = 1.0 / self.num_domains
        if self.mode == "legacy":
            # Preserve branch order, batch accuracy, 0.1 threshold and epoch reset.
            value, upper_threshold = accuracy, 0.1
        else:
            self.accuracy_ema = 0.9 * self.accuracy_ema + 0.1 * accuracy
            value, upper_threshold = self.accuracy_ema, chance
        if value > upper_threshold:
            self.enabled = True
            self.strength = min(self.strength * 1.05, self.maximum)
        elif value < chance:
            self.enabled = False
            self.strength = max(self.strength * 0.1, self.minimum)

    def apply(self, model):
        model.set_grl(self.enabled, self.strength)

    def state_dict(self):
        return asdict(self)
