from dataclasses import dataclass, field
from typing import List

@dataclass
class SingleLossConfig:
    name: str
    weight: float = 1.
    init_params: dict = field(default_factory=dict)
    visualize_every_k: int = -1


@dataclass
class LossesConfig:
    diffusion_losses: List[SingleLossConfig]
    lcm_losses: List[SingleLossConfig]