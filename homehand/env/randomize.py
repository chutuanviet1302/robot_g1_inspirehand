"""Domain randomisation used to estimate the sim gap.

Without a real robot we measure how fast performance degrades when the simulator's assumptions are perturbed:
contact friction, object mass, perception noise on object poses, and actuation latency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Randomization:
    friction: tuple[float, float] = (1.0, 1.0)      # multiplier on object + fingertip sliding friction
    mass: tuple[float, float] = (1.0, 1.0)          # multiplier on object mass
    obs_noise: float = 0.0                          # std (m) of gaussian noise on observed object/handle positions
    latency: tuple[int, int] = (0, 0)               # action delay in control steps (1 step = 40 ms)
    finger_gain: tuple[float, float] = (1.0, 1.0)   # multiplier on finger actuator stiffness

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Randomization":
        if not d:
            return cls()
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)

    def sample(self, rng: np.random.Generator) -> dict:
        return {
            "friction": float(rng.uniform(*self.friction)),
            "mass": float(rng.uniform(*self.mass)),
            "obs_noise": float(self.obs_noise),
            "latency": int(rng.integers(self.latency[0], self.latency[1] + 1)),
            "finger_gain": float(rng.uniform(*self.finger_gain)),
        }


PRESETS: dict[str, Randomization] = {
    "nominal": Randomization(),
    "low": Randomization(friction=(0.85, 1.15), mass=(0.85, 1.2), obs_noise=0.003, latency=(0, 1),
                         finger_gain=(0.9, 1.1)),
    "medium": Randomization(friction=(0.7, 1.3), mass=(0.7, 1.5), obs_noise=0.006, latency=(0, 2),
                            finger_gain=(0.8, 1.2)),
    "high": Randomization(friction=(0.5, 1.5), mass=(0.5, 2.0), obs_noise=0.012, latency=(1, 3),
                          finger_gain=(0.6, 1.4)),
}


def resolve(r: str | dict | Randomization | None) -> Randomization:
    if r is None:
        return PRESETS["nominal"]
    if isinstance(r, Randomization):
        return r
    if isinstance(r, str):
        return PRESETS[r]
    return Randomization.from_dict(r)
