"""Command bridge from the simulated Inspire hand to the real RH56DFX interface (sim-to-real groundwork).

The simulation commands the 6 actuated joints of each hand in radians, in the order of `spec.hand_actuators`:

    thumb_proximal_yaw, thumb_proximal_pitch, index, middle, ring, pinky       (0 = open, max = closed)

The real hand (Unitree `dfx_inspire_service`, Inspire RH56DFX) takes six *normalised* targets in [0, 1] with
1 = open and 0 = closed, in the order

    pinky, ring, middle, index, thumb_bend, thumb_rotation

(the native Modbus registers are the same values x1000), plus a per-finger force limit (~9.8 N max).
`InspireBridge` converts between the two, rate-limits the commands like the simulation's safety layer and
never forwards a non-finite command. The mapping is linear in joint angle over the simulated joint range;
it has not been calibrated on hardware (no real hand was available), so `calibrate()` lets measured
open / closed readings replace the simulated end points per channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from homehand.model import spec

# simulated joint range of each actuator (rad), order of spec.hand_actuators
SIM_RANGE = np.array([1.308, 0.6, 1.47, 1.47, 1.47, 1.47])
# real channel k is driven by simulated actuator REAL_FROM_SIM[k]
REAL_ORDER = ("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation")
REAL_FROM_SIM = np.array([5, 4, 3, 2, 1, 0])
MAX_FINGER_FORCE_N = 9.8          # RH56DFX rated grip force per finger
MAX_RATE = 3.0 / SIM_RANGE.max()  # normalised units per second, same as the 3 rad/s finger limit in simulation


@dataclass
class InspireBridge:
    """Stateful converter for one hand (keeps the last command for rate limiting and NaN fallback)."""

    side: str = "right"
    dt: float = spec.CONTROL_DT
    force_limit_n: float = 6.0                   # per finger; the simulated grasps need ~4-6 N
    open_pt: np.ndarray = field(default_factory=lambda: np.zeros(6))      # sim rad that maps to real 1 (open)
    closed_pt: np.ndarray = field(default_factory=lambda: SIM_RANGE.copy())  # sim rad that maps to real 0
    last: np.ndarray | None = None

    def calibrate(self, open_rad, closed_rad) -> None:
        """Replace the end points with joint angles measured on the real hand (sim order, rad)."""
        self.open_pt, self.closed_pt = np.asarray(open_rad, float), np.asarray(closed_rad, float)

    # ------------------------------------------------------------------ sim -> real
    def to_real(self, hand6_rad) -> np.ndarray:
        """6 simulated joint targets (rad, sim order) -> 6 normalised real targets (real order, 1 = open)."""
        q = np.asarray(hand6_rad, float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            return self.last.copy() if self.last is not None else np.ones(6)   # hold (or stay open)
        closure = (q - self.open_pt) / (self.closed_pt - self.open_pt)
        cmd = np.clip(1.0 - closure, 0.0, 1.0)[REAL_FROM_SIM]
        if self.last is not None:
            step = MAX_RATE * self.dt
            cmd = np.clip(cmd, self.last - step, self.last + step)
        self.last = cmd
        return cmd.copy()

    def to_registers(self, hand6_rad) -> np.ndarray:
        """Same as `to_real`, as the RH56DFX angle registers (int 0..1000)."""
        return np.rint(1000 * self.to_real(hand6_rad)).astype(int)

    def force_registers(self) -> np.ndarray:
        """Per-finger force limit registers (0..1000 of the rated force), clipped to the rated maximum."""
        f = float(np.clip(self.force_limit_n, 0.0, MAX_FINGER_FORCE_N))
        return np.full(6, int(round(1000 * f / MAX_FINGER_FORCE_N)))

    # ------------------------------------------------------------------ real -> sim
    def to_sim(self, real6) -> np.ndarray:
        """Normalised real feedback (real order, 1 = open) -> simulated joint angles (rad, sim order)."""
        r = np.clip(np.asarray(real6, float), 0.0, 1.0)
        out = np.empty(6)
        out[REAL_FROM_SIM] = 1.0 - r
        return self.open_pt + out * (self.closed_pt - self.open_pt)


def split_action(action26) -> dict[str, np.ndarray]:
    """The 6 hand targets of each side from a full 26-dim simulation action."""
    a = np.asarray(action26, float)
    return {"left": a[14:20], "right": a[20:26]}
