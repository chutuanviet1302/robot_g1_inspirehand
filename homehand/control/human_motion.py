"""Human motor-control models used to make the expert's motions look like a person's.

* Reaching is one continuous movement with a single bell-shaped speed profile (Flash & Hogan 1985). When the
  movement ends on an object to grasp, the profile is skewed: the peak comes at ~40 % of the movement and the
  deceleration towards the object is longer (Marteniuk et al. 1987).
* Movement time grows with the index of difficulty, Fitts' law: T = a + b log2(2D / W) (Fitts 1954).
* The hand opens *during* the reach, reaches its maximum aperture at ~60-70 % of the movement and is then
  carried, shaped, to the object (Jeannerod 1984); it does not open in one go before moving.
"""

from __future__ import annotations

import numpy as np

# speed profile v(tau) ~ tau^2 (1 - tau)^3: zero velocity and acceleration at both ends, peak at tau = 0.4
ASYM_PEAK_TIME = 0.4
ASYM_PEAK = 60 * 0.4 ** 2 * 0.6 ** 3      # peak / mean speed (2.07; minimum jerk: 1.875)

# Fitts' law constants for reach-to-grasp (order of magnitude of published arm-reaching fits)
FITTS_A = 0.25            # s
FITTS_B = 0.12            # s / bit
APERTURE_PEAK = 0.9       # fraction of the reach (start -> pre-grasp pose) at which the hand is fully pre-shaped;
                          # with the slow final approach this is ~60-70 % of the movement to contact, as in people


def asym_min_jerk(tau: float) -> float:
    """Position along a reach-to-grasp movement, tau in [0, 1] -> [0, 1] (integral of 60 t^2 (1 - t)^3)."""
    t = min(max(tau, 0.0), 1.0)
    return t ** 3 * (20 - 45 * t + 36 * t ** 2 - 10 * t ** 3)


def fitts_time(distance: float, width: float) -> float:
    """Movement time (s) for a reach of `distance` to a target of `width` (both m)."""
    return FITTS_A + FITTS_B * float(np.log2(max(2 * distance / max(width, 1e-3), 1.0)))


def preshape(tau: float, start, shaped) -> np.ndarray:
    """Hand command during a reach: from `start` to the pre-grasp shape, completed at APERTURE_PEAK and then
    held while the hand is carried onto the object. The thumb (index 0, opposition) leads the fingers."""
    start, shaped = np.asarray(start, float), np.asarray(shaped, float)
    u = np.full(len(start), min(1.0, max(tau, 0.0) / APERTURE_PEAK))
    u[0] = min(1.0, max(tau, 0.0) / (0.8 * APERTURE_PEAK))
    w = u * u * (3 - 2 * u)            # smooth start / end of the hand motion
    return start + (shaped - start) * w
