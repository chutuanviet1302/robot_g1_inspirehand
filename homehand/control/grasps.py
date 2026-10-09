"""Grasp library: one grasp type per object class.

Every grasp uses the forward "handshake" approach (see `skills.py`), but the hand preshape, which fingers
close and by how much, the thumb opposition and the grasp height differ per object:

  cylindrical power wrap  all four fingers wrap the can, thumb opposes across it
  low power wrap          tall bottle: grip the body below the shoulder, deeper thumb opposition
  palmar box grasp        palm flat on a box face, fingers hook the far edge, thumb presses the near face
  tripod pinch            short box: thumb + index + middle pinch the upper half, ring/pinky only support

Finger order of a hand command: (thumb_yaw, thumb_pitch, index, middle, ring, pinky), radians.
"""

from __future__ import annotations

from dataclasses import dataclass


# A resting human hand: fingers slightly curled, more towards the little finger, thumb along the index.
RELAXED_HAND = (0.0, 0.1, 0.3, 0.35, 0.4, 0.45)


@dataclass(frozen=True)
class GraspType:
    name: str
    label: str
    thumb_yaw: float                  # thumb opposition, held from the approach on
    close: tuple                      # (thumb_pitch, index, middle, ring, pinky) when closed
    z_frac: float = 0.5               # grasp height as a fraction of object height
    min_z: float = 0.05               # palm height floor (lowest finger must clear the counter)
    reach: float = 0.016              # extra distance between the thumb and the near side of the object
    palm_gap: float = -0.004          # palm-site offset relative to the object's side (negative = press in)
    approach: str = "side"            # "side": handshake from behind; "top": palm down from above (lying objects)
    pre_thumb_yaw: float | None = None  # thumb opposition while approaching (top grasps keep the thumb out of
                                        # the way: opposed, it hangs 7 cm below the palm and hits the counter)
    # top grasps only (found with scripts/top_grasp_search.py): the palm-down hand is tilted fingers-down by
    # `tilt`; `align` = which way the thumb runs ("along" the lying object's long axis or "across" it); the
    # object centre sits at (dx along the fingers, dy along the palm normal, dz along the thumb) in the palm frame
    tilt_deg: float = 0.0
    align: str = "along"
    obj_in_palm: tuple = (0.06, 0.0, 0.02)
    held_in_palm: tuple | None = None  # where the object ends up once the hand has closed and lifted it
                                       # (measured; the drop point is computed from it, it moves 3-5 cm)

    def open_cmd(self) -> tuple:
        ty = self.thumb_yaw if self.pre_thumb_yaw is None else self.pre_thumb_yaw
        return (ty, 0.0, 0.0, 0.0, 0.0, 0.0)

    def closed_cmd(self) -> tuple:
        return (self.thumb_yaw, *self.close)

    def empty_threshold(self) -> float:
        """Mean flexion of the closing fingers above which the hand is considered empty."""
        flex = [c for c in self.close[1:] if c > 0.9]
        return 0.9 * (sum(flex) / len(flex))


GRASPS: dict[str, GraspType] = {
    g.name: g
    for g in [
        GraspType("cylindrical_wrap", "cylindrical power wrap", 1.0, (0.6, 1.47, 1.47, 1.47, 1.47)),
        GraspType("low_power_wrap", "low power wrap", 1.1, (0.6, 1.47, 1.47, 1.47, 1.47), z_frac=0.3),
        GraspType("palmar_box", "palmar box grasp", 1.2, (0.55, 1.35, 1.35, 1.35, 1.35), z_frac=0.45),
        # deeper thumb opposition, ring / pinky curled under as support, held low: 63/72 -> 69/72 in the grasp
        # benchmark (`homehand grasp-bench --objects meat_can`)
        GraspType("tripod_pinch", "tripod pinch", 1.3, (0.6, 1.47, 1.47, 1.3, 1.3), z_frac=0.5, min_z=0.045,
                  reach=0.016),
        # --- lying objects: palm down from above, the fingers wrap over the object across its axis (the axis
        # runs along the thumb), the thumb swings into opposition only while the hand closes
        # (the thumb is turned into opposition *before* the hand comes down, so it lands on the near side of the
        # object: closed later, it was blocked by the counter and the fingers dragged the object out of the hand)
        GraspType("lying_can_wrap", "top wrap, lying can", 1.0, (0.6, 1.47, 1.47, 1.47, 1.47), approach="top",
                  pre_thumb_yaw=0.5, tilt_deg=55, align="along", obj_in_palm=(0.06, 0.0, 0.015), held_in_palm=(0.028, 0.029, 0.029)),
        GraspType("lying_bottle_wrap", "top wrap, lying bottle", 1.0, (0.6, 1.47, 1.47, 1.47, 1.47), approach="top",
                  pre_thumb_yaw=0.5, tilt_deg=55, align="along", obj_in_palm=(0.06, -0.01, 0.015), held_in_palm=(0.048, 0.032, 0.020)),
        GraspType("flat_box_claw", "steep claw, flat box", 1.5, (0.6, 1.47, 1.47, 1.47, 1.47), approach="top",
                  pre_thumb_yaw=1.0, tilt_deg=65, align="along", obj_in_palm=(0.06, -0.015, 0.0), held_in_palm=(0.055, 0.016, -0.002)),
        GraspType("lying_block_pinch", "top pinch along, lying block", 1.0, (0.6, 1.47, 1.47, 1.47, 1.47),
                  approach="top", pre_thumb_yaw=1.0, tilt_deg=55, align="across", obj_in_palm=(0.05, -0.01, 0.015), held_in_palm=(0.062, 0.024, 0.020)),
    ]
}

# grasp for each object when it lies on its side (upright objects use ObjectSpec.grasp)
LYING_GRASP = {"soup_can": "lying_can_wrap", "mustard": "lying_bottle_wrap", "sugar_box": "flat_box_claw",
               "meat_can": "lying_block_pinch"}
