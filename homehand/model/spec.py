"""Names and constants shared by the model builder, environment, controllers and policy."""

SIM_DT = 0.002          # physics step (s)
CONTROL_DT = 0.04       # policy / expert step (s) -> 25 Hz
N_SUBSTEPS = round(CONTROL_DT / SIM_DT)

SIDES = ("left", "right")
PREFIX = {"left": "L_", "right": "R_"}

_ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
_HAND = [
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
]
HAND_URDF_JOINTS = _HAND  # actuated joint names as they appear in the Inspire URDF


def arm_joints(side: str) -> list[str]:
    return [f"{side}_{j}_joint" for j in _ARM]


def hand_actuators(side: str) -> list[str]:
    return [PREFIX[side] + j for j in _HAND]


def palm_site(side: str) -> str:
    return PREFIX[side] + "palm"


def fingertip_sites(side: str) -> list[str]:
    return [PREFIX[side] + f for f in ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")]


def hand_root(side: str) -> str:
    return PREFIX[side] + "hand_root"


ARM_JOINTS = arm_joints("left") + arm_joints("right")
# Action = absolute position targets, ordered exactly like the model's actuators:
# [left arm (7), right arm (7), left hand (6), right hand (6)]
ACTUATORS = ARM_JOINTS + hand_actuators("left") + hand_actuators("right")
ACTION_DIM = len(ACTUATORS)  # 26
N_ARM = 7
N_HAND = 6
