"""Central place for filesystem locations. Everything is pathlib-based so it works on Windows and Linux."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# Third-party robot descriptions (downloaded by `homehand fetch-assets`, never vendored: dex-urdf is CC BY-NC-SA).
THIRD_PARTY_DIR = Path(os.environ.get("HOMEHAND_THIRD_PARTY", PROJECT_ROOT / "third_party"))
MENAGERIE_DIR = THIRD_PARTY_DIR / "mujoco_menagerie"
DEX_URDF_DIR = THIRD_PARTY_DIR / "dex-urdf"
G1_DIR = MENAGERIE_DIR / "unitree_g1"
INSPIRE_DIR = DEX_URDF_DIR / "robots" / "hands" / "inspire_hand"
YCB_DIR = THIRD_PARTY_DIR / "ycb"
# Optional ready-made G1 + Inspire MJCF (hands attached); used instead of Menagerie + dex-urdf when present.
REPO_MODEL_DIR = Path(os.environ.get("HOMEHAND_G1_INSPIRE_DIR", THIRD_PARTY_DIR / "unitree_g1_inspire"))

# Hand-written scene pieces shipped with the package.
ASSETS_DIR = PACKAGE_DIR / "assets"

# Output of `homehand build-model`.
GENERATED_DIR = Path(os.environ.get("HOMEHAND_GENERATED", PROJECT_ROOT / "generated"))
ROBOT_XML = GENERATED_DIR / "g1_inspire.xml"
SCENE_XML = GENERATED_DIR / "scene_kitchen.xml"
OBJECTS_JSON = GENERATED_DIR / "objects.json"

# Runtime data: datasets, eval database, episode replays, checkpoints.
DATA_DIR = Path(os.environ.get("HOMEHAND_DATA", PROJECT_ROOT / "data"))
DATASET_DIR = DATA_DIR / "datasets"
EPISODE_DIR = DATA_DIR / "episodes"
DB_PATH = DATA_DIR / "homehand.sqlite"
MODELS_DIR = Path(os.environ.get("HOMEHAND_MODELS", PROJECT_ROOT / "models"))

# Built frontend (vite build output).
WEB_DIST_DIR = PACKAGE_DIR / "web_static"
