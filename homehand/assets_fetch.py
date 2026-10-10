"""Download third-party assets: robot descriptions at pinned commits (sparse git checkout) and YCB objects.

Only the folders we need are fetched, so the download stays ~90 MB.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from homehand import paths

SOURCES = {
    "mujoco_menagerie": {
        "url": "https://github.com/google-deepmind/mujoco_menagerie.git",
        "commit": "f054586a8e90465d49ee5be15335c4a0c7f57caf",
        "sparse": ["unitree_g1"],
    },
    "dex-urdf": {
        "url": "https://github.com/dexsuite/dex-urdf.git",
        "commit": "f5e7132f22108164577fea4c25ef99b5cc0e1900",
        "sparse": ["robots/hands/inspire_hand"],
    },
}


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def fetch_one(name: str, force: bool = False) -> Path:
    src = SOURCES[name]
    dest = paths.THIRD_PARTY_DIR / name
    marker = dest / src["sparse"][0]
    if marker.exists() and not force:
        print(f"[fetch] {name}: already present at {dest}")
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    print(f"[fetch] {name}: {src['url']} @ {src['commit'][:10]}")
    _git(["init", "-q"], dest)
    _git(["remote", "add", "origin", src["url"]], dest)
    _git(["sparse-checkout", "set", "--no-cone", *[f"/{p}/" for p in src["sparse"]]], dest)
    _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", src["commit"]], dest)
    _git(["checkout", "-q", "FETCH_HEAD"], dest)
    return dest


YCB_URL = "https://ycb-benchmarks.s3.amazonaws.com/data/google/{}_google_16k.tgz"


def fetch_ycb(force: bool = False) -> None:
    """YCB object meshes (CC BY 4.0), google_16k scans with textures, ~5-9 MB each."""
    import tarfile
    import tempfile
    import urllib.request

    from homehand.model.objects import OBJECTS

    paths.YCB_DIR.mkdir(parents=True, exist_ok=True)
    for o in OBJECTS.values():
        dest = paths.YCB_DIR / o.ycb_id / "google_16k" / "textured.obj"
        if dest.exists() and not force:
            print(f"[fetch] ycb {o.ycb_id}: already present")
            continue
        url = YCB_URL.format(o.ycb_id)
        print(f"[fetch] ycb {o.ycb_id}: {url}")
        with tempfile.TemporaryDirectory() as tmp:
            tgz = Path(tmp) / "obj.tgz"
            for attempt in range(5):
                try:
                    # (urlretrieve has no timeout: a stalled S3 connection hung the fetch forever)
                    with urllib.request.urlopen(url, timeout=60) as r, open(tgz, "wb") as f:
                        shutil.copyfileobj(r, f, 1 << 20)
                    break
                except OSError as e:
                    if attempt == 4:
                        raise
                    print(f"[fetch]   retry ({e})")
            with tarfile.open(tgz) as tf:
                members = [m for m in tf.getmembers() if m.name.startswith(f"{o.ycb_id}/google_16k/")
                           and Path(m.name).name in ("textured.obj", "texture_map.png", "textured.mtl")]
                tf.extractall(paths.YCB_DIR, members=members)


def fetch_all(force: bool = False) -> None:
    if shutil.which("git") is None:
        raise SystemExit("git is required to fetch assets (Windows: install Git for Windows).")
    for name in SOURCES:
        fetch_one(name, force=force)
    fetch_ycb(force=force)
