"""Writes kaggle/flat_grasp_ppo.ipynb: a self-contained notebook (the env, the trainer and the model are embedded
with %%writefile, nothing to upload) that trains the flat-box grasp policy with Brax PPO on MJX.

    python -m homehand.rl.flat_grasp_model      # first: generated/flat_grasp_train.xml + flat_grasp.json
    python kaggle/make_flat_grasp_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = 150_000_000

EMBED = [("flat_grasp_env.py", ROOT / "homehand/rl/flat_grasp_env.py"),
         ("train_flat_grasp.py", ROOT / "homehand/rl/train_flat_grasp.py"),
         ("mjx_bench.py", ROOT / "homehand/rl/mjx_bench.py"),
         ("flat_grasp_train.xml", ROOT / "generated/flat_grasp_train.xml"),
         ("flat_grasp.json", ROOT / "generated/flat_grasp.json")]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def main() -> None:
    cells = [md("""
# HomeHand — flat-box grasp with PPO on MJX

Settings (right panel): **Accelerator: GPU T4 x2**, **Internet: on**, **Persistence: Files only**. No dataset needed:
the environment, the trainer and the MuJoCo model are written out by the cells below.

Run with **Save Version → Save & Run All (Commit)**. Everything in `/kaggle/working/out` (policy, progress) becomes
the version's output; `flat_grasp_out.zip` holds all of it.
"""),
             code("""
!pip install -q "jax[cuda12]==0.9.2" "flax==0.12.6" "brax==0.14.2" "mujoco==3.15.0" "mujoco-mjx==3.15.0"
!nvidia-smi --query-gpu=name,memory.total --format=csv
import os
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
import jax; print(jax.__version__, jax.devices())
""")]
    for name, path in EMBED:
        cells.append(code(f"%%writefile {name}\n" + path.read_text(encoding="utf-8")))
    cells += [
        md("Throughput (policy steps / s on one GPU), a learning check (20M steps), then the real run."),
        code("!python mjx_bench.py flat_grasp_train.xml --envs 1024 2>&1 | grep -v Warning"),
        code("!python train_flat_grasp.py --steps 20000000 --evals 6 --out check 2>&1 | grep -v Warning"),
        code(f"!python train_flat_grasp.py --steps {STEPS} --evals 30 --out out 2>&1 | grep -v Warning"),
        code("!cd out && zip -q -r ../flat_grasp_out.zip . && ls -la /kaggle/working/out"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "kaggle/flat_grasp_ppo.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
