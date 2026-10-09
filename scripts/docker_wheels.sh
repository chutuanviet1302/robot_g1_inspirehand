#!/usr/bin/env bash
# Pre-download the Linux / Python 3.10 wheels the Docker image installs (docker/wheels/), with retries.
# Run once on the host before `docker compose build`; re-run after changing dependencies in pyproject.toml.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p docker/wheels
for i in 1 2 3 4 5; do
  PIP_DEFAULT_TIMEOUT=120 pip download --retries 10 --find-links docker/wheels -d docker/wheels \
    --python-version 3.10 --only-binary=:all: \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
    --platform linux_x86_64 --platform any \
    "mujoco>=3.2" "mink>=0.0.10" daqp "numpy>=1.24" scipy pillow "fastapi>=0.110" "uvicorn[standard]>=0.29" \
    "typer>=0.12" "pydantic>=2" pytest httpx "setuptools>=68" wheel && exit 0
  echo "download interrupted, retrying ($i/5)"
done
exit 1
