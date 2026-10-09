# HomeHand: G1 + two Inspire hands tidying a kitchen counter (MuJoCo).
#
# The image holds the code and its Python dependencies only. Robot / object assets are mounted at run time
# (third_party/ is not redistributable: the G1+Inspire MJCF has no licence, dex-urdf is CC BY-NC-SA), as are
# generated/, data/ and models/. See docker-compose.yml and README ("Docker").
#
#   docker compose build                 # core + tests (~1 GB)
#   docker compose build --build-arg EXTRAS=dev,learn   # + CPU PyTorch / YOLO for collect / train / eval
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    MUJOCO_GL=egl

# OpenGL for MuJoCo: OSMesa for headless rendering (works in any container, CPU only; EGL needs the NVIDIA
# container toolkit), Mesa GLX + X11 client libraries for the native viewer (MUJOCO_GL=glfw, see compose `view`).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libegl-mesa0 libgl1-mesa-dri libosmesa6 libglib2.0-0 \
        libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 libxcursor1 libxi6 libxxf86vm1 libxkbcommon0 \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Headless rendering through OSMesa (CPU): works in any container, EGL would need the NVIDIA toolkit.
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

# Same uid as a typical desktop user, so files written to the mounted volumes stay owned by the host user.
ARG UID=1000
RUN useradd -m -u ${UID} homehand
WORKDIR /app

# 1) dependencies only (cached until pyproject.toml or the wheelhouse changes). docker/wheels/ holds the
#    Linux wheels pre-downloaded on the host (scripts/docker_wheels.sh): the build then needs no PyPI access,
#    which matters on networks where large downloads from inside Docker break off.
ARG EXTRAS=dev
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY docker/wheels /wheels
RUN mkdir homehand && touch homehand/__init__.py \
    && if echo "${EXTRAS}" | grep -q learn; then \
         pip install torch torchvision --index-url "${TORCH_INDEX}" \
         && pip install --find-links /wheels ".[${EXTRAS}]"; \
       else \
         pip install --no-index --find-links /wheels ".[${EXTRAS}]"; \
       fi \
    && pip uninstall -y homehand && rm -rf homehand build /wheels

# 2) the code
COPY homehand ./homehand
COPY tests ./tests
COPY scripts ./scripts
RUN pip install --no-deps --no-build-isolation -e . \
    && mkdir -p third_party generated data models \
    && chown -R homehand:homehand /app
USER homehand

EXPOSE 8000
ENTRYPOINT ["homehand"]
CMD ["serve", "--host", "0.0.0.0", "--no-open-browser"]
