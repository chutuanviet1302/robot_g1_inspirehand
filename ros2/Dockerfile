# HomeHand ROS 2 (Humble): simulation bridge + controller + Inspire hand bridge in one container.
#
#   docker compose build ros                                  # core (expert / bimanual), ~1.3 GB
#   docker compose build ros --build-arg WITH_TORCH=1         # + CPU PyTorch / SB3 (residual RL, ACT, flow)
#   docker compose run --rm ros                               # one tidy-table episode over ROS topics
#   docker compose run --rm ros ros2 launch homehand_ros tidy.launch.py task:=bimanual_lift seed:=4
#
# generated/ (the MuJoCo model, built on the host) and models/ are mounted, as in the main image.
FROM ros:humble-ros-base

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip libgl1 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Ubuntu 22.04's pip 22.0 installs a pyproject.toml project as "UNKNOWN-0.0.0" without its dependencies: use a
# current pip (setuptools stays the system's 59, which colcon needs for ament_python packages).
# numpy < 2: the Humble Python packages (rclpy, message types) are built against Ubuntu 22.04's numpy 1.x
RUN python3 -m pip install --upgrade "pip>=24"
COPY pyproject.toml README.md ./
RUN mkdir homehand && touch homehand/__init__.py \
    && pip3 install "numpy<2" "." && pip3 uninstall -y homehand && rm -rf homehand build
ARG WITH_TORCH=0
RUN if [ "$WITH_TORCH" = "1" ]; then \
      pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
      && pip3 install "numpy<2" stable-baselines3==2.5.0 gymnasium==1.0.0 pandas; \
    fi
COPY homehand ./homehand
# installed (not editable: that would need a newer setuptools than colcon tolerates), so the package no longer
# sits next to generated/, models/ and data/: point homehand at the mounted directories
ENV HOMEHAND_GENERATED=/app/generated HOMEHAND_MODELS=/app/models HOMEHAND_DATA=/app/data \
    HOMEHAND_THIRD_PARTY=/app/third_party
RUN pip3 install --no-deps . && mkdir -p generated models data \
    && python3 -c "import mujoco, mink, homehand.env.kitchen_env; print('homehand OK, mujoco', mujoco.__version__)"

COPY ros2/homehand_ros /ws/src/homehand_ros
RUN . /opt/ros/humble/setup.sh && cd /ws && colcon build --symlink-install
COPY ros2/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "homehand_ros", "tidy.launch.py"]
