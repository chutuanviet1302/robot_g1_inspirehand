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
# numpy < 2: the Humble Python packages (rclpy, message types) are built against Ubuntu 22.04's numpy 1.x
COPY pyproject.toml README.md ./
RUN mkdir homehand && touch homehand/__init__.py \
    && pip3 install "numpy<2" "." && pip3 uninstall -y homehand && rm -rf homehand build
ARG WITH_TORCH=0
RUN if [ "$WITH_TORCH" = "1" ]; then \
      pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
      && pip3 install "numpy<2" stable-baselines3==2.5.0 gymnasium==1.0.0 pandas; \
    fi
COPY homehand ./homehand
RUN pip3 install --no-deps --no-build-isolation -e . && mkdir -p generated models data

COPY ros2/homehand_ros /ws/src/homehand_ros
RUN . /opt/ros/humble/setup.sh && cd /ws && colcon build --symlink-install
COPY ros2/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "homehand_ros", "tidy.launch.py"]
