ARG DOCKER_ROS_DISTRO=jazzy
ARG BASE_IMAGE=gazebo-cuda
FROM althack/ros2:${DOCKER_ROS_DISTRO}-dev AS hardware
FROM althack/ros2:${DOCKER_ROS_DISTRO}-gazebo AS gazebo
FROM althack/ros2:${DOCKER_ROS_DISTRO}-cuda-gazebo AS gazebo-cuda

FROM ${BASE_IMAGE:-gazebo-cuda} AS final

# Build arguments
ARG VIRTUALGL_VER="3.1"
ARG TARGETARCH
ARG DOCKER_ROS_DISTRO=jazzy
ARG BASE_IMAGE=gazebo-cuda
ARG USERNAME=ros
ARG WORKSPACE_NAME=ros2_ws
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG BIN_TO_WRAP="ros2 rviz2 rqt rqt_graph rqt_image_view rqt_plot rqt_console rqt_tf_tree python3"
ARG DOCKER_RMW_IMPLEMENTATION=

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    WORKSPACE=/home/${USERNAME}/${WORKSPACE_NAME}

# Use bash as the default shell to avoid sh/bash double-wrapping quoting issues
SHELL ["/bin/bash", "-c"]

# Install VirtualGL and graphics libraries, and configure GPU-accelerated CLI wrappers (gazebo-cuda only)
RUN /bin/bash -c " \
    if [ '${BASE_IMAGE}' = 'gazebo-cuda' ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            wget \
            libglvnd0 \
            libgl1 \
            libglx0 \
            libegl1 \
            libxext6 \
            libx11-6 && \
        wget -O /tmp/virtualgl.deb https://github.com/VirtualGL/virtualgl/releases/download/${VIRTUALGL_VER}/virtualgl_${VIRTUALGL_VER}_${TARGETARCH}.deb && \
        apt-get install -y /tmp/virtualgl.deb && \
        rm -f /tmp/virtualgl.deb && \
        rm -rf /var/lib/apt/lists/* ; \
    fi"

# Copy and configure entrypoint before leaving root user
COPY docker/entrypoint.sh /entrypoint.sh

# Remap existing user to match host UID/GID and grant sudo privileges
RUN groupmod --gid ${HOST_GID} ${USERNAME} \
    && usermod --uid ${HOST_UID} --gid ${HOST_GID} ${USERNAME} \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && chown -R ${HOST_UID}:${HOST_GID} /home/${USERNAME}

# Switch to ros user for building and create the workspace
USER ${USERNAME}
RUN mkdir -p ${WORKSPACE}/src
WORKDIR ${WORKSPACE}

# Configure VGL wrapper functions in user's bashrc (gazebo-cuda only)
RUN if [ "$BASE_IMAGE" = 'gazebo-cuda' ]; then \
        echo "BIN_TO_WRAP=(${BIN_TO_WRAP})" >> ~/.bashrc && \
        echo 'for cmd in "${BIN_TO_WRAP[@]}"; do' >> ~/.bashrc && \
        echo '  eval "function $cmd { vglrun -d /dev/dri/card\${GPU_ID:-0} \$(which $cmd) \"\$@\"; }"' >> ~/.bashrc && \
        echo 'done' >> ~/.bashrc ; \
    fi

# [Optional] Install additional apt dependencies here before rosdep
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        ros-$ROS_DISTRO-realsense2-camera&& \
    sudo rm -rf /var/lib/apt/lists/*

# Install SAM3 and runtime dependencies.
# Pin numpy<2 and opencv-python<4.11: cv_bridge apt binary is compiled against NumPy 1.x
RUN git clone https://github.com/facebookresearch/sam3 ${HOME}/sam3 && \
    pip install einops pycocotools && \
    pip install -e ${HOME}/sam3

# Pre-download SAM3 model weights so the node starts without a network fetch.
# SAM3 requires gated HuggingFace access — pass --build-arg HF_TOKEN=<your_token>.
ARG HF_TOKEN=
RUN HF_TOKEN=${HF_TOKEN} python3 -c "\
import os; \
from huggingface_hub import snapshot_download; \
snapshot_download('facebook/sam3', token=os.environ['HF_TOKEN']); \
print('SAM3 weights pre-loaded successfully.')"

# Install SAM2 and download weights.
# sam2 ships its config YAML files as package data, so a pip-editable install
# is not required — but cloning makes the configs accessible at a known path.
RUN git clone https://github.com/facebookresearch/sam2 ${HOME}/sam2 && \
    pip install -e "${HOME}/sam2[demo]"

# Pre-download SAM2.1 large checkpoint (~900 MB).
RUN mkdir -p ${HOME}/sam2_checkpoints && \
    wget -q -O ${HOME}/sam2_checkpoints/sam2.1_hiera_small.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

RUN pip install litellm "numpy<2" "opencv-python<4.11"

# Install RMW implementation
RUN sudo apt-get update && \
    if [ -n "${DOCKER_RMW_IMPLEMENTATION}" ]; then \
        sudo apt-get install -y --no-install-recommends \
            ros-${DOCKER_ROS_DISTRO}-$(echo ${DOCKER_RMW_IMPLEMENTATION} | tr '_' '-') ; \
    fi && \
    sudo rm -rf /var/lib/apt/lists/*

# Copy source directory
COPY --chown=${USERNAME}:${USERNAME} . ${WORKSPACE}/src/

# Update rosdep and install dependencies
RUN sudo apt-get update && \
    rosdep update --rosdistro=${DOCKER_ROS_DISTRO} && \
    rosdep install \
        --rosdistro=${DOCKER_ROS_DISTRO} \
        --from-paths src \
        -iry \
        --os=ubuntu:$(lsb_release --codename | cut -f2) && \
    sudo rm -rf /var/lib/apt/lists/*

# Build the workspace
RUN /bin/bash -c "source /opt/ros/${DOCKER_ROS_DISTRO}/setup.bash && colcon build"

# Enable lite6 gripper services in the xarm driver
RUN sed -i \
    -e 's/open_lite6_gripper: false/open_lite6_gripper: true/' \
    -e 's/close_lite6_gripper: false/close_lite6_gripper: true/' \
    -e 's/stop_lite6_gripper: false/stop_lite6_gripper: true/' \
    ${WORKSPACE}/src/xarm_ros2/xarm_api/config/xarm_params.yaml

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]