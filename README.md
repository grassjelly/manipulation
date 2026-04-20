# manipulation

## Prerequisites

```bash
pip3 install vcs2l --break-system-packages
```

### gazebo-cuda build

The `gazebo-cuda` base image requires Docker with NVIDIA container runtime support.

**1. Install Docker and NVIDIA container runtime**

```bash
sudo apt update && sudo apt install -y curl

curl -sSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
sudo systemctl restart docker
```

If an NVIDIA GPU is present, install the NVIDIA container toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl daemon-reload && sudo systemctl restart docker
```

**2. Install NVIDIA drivers**

```bash
sudo apt update && sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
```

Reboot after driver installation before building the image.

## Usage

**1. Add your ROS 2 packages**

Choose one of the following approaches:

- **Using vcstool (recommended):** Edit `repos.yml` to declare your repositories, then import them into `src/`:

  ```bash
  vcs import . < repos.yml
  ```

- **Manually:** Place your ROS 2 package directories directly inside `src/`. The Dockerfile clones from `repos.yml` at build time, so any packages you want baked into the image must be listed there. For local development without rebuilding the image, manually placed packages in `src/` are picked up via the volume mount in the `dev` container.

**2. Build the Docker image**

```bash
cd docker
./build
```

**3. Start the development container**

```bash
./dev
```

## Customization

- Edit `repos.yml` to add or change the repositories cloned into `src/` during the image build.
- Edit `docker/docker-compose.yaml` to add services for your project.
- Edit `docker/Dockerfile` to change the ROS2 distro or base image variant (`hardware`, `gazebo`, `gazebo-cuda`).
