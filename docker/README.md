See [docs/09_docker.md](../docs/09_docker.md) for full Docker documentation.

## Image Naming

The Docker image is named using the `PROJECT_NAME` variable defined in `.env`:

```
<PROJECT_NAME>:<BASE_IMAGE>
```

For example, with the defaults (`PROJECT_NAME=my_project`, `BASE_IMAGE=gazebo`) the image will be tagged as:

```
my_project:gazebo
```

To use a different project name, edit `PROJECT_NAME` in `.env` before building.

## Workspace Name

The ROS workspace directory inside the container is controlled by `WORKSPACE_NAME` in `.env`:

```
WORKSPACE_NAME=ros2_ws
```

This resolves to `/home/ros/<WORKSPACE_NAME>` inside the container. To use a different workspace name, edit `WORKSPACE_NAME` in `.env` before building.
