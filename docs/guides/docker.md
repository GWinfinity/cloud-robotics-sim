# Docker Usage Guide

This guide explains how to build and run `genesis-cloud-sim` inside Docker. Two
flavors are provided:

* **CPU** – works on any machine, including laptops and CI runners without a GPU.
* **GPU** – uses the NVIDIA CUDA 12.1 runtime and PyTorch CUDA wheels.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-arg image supporting both CPU and GPU builds |
| `docker-compose.yml` | CPU service definition |
| `docker-compose.gpu.yml` | GPU service definition |
| `.dockerignore` | Keeps the image small by excluding caches, outputs, and VCS data |

## Quick Start (CPU)

```bash
docker compose up -d
```

This builds the `genesis-cloud-sim:cpu` image and starts a container named
`genesis-cloud-sim-cpu`.

Open a shell inside the running container:

```bash
docker exec -it genesis-cloud-sim-cpu bash
```

Run the do-as-i-do scaffold:

```bash
python plugins/do_as_i_do/examples/demo_synthetic.py --hand-type sharpa --num-frames 60
```

## Quick Start (GPU)

Make sure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
is installed on the host.

```bash
docker compose -f docker-compose.gpu.yml up -d
docker exec -it genesis-cloud-sim-gpu bash
```

Verify GPU access:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Manual Build

### CPU image

```bash
docker build -t genesis-cloud-sim:cpu .
```

### GPU image

```bash
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04 \
  --build-arg TORCH_BACKEND=cuda \
  -t genesis-cloud-sim:gpu .
```

### Mainland China builds (Aliyun mirrors)

```bash
docker build --build-arg CHINA_MIRROR=aliyun -t genesis-cloud-sim:cpu .
```

### Moore Threads MUSA image

```bash
docker build --build-arg TORCH_BACKEND=musa -t genesis-cloud-sim:musa .
```

### Build arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `BASE_IMAGE` | `ubuntu:22.04` | Base image (use `nvidia/cuda:...` for GPU builds) |
| `TORCH_BACKEND` | `auto` | `cpu`, `cuda`, `musa`, or `auto` (detect hardware) |
| `CHINA_MIRROR` | `auto` | `official`, `aliyun`, or `auto` (probe connectivity) |
| `PYTORCH_INDEX_URL` | *(empty)* | Explicit wheel index overriding mirror selection |
| `PYTHON_VERSION` | `3.11` | Python version to install |

The PyTorch installation is handled by `tools/install_torch.py` (see the
[Installation Guide](./installation.md#pytorch-wheel-selection-mirror--musa)).

### Run interactively

```bash
# CPU
docker run -it --rm -v $(pwd):/workspace genesis-cloud-sim:cpu

# GPU
docker run -it --rm --gpus all -v $(pwd):/workspace genesis-cloud-sim:gpu
```

## Runtime Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GENESIS_BACKEND` | `cpu` | Genesis backend selector (`cpu` or `cuda`) |
| `TAICHI_ARCH` | `cpu` | Taichi arch selector (`cpu` or `cuda`) |

Override them when starting a container:

```bash
docker run -it --rm -e GENESIS_BACKEND=cuda -e TAICHI_ARCH=cuda genesis-cloud-sim:gpu
```

## GUI / Viewer Support

For headless servers, Genesis can run without a display. If you want to forward
X11 for the viewer, add the host display when running the container:

```bash
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  genesis-cloud-sim:cpu
```

On Windows, use an X server such as VcXsrv or WSLg.

## Development Workflow

The compose files mount the project root into `/workspace`, so code changes on
the host are reflected immediately inside the container. Because the package is
installed in editable mode (`pip install -e .`), you do not need to rebuild the
image when the Python source changes.

To add new runtime dependencies, update `pyproject.toml` and restart the
container (or run `pip install -e ".[dev]"` inside it).

## Troubleshooting

### `libcuda.so.1: cannot open shared object file`

When running the GPU image, ensure the NVIDIA Container Toolkit is installed
and the container was started with `--gpus all` or `runtime: nvidia`.

### Genesis/Taichi kernel compilation is slow on first run

Taichi compiles kernels on first use. Mount a persistent volume for
`/home/dev/.cache` (already configured in the compose files) so the compiled
kernels survive container restarts.

### Permission issues on Linux

The container runs as a non-root user with UID/GID `1000` by default. If your
host user has a different UID, change the build arguments:

```bash
docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t genesis-cloud-sim:cpu .
```
