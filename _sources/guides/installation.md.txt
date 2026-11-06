# Installation Guide

## System Requirements

### Minimum Requirements
- Python 3.10 or higher
- CUDA 11.8+ (for GPU support)
- 8GB RAM
- 10GB free disk space

### Recommended
- Python 3.11
- CUDA 12.1+
- RTX 4090 or A100 GPU
- 32GB RAM
- SSD storage

## Installation Methods

### Option 1: pip (Recommended)

```bash
pip install cloud-robotics-sim
```

### Option 2: From Source

```bash
git clone https://github.com/GWinfinity/cloud-robotics-sim.git
cd cloud-robotics-sim
pip install -e ".[dev]"
```

### Option 3: Docker

Build and run locally (CPU or GPU):

```bash
# CPU
docker compose up -d
docker exec -it genesis-cloud-sim-cpu bash

# GPU (requires NVIDIA Container Toolkit)
docker compose -f docker-compose.gpu.yml up -d
docker exec -it genesis-cloud-sim-gpu bash
```

See the full [Docker Usage Guide](./docker.md) for manual builds, GUI forwarding,
and troubleshooting.

## PyTorch Wheel Selection (Mirror & MUSA)

`tools/install_torch.py` installs the correct PyTorch wheels for the
deployment machine. It makes two decisions automatically:

1. **Compute backend** — Moore Threads MUSA hardware (via `mthreads-gmi`,
   `/dev/mtgpu*`, or `lspci`) selects `torch_musa`; NVIDIA GPUs select the
   CUDA wheels; everything else gets the CPU wheels.
2. **Package mirror** — when the official PyTorch index is unreachable
   (typical in mainland China), it switches to the Aliyun mirrors
   (`mirrors.aliyun.com/pytorch-wheels` and `mirrors.aliyun.com/pypi/simple`).

```bash
# Auto-detect backend and mirror
python tools/install_torch.py

# Preview what would be installed
python tools/install_torch.py --dry-run

# Force Moore Threads MUSA with Aliyun mirrors
python tools/install_torch.py --backend musa --mirror aliyun
```

Both decisions can also be pinned with environment variables:

| Variable | Values | Description |
|----------|--------|-------------|
| `CRS_TORCH_BACKEND` | `cpu` / `cuda` / `musa` | Force the compute backend |
| `CRS_PIP_MIRROR` | `official` / `aliyun` | Force the package mirror |
| `CRS_PYTORCH_INDEX_URL` | URL | Explicit wheel index overriding mirror selection |

The same logic runs inside Docker builds via the `TORCH_BACKEND` and
`CHINA_MIRROR` build arguments (see the [Docker Usage Guide](./docker.md)).

Run this script **before** `pip install -e .` so the project resolves against
the already-installed torch build.

## GPU Setup

### CUDA Installation

Ubuntu/Debian:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### Verify GPU Support

```python
import genesis as gs
gs.init(backend=gs.backends.CUDA)
print("GPU initialized successfully!")
```

## Development Installation

For contributing to the project:

```bash
git clone https://github.com/GWinfinity/cloud-robotics-sim.git
cd cloud-robotics-sim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all dev dependencies
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install
```

## Troubleshooting

### Genesis Import Error

If you encounter:
```
ImportError: libcuda.so.1: cannot open shared object file
```

Solution:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CUDA Version Mismatch

Ensure PyTorch CUDA version matches system CUDA:

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### Memory Issues

For limited GPU memory:

```python
from cloud_robotics_sim import ComposerConfig

config = ComposerConfig(
    headless=True,  # Disable viewer
    resolution=(320, 240),  # Lower resolution
)
```

## Verification

Test your installation:

```bash
python -c "import cloud_robotics_sim; print(cloud_robotics_sim.__version__)"
cloud-robotics-sim test
```
