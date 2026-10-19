# syntax=docker/dockerfile:1

# Dockerfile for genesis-cloud-sim
# Supports CPU, CUDA, and MUSA (Moore Threads) backends via build arguments.
#
# Build CPU image:
#   docker build -t genesis-cloud-sim:cpu .
#
# Build GPU image:
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04 \
#                --build-arg TORCH_BACKEND=cuda \
#                -t genesis-cloud-sim:gpu .
#
# Build in mainland China (Aliyun mirrors for PyTorch wheels and PyPI):
#   docker build --build-arg CHINA_MIRROR=aliyun -t genesis-cloud-sim:cpu .
#
# Build for Moore Threads MUSA (installs torch_musa):
#   docker build --build-arg TORCH_BACKEND=musa -t genesis-cloud-sim:musa .
#
# Run:
#   docker run -it --rm genesis-cloud-sim:cpu
#   docker run -it --rm --gpus all genesis-cloud-sim:gpu

ARG BASE_IMAGE=ubuntu:22.04
ARG TORCH_BACKEND=auto
ARG CHINA_MIRROR=auto
ARG PYTORCH_INDEX_URL=
ARG PYTHON_VERSION=3.11

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1

# Build arguments (must be re-declared after FROM)
ARG TORCH_BACKEND
ARG CHINA_MIRROR
ARG PYTORCH_INDEX_URL
ARG PYTHON_VERSION

# Install system dependencies required by Genesis, Taichi, OpenCV, and mesh processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libegl1 \
    libgl1-mesa-glx \
    libgomp1 \
    libopencv-dev \
    libosmesa6-dev \
    libtinfo5 \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make the selected Python the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip tooling to avoid build failures with Taichi/Genesis wheels,
# and install uv for lockfile-based (reproducible) dependency installation.
RUN python -m pip install --upgrade pip wheel setuptools uv

# Create a non-root user for development and runtime
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

WORKDIR /workspace

# Dependencies are installed from uv.lock (frozen, no re-resolution). This
# layer is cached as long as pyproject.toml / uv.lock are unchanged. uv keeps
# the environment in its own venv; the PATH below makes `python` resolve to
# it at runtime.
COPY pyproject.toml uv.lock /workspace/
RUN uv sync --frozen --extra dev --no-install-project

# Copy the project and finish the editable install from the lockfile.
COPY --chown=${USERNAME}:${USERNAME} . /workspace
RUN uv sync --frozen --extra dev

# Make the uv-managed environment the default interpreter.
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH=/workspace/.venv/bin:$PATH

# Overlay the backend-specific PyTorch wheels AFTER the uv sync — the same
# convention as local development (see AGENTS.md): the lock pins plain
# torch, so every `uv sync` must be followed by re-applying the overlay.
# tools/install_torch.py auto-detects MUSA/CUDA hardware and switches to
# Aliyun mirrors when the official PyTorch index is unreachable (mainland
# China). Set PYTORCH_INDEX_URL to pin an explicit wheel index. The uv venv
# ships without pip, so seed it first for the overlay step.
COPY tools/install_torch.py /tmp/install_torch.py
RUN python -m ensurepip --upgrade \
    && python /tmp/install_torch.py \
        --backend "${TORCH_BACKEND}" \
        --mirror "${CHINA_MIRROR}" \
        ${PYTORCH_INDEX_URL:+--index-url "${PYTORCH_INDEX_URL}"}

USER ${USERNAME}

# Default to CPU backend; override at runtime with -e GENESIS_BACKEND=cuda
ENV GENESIS_BACKEND=cpu
ENV TAICHI_ARCH=cpu

# Verify import on container start
CMD ["python", "-c", "import genesis as gs; gs.init(backend=gs.cpu); print('genesis-cloud-sim ready')"]
