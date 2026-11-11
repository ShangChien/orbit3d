# syntax=docker/dockerfile:1.6
# ==============================================================================
# Ultra-Slim Multi-Stage Dockerfile for ORBIT-3D (Powered by `uv`)
# (Uni-3DAR + Gated-GRPO + External Uni-QSAR / Internal Uni-Mol / Internal DPA-4 / Internal HyFF
#  + Iso-Property DPA-4 & Charge/Spin-Conditioned HyFF Conformer Relaxation)
#
# Size Breakdown (~2.52 GB total vs ~14.5 GB legacy Uni-QSAR unicore image):
# - Base OS (`python:3.11-slim-bookworm` + minimal runtime `.so` libs): ~150 MB
# - Stripped `uv` Python venv (`torch 2.3.1+cu121`, `safetensors`, `rdkit`, `scikit-learn`,
#   `scipy`, `deepmd-kit` pt backend with debug symbols & test suites stripped)
#   + Vendored minimal `hyff` runtime (`src/orbit3d/compat/hyff`, 1.4 MB): ~1.96 GB
# - Built-in Pretrained Weights (`/opt/orbit3d/pretrained`): 411.82 MB
#   * Full DPA-4 Series (`Nano` 1.95MB, `Mini` 2.62MB, `Neo` 4.42MB,
#     `Air` 19.86MB, `Plus` 34.01MB) = 62.86 MB
#   * HyFF Small/Medium Sweet-Spot Series (`presto` 8.89MB, `vivace` 25.50MB,
#     `moderato` 132.89MB) = 167.28 MB
#   * FP16-compressed Uni-Mol 3D backbones (`all_h` 90.84MB + `no_h` 90.84MB) = 181.68 MB
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Dependency Builder & Binary Stripper (Accelerated by `uv`)
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=0 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV="/opt/venv" \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        binutils \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md /build/
COPY src/ /build/src/

RUN uv venv /opt/venv && \
    uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 && \
    uv pip install \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scipy==1.13.1 \
        scikit-learn==1.4.2 \
        joblib==1.4.2 \
        pyyaml==6.0.1 \
        safetensors==0.4.5 \
        rdkit-pypi==2022.9.5 \
        ase==3.23.0 && \
    uv pip install --no-deps . && \
    (uv pip install --no-deps deepmd-kit==3.0.0b4 || true)

RUN find /opt/venv -type d -name "tests" -o -name "test" -prune -exec rm -rf {} + && \
    find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    find /opt/venv -name "*.a" -delete && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv/lib -name "*.so*" -exec strip --strip-unneeded {} + 2>/dev/null || true

# ------------------------------------------------------------------------------
# Stage 2: Production Runtime Image
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="Shangqian Chen, Guojiang Zhao, Zheng Cheng, Jiankun Wang, Hongshuai Wang, Zhifeng Gao (DP Technology)" \
      description="ORBIT-3D: 3D Molecular Generation & DPA-4 / HyFF Conformer Relaxation (uv managed)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/orbit3d/src/orbit3d/compat:/opt/orbit3d/src:/opt/deepmd_v32" \
    ORBIT3D_PRETRAINED_DIR="/opt/orbit3d/pretrained" \
    HYFF_HOME="/opt/orbit3d/pretrained/hyff"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libxrender1 \
        libxext6 \
        libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /opt/orbit3d
COPY pyproject.toml README.md /opt/orbit3d/
COPY src/ /opt/orbit3d/src/
COPY scripts/ /opt/orbit3d/scripts/
COPY configs/ /opt/orbit3d/configs/
COPY pretrained/ /opt/orbit3d/pretrained/

RUN mkdir -p /workspace/data \
             /workspace/external_uniqsar \
             /workspace/checkpoints/save \
             /workspace/checkpoints/load \
             /workspace/outputs

VOLUME ["/workspace/data", "/workspace/external_uniqsar", "/workspace/checkpoints/save", "/workspace/checkpoints/load", "/workspace/outputs"]

WORKDIR /workspace

ENTRYPOINT ["orbit3d", "--config"]
CMD ["/opt/orbit3d/configs/external_uniqsar.json"]
