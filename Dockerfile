# Locked DPA runtime. Mount weights/data; GPU image execution must be validated separately.
FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/orbit3d
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
ARG ORBIT3D_EXTRA=deepmd
RUN uv sync --frozen --no-dev --extra octree --extra ${ORBIT3D_EXTRA}
ENV PATH="/opt/orbit3d/.venv/bin:$PATH"
ENTRYPOINT ["orbit3d", "--config"]
