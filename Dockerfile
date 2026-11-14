# Locked runtime; model weights and datasets are mounted separately.
FROM python:3.11-slim-bookworm AS build
ARG PYTHON_INDEX_URL=https://pypi.org/simple
ARG DEBIAN_MIRROR=https://deb.debian.org
RUN python -m pip install --no-cache-dir --index-url "$PYTHON_INDEX_URL" uv==0.12.3
RUN sed -i "s|http://deb.debian.org|$DEBIAN_MIRROR|g" /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/loom
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
ARG LOOM_EXTRA=deepmd
ENV DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=0 UV_NO_CACHE=1
RUN export UV_DEFAULT_INDEX="$PYTHON_INDEX_URL" && \
    uv venv .venv && \
    uv export --frozen --no-dev --extra octree --extra "$LOOM_EXTRA" --no-emit-project -o /tmp/requirements.lock.txt && \
    uv pip install --python .venv/bin/python --require-hashes --no-binary deepmd-kit \
        --index-url "$PYTHON_INDEX_URL" -r /tmp/requirements.lock.txt && \
    uv build --wheel && \
    uv pip install --python .venv/bin/python --no-deps dist/loom-*.whl

FROM python:3.11-slim-bookworm
ARG DEBIAN_MIRROR=https://deb.debian.org
RUN sed -i "s|http://deb.debian.org|$DEBIAN_MIRROR|g" /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/loom
COPY --from=build /opt/loom/.venv .venv
ENV PATH="/opt/loom/.venv/bin:$PATH"
ENTRYPOINT ["loom", "--config"]
