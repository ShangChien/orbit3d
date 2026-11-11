#!/usr/bin/env python3
"""
Unified CLI Entrypoint for ORBIT-3D (`orbit3d`).

Usage:
    # Via uv (recommended):
    uv run orbit3d --config configs/internal_dpa4.json

    # Or directly inside container:
    orbit3d --config /workspace/config.json
"""

import argparse
import os
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_SRC_DIR = _PKG_DIR.parent
for _p in [str(_SRC_DIR), "/opt/orbit3d/src", "/opt/deepmd_v32", "/sjtu-combustion/wjw/chensq/deepmd_v32"]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from orbit3d.pipeline import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="orbit3d",
        description="ORBIT-3D: Octree-Based Reinforced Bidirectional Interval-Targeted 3D Molecular Generation",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the runtime JSON configuration file (e.g., configs/internal_dpa4.json)",
    )
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
