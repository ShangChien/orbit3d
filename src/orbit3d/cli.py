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
