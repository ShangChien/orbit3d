#!/usr/bin/env python3
"""
Unified CLI Entrypoint for ORBIT-3D (`orbit3d`).

Usage:
    # Via uv (recommended):
    uv run orbit3d --config configs/qm9s_ocelot_octree.json

    # Or directly inside container:
    orbit3d --config /workspace/config.json
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="orbit3d",
        description="ORBIT-3D: Interval-targeted molecular generation and constrained 3D refinement",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the runtime JSON configuration file (e.g., configs/qm9s_ocelot_octree.json)",
    )
    args = parser.parse_args()
    from orbit3d.pipeline import run_from_config

    run_from_config(args.config)


if __name__ == "__main__":
    main()
