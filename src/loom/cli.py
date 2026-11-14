#!/usr/bin/env python3
"""
Unified CLI Entrypoint for Uni-Loom (`loom`).

Usage:
    # Via uv (recommended):
    uv run loom --config configs/qm9s_ocelot_octree.json

    # Or directly inside container:
    loom --config /workspace/config.json
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="loom",
        description="Uni-Loom: Weaving Generative, Potential, and Property Foundation Models for Calibrated 3D Molecular Generation",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the runtime JSON configuration file",
    )
    args = parser.parse_args()
    from loom.pipeline import run_from_config

    run_from_config(args.config)


if __name__ == "__main__":
    main()
