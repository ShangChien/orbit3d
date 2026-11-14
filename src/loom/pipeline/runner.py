"""Dispatch the explicit proposal backend to the shared research workflows."""

import json
from typing import Any


def run_from_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as stream:
        cfg = json.load(stream)
    backend = cfg.get("generator_config", {}).get("backend")
    if backend == "uni3dar_octree":
        from .octree_runner import run_octree_from_config
        return run_octree_from_config(cfg, config_path)
    if backend == "scaffold_graph_growth":
        from .scaffold_runner import run_scaffold_from_config
        return run_scaffold_from_config(cfg, config_path)
    raise ValueError("Declare generator_config.backend explicitly as uni3dar_octree or scaffold_graph_growth")
