"""
Deterministic Environment Lock & Built-in Pretrained Weight Path Resolvers.
"""

from __future__ import annotations

import os
import random
from typing import Optional
import numpy as np
import torch


def lock_deterministic_environment(seed: int = 42) -> None:
    """Locks Python, NumPy, and PyTorch/CUDA RNG streams for 100% deterministic execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_dpa4_weight_path(builtin_dir: str, scale: str = "Mini") -> str:
    """Resolves the built-in or mounted DPA-4 weight path for `Nano`, `Mini`, `Neo`, `Air`, or `Plus`."""
    clean_scale = scale.replace("DPA4-", "").strip()
    env_dir = os.environ.get("ORBIT3D_PRETRAINED_DIR", builtin_dir)
    candidates = [
        os.path.join(builtin_dir, "dpa4", f"DPA4-{clean_scale}-OMol25-100M-v20260820.pt"),
        os.path.join(builtin_dir, f"DPA4-{clean_scale}-OMol25-100M-v20260820.pt"),
        os.path.join(builtin_dir, f"DPA4-{clean_scale}-OMol25-100M.pt"),
        os.path.join(env_dir, "dpa4", f"DPA4-{clean_scale}-OMol25-100M-v20260820.pt"),
        f"/sjtu-combustion/wjw/chensq/pretrained/DPA4-{clean_scale}-OMol25-100M-v20260820.pt",
        os.path.join(builtin_dir, "DPA-3.1-3M.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find DPA-4 ({scale}) pretrained weight in {candidates}")


def resolve_unimol_weight_path(builtin_dir: str, remove_hs: bool = False) -> Optional[str]:
    """Resolves the built-in FP16-compressed or raw Uni-Mol 3D backbone weight path (`all_h` or `no_h`)."""
    tag = "no_h" if remove_hs else "all_h"
    env_dir = os.environ.get("ORBIT3D_PRETRAINED_DIR", builtin_dir)
    candidates = [
        os.path.join(builtin_dir, "unimol", f"mol_pre_{tag}_fp16.pt"),
        os.path.join(builtin_dir, f"mol_pre_{tag}_fp16.pt"),
        os.path.join(env_dir, "unimol", f"mol_pre_{tag}_fp16.pt"),
        os.path.join(builtin_dir, f"mol_pre_{tag}_220816.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def resolve_hyff_weight_path(builtin_dir: str, tier: str = "vivace") -> Optional[str]:
    """Resolves a packed or mounted HyFF checkpoint directory/safetensors path if present, else None (falls back to HYFF_HOME)."""
    t_clean = tier.lower().replace("hyff-", "").replace("hyff_", "").strip()
    env_dir = os.environ.get("ORBIT3D_PRETRAINED_DIR", builtin_dir)
    hyff_home = os.environ.get("HYFF_HOME", "")
    candidates = [
        os.path.join(builtin_dir, "hyff", "checkpoints", t_clean, "model.safetensors"),
        os.path.join(builtin_dir, "hyff", "checkpoints", t_clean),
        os.path.join(env_dir, "hyff", "checkpoints", t_clean, "model.safetensors"),
        os.path.join(hyff_home, "checkpoints", t_clean, "model.safetensors") if hyff_home else "",
        f"/sjtu-combustion/wjw/chensq/pretrained/hyff/checkpoints/{t_clean}/model.safetensors",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None
