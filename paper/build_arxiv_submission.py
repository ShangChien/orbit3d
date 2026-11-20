#!/usr/bin/env python3
"""Package Uni-Loom LaTeX source, synchronized BBL, BibTeX, and vector PDF figures into an arXiv .tar.gz archive."""

from __future__ import annotations

import tarfile
from pathlib import Path


def build_arxiv_tarball() -> Path:
    paper_dir = Path(__file__).resolve().parent
    out_path = paper_dir / "UniLoom_arXiv_Submission.tar.gz"

    required_files = [
        "main.tex",
        "main.bbl",
        "references.bib",
        "figures/fig1_uniloom_architecture.pdf",
        "figures/fig2_stage_progression.pdf",
        "figures/fig3_audit_and_hotswap.pdf",
        "figures/fig4_gp_atlas_1d_homo.pdf",
        "figures/fig5_gp_atlas_2d_joint.pdf",
        "figures/fig6_zwitterion_and_synthesizability.pdf",
        "figures/fig7_grounded_rsi_closed_loop.pdf",
        "figures/fig8_oe62_cross_paradigm_scaling.pdf",
        "figures/fig9_oe62_atom_size_scaling.pdf",
        "figures/fig10_uni3dar_m4_ablation_and_sdl.pdf",
    ]

    with tarfile.open(out_path, "w:gz") as tar:
        for rel_path in required_files:
            full_path = paper_dir / rel_path
            if not full_path.exists():
                raise FileNotFoundError(f"Missing required arXiv file: {full_path}")
            tar.add(full_path, arcname=rel_path)

    print(
        f"[arXiv] Packaged {len(required_files)} files -> {out_path} "
        f"({out_path.stat().st_size / 1024:.1f} KB)"
    )
    return out_path


if __name__ == "__main__":
    build_arxiv_tarball()
