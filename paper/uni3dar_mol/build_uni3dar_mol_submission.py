#!/usr/bin/env python3
"""Package Uni-3DAR-Mol LaTeX source, BBL, BibTeX, and vector PDF figures into a clean .tar.gz archive."""

from __future__ import annotations

import tarfile
from pathlib import Path


def build_uni3dar_mol_tarball() -> Path:
    paper_dir = Path(__file__).resolve().parent
    out_path = paper_dir / "Uni3DAR_Mol_Submission.tar.gz"

    required_files = [
        "main_uni3dar_mol.tex",
        "main_uni3dar_mol.bbl",
        "references.bib",
        "figures/geom_drugs_pareto_and_scaling.pdf",
        "figures/fig2_oe62_cross_paradigm_scaling.pdf",
        "figures/fig3_oe62_atom_size_scaling.pdf",
        "figures/fig4_uni3dar_m4_ablation.pdf",
    ]

    with tarfile.open(out_path, "w:gz") as tar:
        for rel_path in required_files:
            full_path = paper_dir / rel_path
            if not full_path.exists():
                raise FileNotFoundError(f"Missing required Uni-3DAR-Mol file: {full_path}")
            tar.add(full_path, arcname=rel_path)

    print(
        f"[Uni-3DAR-Mol] Packaged {len(required_files)} files -> {out_path} "
        f"({out_path.stat().st_size / 1024:.1f} KB)"
    )
    return out_path


if __name__ == "__main__":
    build_uni3dar_mol_tarball()
