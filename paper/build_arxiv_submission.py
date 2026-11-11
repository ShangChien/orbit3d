#!/usr/bin/env python3
"""Package ORBIT-3D LaTeX manuscript and figures into an arXiv-ready .tar.gz archive."""

import tarfile
from pathlib import Path


def build_arxiv_tarball() -> Path:
    paper_dir = Path(__file__).resolve().parent
    out_path = paper_dir / "ORBIT3D_arXiv_Submission.tar.gz"

    required_files = [
        "ORBIT3D_arXiv_Paper.tex",
        "ORBIT3D_arXiv_Paper.bbl",
        "references.bib",
        "figures/fig1_orbit3d_vs_propmolflow.png",
        "figures/fig2_five_stage_ablation.png",
        "figures/fig3_dpa4_scaling_pareto.png",
        "figures/fig4_pluggable_oracles_relaxation.png",
        "figures/fig5_latency_and_hardware_profile.png",
        "figures/fig6_dataset_distributions.png",
        "figures/fig7_hyff_vs_dpa4_pareto_and_spin.png",
        "figures/fig8_optoelectronic_paradigms_and_forcefields.png",
    ]

    with tarfile.open(out_path, "w:gz") as tar:
        for rel_path in required_files:
            full_path = paper_dir / rel_path
            if not full_path.exists():
                raise FileNotFoundError(f"Missing required arXiv file: {full_path}")
            tar.add(full_path, arcname=rel_path)

    print(f"[arXiv] Packaged {len(required_files)} files -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


if __name__ == "__main__":
    build_arxiv_tarball()
