"""Package the clean Nature Computational Science LaTeX submission tarball."""

from __future__ import annotations

import pathlib
import tarfile


def main() -> None:
    """Create deterministic ``UniLoom_NCS_Submission.tar.gz`` with source files only."""
    ncs_dir = pathlib.Path(__file__).resolve().parent
    required_files = [
        "main_ncs.tex",
        "main_ncs.bbl",
        "references.bib",
        "figures/ed_fig1_stage_progression.pdf",
        "figures/ed_fig2_gp_atlas_1d_homo.pdf",
        "figures/ed_fig3_oe62_cross_paradigm_scaling.pdf",
        "figures/ed_fig4_oe62_atom_size_scaling.pdf",
        "figures/ed_fig5_uni3dar_m4_ablation_and_sdl.pdf",
        "figures/fig1_uniloom_architecture.pdf",
        "figures/fig2_audit_and_hotswap.pdf",
        "figures/fig3_gp_atlas_2d_joint.pdf",
        "figures/fig4_zwitterion_and_synthesizability.pdf",
        "figures/fig5_grounded_rsi_closed_loop.pdf",
    ]
    missing = [rel for rel in required_files if not (ncs_dir / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required NCS submission files: {missing}")

    out_tar = ncs_dir / "UniLoom_NCS_Submission.tar.gz"
    with tarfile.open(out_tar, "w:gz") as tar:
        for rel in sorted(required_files):
            src = ncs_dir / rel
            tar.add(src, arcname=rel)

    print(
        f"[build_ncs_submission] Wrote {len(required_files)} files -> "
        f"{out_tar} ({out_tar.stat().st_size / 1024:.1f} KiB)"
    )


if __name__ == "__main__":
    main()
