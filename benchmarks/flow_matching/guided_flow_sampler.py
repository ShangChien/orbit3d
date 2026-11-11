from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import types
import torch

import dgl
from propmolflow.analysis.molecule_builder import SampledMolecule
from propmolflow.data_processing.utils import (
    build_edge_idxs,
    get_batch_idxs,
    get_edge_batch_idxs,
    get_upper_edge_mask,
)

from .dpa_potential import DPAPotentialGuidance
from .interval_guidance import (
    HingeIntervalPropertyGuidance,
    orthogonal_manifold_projection,
    zero_com_per_molecule,
)


@dataclass
class GuidanceConfig:
    mode: str = "dpa_orthogonal_relaxed"  # 'baseline', 'naive_guidance', 'dpa_orthogonal', 'dpa_orthogonal_relaxed', 'dpa_co_guided_relaxed'
    lambda_prop: float = 1.5              # Property interval hinge gradient guidance scale
    eta_phys: float = 0.025               # DPA-3 physical force drift scale during flow (A / (eV/A))
    t_start: float = 0.50                 # Start time for manifold-projected guidance
    t_end: float = 1.00                   # End time for manifold-projected guidance
    guide_every_n_steps: int = 1          # Apply guidance every N flow steps
    max_guidance_disp: float = 0.08       # Max per-step coordinate shift (Angstrom)
    atom_level_proj: bool = True          # Per-atom orthogonal projection against opposing DPA force
    mol_level_proj: bool = True           # Per-molecule 3N-D orthogonal projection
    discrete_guidance_scale: float = 0.0  # Top-1: Doob's h-transform tilting scale on CTMC atom/bond probabilities
    discrete_t_start: float = 0.20        # Start time for discrete CTMC topology guidance
    discrete_t_end: float = 0.85          # End time for discrete CTMC topology guidance
    iso_barrier_relax: bool = True        # Top-2: Iso-Property Manifold + Interval-Barrier Backtracking relaxation
    terminal_relax_steps: int = 12        # Number of post-flow DPA-3 conformational relaxation steps
    terminal_relax_lr: float = 0.008      # Step size for post-flow DPA-3 relaxation
    terminal_max_disp: float = 0.025      # Max per-step displacement during terminal relaxation


class GuidedFlowSampler:
    """
    3D Equivariant Flow Matching Sampler with DPA-3 MLIP Orthogonal Manifold-Projected
    Interval Guidance.
    """

    def __init__(
        self,
        flow_ckpt_path: Union[str, Path],
        propmolflow_root: Union[str, Path],
        property_names: List[str],
        dpa_model_path: Optional[Union[str, Path]] = None,
        dpa_head: str = "SPICE2",
        device: str = "cuda:0",
    ):
        self.flow_ckpt_path = Path(flow_ckpt_path)
        self.propmolflow_root = Path(propmolflow_root)
        self.property_names = list(property_names)
        self.device = torch.device(device)

        import os
        from propmolflow.models.flowmol import FlowMol

        prev_cwd = os.getcwd()
        try:
            os.chdir(self.propmolflow_root)
            self.flow_model: FlowMol = FlowMol.load_from_checkpoint(
                str(self.flow_ckpt_path), map_location=self.device, weights_only=False
            )
        finally:
            os.chdir(prev_cwd)
        self.flow_model.to(self.device)
        self.flow_model.eval()

        self.prop_guidance = HingeIntervalPropertyGuidance(
            propmolflow_root=self.propmolflow_root,
            property_names=self.property_names,
            device=str(self.device),
        )

        self.dpa_guidance: Optional[DPAPotentialGuidance] = None
        if dpa_model_path is not None:
            self.dpa_guidance = DPAPotentialGuidance(
                model_path=str(dpa_model_path),
                head=dpa_head,
                atom_type_map=self.flow_model.atom_type_map,
                device=str(self.device),
            )

    @torch.no_grad()
    def sample_Within_Interval(
        self,
        n_atoms: torch.Tensor,
        target_intervals_norm: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        primary_cond_prop: str,
        guidance_cfg: GuidanceConfig,
        n_timesteps: int = 100,
        seed: Optional[int] = None,
    ) -> Tuple[List[SampledMolecule], Dict[str, float]]:
        """
        Sample 3D molecules conditioned on target property intervals [y_min, y_max]
        with DPA-3 Orthogonal Manifold-Projected Guidance.

        Args:
            n_atoms: (B,) integer tensor of atom counts per molecule.
            target_intervals_norm: dict mapping prop -> (y_min_norm, y_max_norm), each (B,).
            primary_cond_prop: the property name that the base FlowMol checkpoint conditions on.
            guidance_cfg: GuidanceConfig controlling guidance mode and hyperparameters.
            n_timesteps: number of ODE / CTMC integration steps.
            seed: optional random seed for reproducible paired comparisons across methods.

        Returns:
            molecules: list of SampledMolecule objects.
            diagnostics: dictionary of trajectory statistics (conflict rate, DPA energy, etc.).
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        n_atoms = n_atoms.to(self.device)
        batch_size = n_atoms.shape[0]

        # Base FlowMol conditioning uses the midpoint of the primary property interval (normalized)
        y_min_p, y_max_p = target_intervals_norm[primary_cond_prop]
        property_context = (0.5 * (y_min_p + y_max_p)).to(self.device)
        prop_values_list = [float(v) for v in property_context.detach().cpu().view(-1).tolist()]

        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            n_i = int(n_atoms_i.item())
            edge_idxs_dict[n_i] = build_edge_idxs(n_atoms_i)

        graphs = []
        for n_atoms_i in n_atoms:
            n_i = int(n_atoms_i.item())
            edge_idxs = edge_idxs_dict[n_i]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_i, device=self.device)
            graphs.append(g_i)

        g = dgl.batch(graphs)
        upper_edge_mask = get_upper_edge_mask(g)
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # Sample prior x_0, a_0, c_0, e_0
        g = self.flow_model.sample_prior(g, node_batch_idx, upper_edge_mask)

        # Integrate with guided step hook
        vf_module = self.flow_model.vector_field
        orig_step = vf_module.step
        orig_forward = vf_module.forward

        step_counter = {"idx": 0, "dpa_calls": 0}
        cached_f_phys = {"val": None}
        conflict_fracs = []
        cos_sims = []
        hinge_losses = []

        use_prop_guidance = (
            guidance_cfg.mode in (
                "naive_guidance",
                "dpa_orthogonal",
                "dpa_orthogonal_relaxed",
                "dpa_co_guided_relaxed",
            )
            and guidance_cfg.lambda_prop > 0.0
        )
        use_dpa_proj = guidance_cfg.mode in (
            "dpa_orthogonal",
            "dpa_orthogonal_relaxed",
            "dpa_co_guided_relaxed",
        )
        use_discrete_guidance = (
            guidance_cfg.discrete_guidance_scale > 0.0
            or guidance_cfg.mode == "dpa_co_guided_relaxed"
        )

        def guided_forward(
            vf_self,
            g: dgl.DGLGraph,
            t: torch.Tensor,
            node_batch_idx: torch.Tensor,
            upper_edge_mask: torch.Tensor,
            apply_softmax: bool = False,
            remove_com: bool = False,
        ):
            dst_dict = orig_forward(
                g,
                t=t,
                node_batch_idx=node_batch_idx,
                upper_edge_mask=upper_edge_mask,
                apply_softmax=apply_softmax,
                remove_com=remove_com,
            )
            if not use_discrete_guidance or not apply_softmax:
                return dst_dict

            t_scalar = float(t[0].item()) if t.ndim == 1 else float(t[0, 0].item())
            if (
                not (guidance_cfg.discrete_t_start <= t_scalar <= guidance_cfg.discrete_t_end)
                or (step_counter["idx"] % 2 != 0)
            ):
                return dst_dict

            # Top-1: Straight-Through Discrete-Continuous Gradient + Doob's h-transform Tilting
            x_1_hat = dst_dict["x"]
            p_a = dst_dict["a"]  # (N, 5)
            p_e_upper = dst_dict["e"]  # (E_upper, 5)
            e_full = torch.zeros(
                (g.num_edges(), p_e_upper.shape[-1]),
                dtype=p_e_upper.dtype,
                device=p_e_upper.device,
            )
            e_full[upper_edge_mask] = p_e_upper
            e_full[~upper_edge_mask] = p_e_upper

            blend_t = float(t_scalar)
            x_eval_disc = (1.0 - blend_t) * x_1_hat + blend_t * g.ndata["x_t"]

            _, _, hinge_mol_disc, extra_disc = self.prop_guidance.compute_hinge_interval_gradient(
                g=g,
                coords=x_eval_disc,
                atom_probs=p_a,
                edge_probs_full=e_full,
                target_intervals_norm=target_intervals_norm,
                node_batch_idx=node_batch_idx,
                return_discrete_grads=True,
            )

            disc_scale = (
                guidance_cfg.discrete_guidance_scale
                if guidance_cfg.discrete_guidance_scale > 0.0
                else 0.85
            ) * (1.0 - 0.5 * t_scalar)

            g_atom = extra_disc["g_atom"]  # (N, 5)
            g_edge_full = extra_disc["g_edge"]  # (E, 5)

            # Tilt heavy-atom probabilities while preserving H probability to maintain closed-shell valency
            a_is_masked = (g.ndata["a_t"][:, -1:] > 0.5).to(p_a.dtype)
            mol_viol_node = (hinge_mol_disc[node_batch_idx] > 1e-6).unsqueeze(-1).to(p_a.dtype)
            heavy_mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0], dtype=p_a.dtype, device=p_a.device).unsqueeze(0)
            g_a_heavy = g_atom * heavy_mask
            g_a_mean = g_a_heavy.sum(dim=-1, keepdim=True) / 4.0
            g_a_c = (g_a_heavy - g_a_mean) * heavy_mask
            g_a_norm = g_a_c / (g_a_c.std(dim=-1, keepdim=True) + 1e-5)
            g_a_tilt = torch.clamp(
                disc_scale * g_a_norm * a_is_masked * mol_viol_node, min=-1.0, max=1.0
            )
            dst_dict["a"] = torch.softmax(torch.log(p_a.clamp(min=1e-8)) + g_a_tilt, dim=-1)

            e_is_masked = (g.edata["e_t"][upper_edge_mask, -1:] > 0.5).to(p_e_upper.dtype)
            edge_mol_idx = edge_batch_idx[upper_edge_mask]
            mol_viol_edge = (hinge_mol_disc[edge_mol_idx] > 1e-6).unsqueeze(-1).to(p_e_upper.dtype)
            g_e_upper = 0.5 * (g_edge_full[upper_edge_mask] + g_edge_full[~upper_edge_mask])
            g_e_c = g_e_upper - g_e_upper.mean(dim=-1, keepdim=True)
            g_e_norm = g_e_c / (g_e_c.std(dim=-1, keepdim=True) + 1e-5)
            g_e_tilt = torch.clamp(
                0.6 * disc_scale * g_e_norm * e_is_masked * mol_viol_edge, min=-0.8, max=0.8
            )
            dst_dict["e"] = torch.softmax(torch.log(p_e_upper.clamp(min=1e-8)) + g_e_tilt, dim=-1)
            return dst_dict

        def guided_step(
            vf_self,
            g_curr: dgl.DGLGraph,
            s_i: torch.Tensor,
            t_i: torch.Tensor,
            alpha_t_i: torch.Tensor,
            alpha_s_i: torch.Tensor,
            alpha_t_prime_i: torch.Tensor,
            node_batch_idx_curr: torch.Tensor,
            *step_args,
            **kwargs,
        ):
            step_counter["idx"] += 1
            t_val = float(t_i.item())
            dt = float((s_i - t_i).item())

            # First execute standard FlowMol CTMC/ODE step
            g_curr = orig_step(
                g_curr,
                s_i,
                t_i,
                alpha_t_i,
                alpha_s_i,
                alpha_t_prime_i,
                node_batch_idx_curr,
                *step_args,
                **kwargs,
            )

            # Check if we are in the active guidance time window [t_start, t_end]
            if (
                use_prop_guidance
                and guidance_cfg.t_start <= t_val <= guidance_cfg.t_end
                and (step_counter["idx"] % guidance_cfg.guide_every_n_steps == 0)
            ):
                # Retrieve predicted clean endpoint features stored by orig_step
                x_1_pred = g_curr.ndata.get("x_1_pred", g_curr.ndata["x_t"])
                a_1_pred = g_curr.ndata.get("a_1_pred", g_curr.ndata["a_t"][:, :5])
                e_1_pred = g_curr.edata.get("e_1_pred", g_curr.edata["e_t"][:, :5])

                # Evaluate property gradient at denoised geometry x_eval
                # Smoothly transition from x_1_pred to x_t as t -> 1
                blend = float(alpha_s_i[0].item())
                x_eval = (1.0 - blend) * x_1_pred + blend * g_curr.ndata["x_t"]

                g_prop, _, hinge_mol = self.prop_guidance.compute_hinge_interval_gradient(
                    g=g_curr,
                    coords=x_eval,
                    atom_probs=a_1_pred,
                    edge_probs_full=e_1_pred,
                    target_intervals_norm=target_intervals_norm,
                    node_batch_idx=node_batch_idx_curr,
                )
                hinge_losses.append(float(hinge_mol.mean().item()))

                if use_dpa_proj and self.dpa_guidance is not None:
                    # Reuse cached DPA-3 force every 2 guidance steps to double MLIP throughput
                    if cached_f_phys["val"] is None or (step_counter["dpa_calls"] % 2 == 0):
                        a_idx = a_1_pred[:, :5].argmax(dim=-1)
                        _, f_phys = self.dpa_guidance.compute_energy_and_forces(
                            coords=x_eval,
                            dataset_atom_indices=a_idx,
                            node_batch_idx=node_batch_idx_curr,
                            clip_forces=True,
                            zero_com_forces=True,
                        )
                        cached_f_phys["val"] = f_phys
                    else:
                        f_phys = cached_f_phys["val"]
                    step_counter["dpa_calls"] += 1
                    g_tilde, proj_stats = orthogonal_manifold_projection(
                        g_prop=g_prop,
                        f_phys=f_phys,
                        node_batch_idx=node_batch_idx_curr,
                        batch_size=g_curr.batch_size,
                        atom_level_projection=guidance_cfg.atom_level_proj,
                        mol_level_projection=guidance_cfg.mol_level_proj,
                    )
                    conflict_fracs.append(proj_stats["conflict_frac"])
                    cos_sims.append(proj_stats["mean_cos_sim"])

                    # Combine orthogonal manifold-projected property gradient + DPA-3 physical drift
                    delta_x = dt * (guidance_cfg.lambda_prop * g_tilde + guidance_cfg.eta_phys * f_phys)
                else:
                    # Naive property gradient without DPA-3 manifold projection
                    delta_x = dt * (guidance_cfg.lambda_prop * g_prop)

                # Cap per-step displacement norm for numerical stability
                disp_norm = torch.norm(delta_x, dim=-1, keepdim=True) + 1e-8
                delta_x = delta_x * torch.clamp(guidance_cfg.max_guidance_disp / disp_norm, max=1.0)
                delta_x = zero_com_per_molecule(delta_x, node_batch_idx_curr, g_curr.batch_size)

                g_curr.ndata["x_t"] = g_curr.ndata["x_t"] + delta_x

            return g_curr

        try:
            vf_module.step = types.MethodType(guided_step, vf_module)
            if use_discrete_guidance:
                vf_module.forward = types.MethodType(guided_forward, vf_module)
            g = vf_module.integrate(
                g,
                node_batch_idx,
                upper_edge_mask=upper_edge_mask,
                n_timesteps=n_timesteps,
                visualize=False,
                conditional_generation=True,
                property_name=primary_cond_prop,
                multilple_values_to_one_property=prop_values_list,
                training_mode=True,
                properties_handle_method="concatenate_sum",
                normalization_file_path=None,
            )
        finally:
            vf_module.step = orig_step
            vf_module.forward = orig_forward

        # Post-flow Terminal Relaxation on the DPA-3 potential surface
        # Top-2: Iso-Property Manifold Projection + Per-Molecule Interval-Barrier Backtracking
        init_dpa_E = None
        final_dpa_E = None
        if self.dpa_guidance is not None:
            a_final_idx = g.ndata["a_1"][:, :5].argmax(dim=-1)
            if (
                guidance_cfg.mode in ("dpa_orthogonal_relaxed", "dpa_co_guided_relaxed")
                and guidance_cfg.terminal_relax_steps > 0
            ):
                x_best = g.ndata["x_1"].clone()
                a_1_final = g.ndata["a_1"][:, :5]
                e_1_final = g.edata["e_1"][:, :5]
                init_E, best_f_phys = self.dpa_guidance.compute_energy_and_forces(
                    coords=x_best,
                    dataset_atom_indices=a_final_idx,
                    node_batch_idx=node_batch_idx,
                    clip_forces=True,
                    zero_com_forces=True,
                )
                best_E = init_E.clone()
                best_g_prop, _, best_hinge, best_extra = self.prop_guidance.compute_hinge_interval_gradient(
                    g=g,
                    coords=x_best,
                    atom_probs=a_1_final,
                    edge_probs_full=e_1_final,
                    target_intervals_norm=target_intervals_norm,
                    node_batch_idx=node_batch_idx,
                    return_midpoint_grad=True,
                )
                best_j_mid = best_extra["j_mid"]
                lr_mol = torch.full(
                    (g.batch_size,),
                    guidance_cfg.terminal_relax_lr,
                    dtype=x_best.dtype,
                    device=x_best.device,
                )

                for _ in range(guidance_cfg.terminal_relax_steps):
                    g_tilde, _ = orthogonal_manifold_projection(
                        g_prop=best_g_prop,
                        f_phys=best_f_phys,
                        node_batch_idx=node_batch_idx,
                        batch_size=g.batch_size,
                        atom_level_projection=True,
                        mol_level_projection=True,
                    )
                    # Top-2: Project physical force orthogonal to opposing midpoint property gradient
                    # (Iso-Property Manifold Tangent Space) and opposing hinge gradient
                    f_iso, _ = orthogonal_manifold_projection(
                        g_prop=best_f_phys,
                        f_phys=best_j_mid,
                        node_batch_idx=node_batch_idx,
                        batch_size=g.batch_size,
                        atom_level_projection=True,
                        mol_level_projection=True,
                    )
                    f_iso, _ = orthogonal_manifold_projection(
                        g_prop=f_iso,
                        f_phys=best_g_prop,
                        node_batch_idx=node_batch_idx,
                        batch_size=g.batch_size,
                        atom_level_projection=True,
                        mol_level_projection=True,
                    )
                    in_interval_node = (best_hinge <= 1e-7)[node_batch_idx].unsqueeze(-1)
                    lr_node = lr_mol[node_batch_idx].unsqueeze(-1)
                    disp_in = lr_node * f_iso
                    disp_out = lr_node * f_iso + 0.045 * g_tilde
                    disp = torch.where(in_interval_node, disp_in, disp_out)

                    disp_norm = torch.norm(disp, dim=-1, keepdim=True) + 1e-8
                    disp = disp * torch.clamp(guidance_cfg.terminal_max_disp / disp_norm, max=1.0)
                    disp = zero_com_per_molecule(disp, node_batch_idx, g.batch_size)
                    x_cand = x_best + disp

                    cand_g_prop, _, cand_hinge, cand_extra = self.prop_guidance.compute_hinge_interval_gradient(
                        g=g,
                        coords=x_cand,
                        atom_probs=a_1_final,
                        edge_probs_full=e_1_final,
                        target_intervals_norm=target_intervals_norm,
                        node_batch_idx=node_batch_idx,
                        return_midpoint_grad=True,
                    )
                    cand_E, cand_f_phys = self.dpa_guidance.compute_energy_and_forces(
                        coords=x_cand,
                        dataset_atom_indices=a_final_idx,
                        node_batch_idx=node_batch_idx,
                        clip_forces=True,
                        zero_com_forces=True,
                    )

                    if guidance_cfg.iso_barrier_relax:
                        # Per-Molecule Interval-Barrier Acceptance:
                        # 1) Molecules already inside interval MUST stay inside interval (cand_hinge <= 1e-7).
                        # 2) Molecules outside interval MUST not increase their interval violation.
                        was_in = best_hinge <= 1e-7
                        stays_in = cand_hinge <= 1e-7
                        improves_out = (~was_in) & (cand_hinge <= best_hinge + 1e-7)
                        accept_mol = (was_in & stays_in) | improves_out
                        accept_node = accept_mol[node_batch_idx].unsqueeze(-1)

                        x_best = torch.where(accept_node, x_cand, x_best)
                        best_f_phys = torch.where(accept_node, cand_f_phys, best_f_phys)
                        best_g_prop = torch.where(accept_node, cand_g_prop, best_g_prop)
                        best_j_mid = torch.where(accept_node, cand_extra["j_mid"], best_j_mid)
                        best_hinge = torch.where(accept_mol, cand_hinge, best_hinge)
                        best_E = torch.where(accept_mol, cand_E, best_E)
                        # Backtrack step size for molecules that hit the interval barrier
                        lr_mol = torch.where(accept_mol, lr_mol, lr_mol * 0.45)
                    else:
                        x_best = x_cand
                        best_f_phys = cand_f_phys
                        best_g_prop = cand_g_prop
                        best_j_mid = cand_extra["j_mid"]
                        best_hinge = cand_hinge
                        best_E = cand_E

                g.ndata["x_1"] = x_best
                init_dpa_E = float(init_E.mean().item())
                final_dpa_E = float(best_E.mean().item())
            else:
                mol_E, _ = self.dpa_guidance.compute_energy_and_forces(
                    coords=g.ndata["x_1"],
                    dataset_atom_indices=a_final_idx,
                    node_batch_idx=node_batch_idx,
                )
                init_dpa_E = float(mol_E.mean().item())
                final_dpa_E = init_dpa_E

        g.edata["ue_mask"] = upper_edge_mask
        g = g.to("cpu")

        ctmc_mol = getattr(self.flow_model, "parameterization", "ctmc") == "ctmc"
        molecules = []
        for g_i in dgl.unbatch(g):
            molecules.append(
                SampledMolecule(
                    g_i,
                    self.flow_model.atom_type_map,
                    ctmc_mol=ctmc_mol,
                    exclude_charges=self.flow_model.exclude_charges,
                )
            )

        diagnostics = {
            "mean_conflict_frac": sum(conflict_fracs) / len(conflict_fracs) if conflict_fracs else 0.0,
            "mean_cos_sim": sum(cos_sims) / len(cos_sims) if cos_sims else 0.0,
            "final_hinge_loss": hinge_losses[-1] if hinge_losses else 0.0,
            "init_dpa_energy_eV": init_dpa_E if init_dpa_E is not None else 0.0,
            "final_dpa_energy_eV": final_dpa_E if final_dpa_E is not None else 0.0,
        }
        return molecules, diagnostics
