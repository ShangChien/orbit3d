# `uni3dar-rs` — Unified Client-Side Rust + WebGPU Inference Engine for `Uni-3DAR-Mol`

Zero-Python, unified Desktop (`Metal / DX12 / Vulkan`) + Browser (`WebAssembly + WebGPU`) client-side inference core for **`Uni-3DAR`** and **`Uni-3DAR-Mol`**.

## Supported `Uni-3DAR-Mol` Full-Stack Capabilities

1. **12-Layer 3D Autoregressive Octree Transformer**:
   - Out-of-place `RMSNorm`, 3D Axial Rotary Position Embeddings (`3D RoPE` at `0.01 Å` grid resolution), 2-Token BFS Block-Causal KV-Cache (`[prev_predicted, next_MASK]`), and `SwiGLU` FFN.
   - Merged `LoRALinear` weights (`q_proj, k_proj, v_proj, out_proj, w1, w2, w3`, `TreeHead`, and 4-axis `AutoRegressivePredictionHead` including `XYZ` LoRA).
   - `F32`, `F16`, **`Q8_0` (`~145 MB`)**, and **`Q4_0` (`~92 MB` for full bundle!)** fused `64x64` (`4x4 vec4`) WGSL GEMM kernels across `4 × 126 MiB` WebGPU storage bindings.
2. **`LoomConditioner` & Scale-Free Atom Budget Guidance**:
   - Multi-property `32-RBF + 16-Fourier` encoder, `num_prefix_tokens` causal KV prefix, step-0 `root_proj` residual, and unclamped `tanh`-capped AdaLN-Zero layer/output modulations.
   - 6-D non-trapping `ScaleFreeBudgetEmbedding`, hierarchical volume packing priors (`L5 -> L1`), and tree-depth-annealed selective Logit-CFG.
3. **3D Steric / Coordination Guards & Terminal Parity Mask**:
   - Bond-order-aware neutral valence saturation guard (`_element_guard_logit_bias` on `axis=0` + `_zbin_guard_logit_bias` on `axis=3`) and closed-shell even-electron terminal parity enforcement.
4. **Two-Stage Speculative Octree Funnel (`m_s1 -> m_s2 -> 1`)**:
   - Stage-1 `OctreeProcessCritic` (`14-D` coarse voxel PCA/bounding geometry) prunes coarse `L7 -> L1` octree trunks before leaf expansion.
   - `PostTreeGraphRefiner` (`SE(3)`-equivariant 3D-RBF attention + radial force seam healing + symmetric `BilinearBondHead` 5-class bond-order probabilities).
   - Stage-2 `OctreeElementProcessCritic` (`42-D` element cloud + `64-RBF` edge-gated GNN + `12-ch` charge-neutralized dipole norms & quadrupole traces).
5. **Pure-Rust `ValenceReconciler` (`RMSD_heavy == 0.0 Å`)**:
   - Branch-and-bound Kekulé bond-order CSP + dipolar formal-charge DP + exact VSEPR hydrogen synthesis & 36-step sphere-tangent steric relaxation with strictly frozen heavy-atom coordinates.

## Quick Start

```bash
# 1. Export a bundled Uni-3DAR-Mol .pt checkpoint to Q4_0 / Q8_0 .u3m
python3 scripts/export_uni3dar_rs.py \
  --checkpoint /path/to/uni3dar_mol_geom_bundled.pt \
  --out pretrained/uni3dar_mol/geom_drugs_q4.u3m \
  --dtype q4_0

# 2. Build native Metal/Vulkan/DX12 CLI
cargo build --release -p uni3dar-rs

# 3. Sample a drug-like 3D molecule with K=2 Two-Stage Speculative Funnel
./target/release/uni3dar-cli sample \
  --model /path/to/geom_drugs_q8.u3m \
  --budget 28 --k 2 --out drug.sdf

# 4. Build Browser WASM + WebGPU package
wasm-pack build crates/uni3dar-rs --target web --release --out-dir web/flowmol-playground/pkg_uni3dar
```
