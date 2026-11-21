# `flowmol-rs`: Unified Client-Side Inference Engine (Desktop + Browser WebGPU)

Pure-Rust + `wgpu` (Metal / DX12 / Vulkan / Browser WebGPU) inference engine for the 118-element `FlowMol-118` 3D molecular foundation family (`small 4.23M`, `standard 12.74M`, `flagship 40.19M`), including SFT, merged-LoRA, Flow-GRPO, and `LoomConditioner` checkpoints.

## Key Engineering Highlights

1. **Zero-Retrain `-34%` FLOPs Algebraic Node-Before-Gather Split**:
   - `scripts/export_flowmol_rs.py` pre-splits node-scalar & node-vector projections (`GVPConv.edge_message[0]` and `EdgeUpdate.edge_update_fn[0]`) so node terms execute in $O(N)$ rather than $O(N^2)$, cutting total forward FLOPs by `31.5%–34.9%` with exact mathematical equivalence.
2. **Single-CommandEncoder `wgpu` Execution**:
   - All ~200 dispatches of a full 6–12 block forward pass use a single static `BindGroup` + dynamic 256B uniform offsets and execute inside **one `queue.submit()`** per denoising step.
3. **Precision-Guarded Mixed Quantization (`.fm118`)**:
   - Heavy `Linear` weights (~99% of parameters) pack as `fp32`, `fp16` (`unpack2x16float`), or `q8_0` (`extractBits` block-32 symmetric INT8), while `<1%` SO(3)-equivariant `Wh/Wcp/Wu`, `LayerNorm`, `periodic_table_feats`, and `LoomConditioner` weights stay strictly in `FP32`.
4. **Zero-Retrain Element Pinning & Hard Constraints**:
   - Supports `allowed_elements`, `allowed_charges`, `required_elements` (CTMC node clamping, e.g. `--require Rh,P,P`), and conditional CFG (`cond_values`, `cfg_scale`, `xyz_cfg_scale`).

## Quick Start

### 1. Export any PyTorch `.pt` checkpoint to `.fm118`
```bash
python scripts/export_flowmol_rs.py \
  --checkpoint pretrained/flowmol/selected_best_ema.pt \
  --corpus-meta pretrained/flowmol/corpus_meta.json \
  --out pretrained/flowmol/flagship_fp16.fm118 \
  --dtype fp16 \
  --dump-parity /tmp/parity_flagship.json
```

### 2. Build & Run Native Desktop CLI (Metal / DX12 / Vulkan)
```bash
cargo build --release --manifest-path crates/flowmol-rs/Cargo.toml

# Verify numerical parity against PyTorch (CPU & GPU)
./crates/flowmol-rs/target/release/flowmol-cli parity \
  --model pretrained/flowmol/flagship_fp16.fm118 \
  --fixture /tmp/parity_flagship.json

# Benchmark native GPU generation speed
./crates/flowmol-rs/target/release/flowmol-cli bench \
  --model pretrained/flowmol/flagship_fp16.fm118 \
  --n-atoms 40 --steps 25 --solver heun

# Generate 3D Rhodium-Phosphine complexes via zero-retrain element pinning -> .sdf
./crates/flowmol-rs/target/release/flowmol-cli sample \
  --model pretrained/flowmol/flagship_fp16.fm118 \
  --domain tmqm --n-atoms 68 --require Rh,P,P --charges -1,0,1,2 \
  --n-mols 2 --out rh_complexes.sdf
```

### 3. Build Browser WebAssembly + WebGPU Package
```bash
wasm-pack build crates/flowmol-rs --target web --release --out-dir ../../web/flowmol-playground/pkg
```
Serve `web/flowmol-playground/` over any static HTTP server (e.g., `python3 -m http.server 8080 -d web/flowmol-playground`) and open in Chrome / Edge / Safari 18+.

## Measured Benchmarks & Numerical Parity (Apple M5 `wgpu/Metal`)

- **WASM Runtime Size**: `444 KB` raw (`172 KB` gzipped) — zero external JS/ONNX runtime dependencies.
- **PyTorch Forward Parity** (`flagship 40.19M`, 12 SE(3) blocks, Pass 1 + Pass 2 Self-Conditioning):
  - `fp32`: coord `mean_abs_diff = 2e-6 Å` (`max = 9e-6 Å`), logits `max = 1.56e-4`
  - `fp16`: coord `mean_abs_diff = 3.0e-4 Å` (`max = 1.0e-3 Å`), logits `max = 3.34e-2`
  - `q8_0`: coord `mean_abs_diff = 1.1e-2 Å` (`max = 3.8e-2 Å`), logits `max = 0.73`
  - `SFT LoRA + LoomConditioner + FragmentCond`: `fp32 max = 1.29e-4`, `fp16 max = 3.47e-4`
- **Single-Molecule End-to-End Generation Speed (`FP16`, 25 steps)**:
  - `small` (`4.23M`, `8.73 MB`): `315 ms/mol` at `N=29` (Euler), `429 ms/mol` at `N=45` (Euler), `839 ms/mol` at `N=45` (Heun, 49 NFE)
  - `standard` (`12.74M`, `26.03 MB`): `615 ms/mol` at `N=29` (Euler), `924 ms/mol` at `N=45` (Euler), `1.81 s/mol` at `N=45` (Heun, 49 NFE)
  - `flagship` (`40.19M`, `81.48 MB`): `1.34 s/mol` at `N=29` (Euler), `2.14 s/mol` at `N=45` (Euler), `4.20 s/mol` at `N=45` (Heun, 49 NFE)

