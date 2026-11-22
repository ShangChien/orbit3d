//! `dpa4-rs`: Unified Rust + WebGPU/Metal/Vulkan/DX12 interatomic potential, analytical VJP
//! force engine, and equivariant descriptor/oracle runtime for DeePMD `DPA-4` (`SeZM`:
//! `Nano`, `Mini`, `Neo`, `Air`, `Plus`), built on `mlip-core`.

pub mod cpu;
pub mod gpu;
pub mod model;
pub mod wasm;
pub mod wigner;

pub use cpu::Dpa4CpuEngine;
pub use gpu::{Dpa4Engine, Dpa4ExecMode, Dpa4GpuContext};
pub use model::{Dpa4BlockWeights, Dpa4Config, Dpa4Weights, DPA4_MAGIC};
