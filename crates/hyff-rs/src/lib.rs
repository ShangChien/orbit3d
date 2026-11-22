//! `hyff-rs`: Unified Rust + WebGPU/Metal/Vulkan/DX12 interatomic potential & analytical VJP
//! force engine for `HyFF-L2` (`Presto`, `Vivace`, `Moderato`), built on `mlip-core`.

pub mod cpu;
pub mod gpu;
pub mod model;
pub mod wasm;

pub use cpu::HyffCpuEngine;
pub use gpu::{HyffEngine, HyffExecMode, HyffGpuContext};
pub use model::{HyffBlockWeights, HyffConfig, HyffWeights, HYFF_MAGIC};
