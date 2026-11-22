use crate::config::PrecomputedCond;
use crate::format::{ModelBundle, QuantMode};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use wgpu::util::DeviceExt;

const UNIFORM_ALIGN: u32 = 256;
/// 126 MiB = 33,030,144 u32 words (< 128 MiB WebGPU default `maxStorageBufferBindingSize`).
const W_CHUNK_U32: usize = 33_030_144;
/// Maximum `(slot * kv_tokens)` stored simultaneously in `kv_buf` (`1664 * 12 * 2 * 768 * 4 B = 117.0 MiB < 126 MiB`).
pub const KV_TOTAL_SLOT_TOKENS: usize = 1664;
/// Maximum simultaneous `(branches * seq_q)` tokens in a single step batch.
pub const MAX_STEP_TOKENS: usize = 128;
/// Maximum branches with distinct `PrecomputedCond` modulations in a single batch.
pub const MAX_MOD_BRANCHES: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct OpParams {
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32,
    p5: u32,
    p6: u32,
    p7: u32,
    p8: u32,
    p9: u32,
    p10: u32,
    p11: u32,
    f0: f32,
    f1: f32,
    u14: u32,
    u15: u32,
}

#[derive(Clone, Copy, Debug)]
struct GpuLinearHandle {
    mode: QuantMode,
    out_cols: u32,
    orig_in_cols: u32,
    pad_in_cols: u32,
    blocks_per_row: u32,
    w_u32_off: u32,
    bias_off: u32, // 0xFFFF_FFFF if none
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepHeadMode {
    HiddenOnly,
    HiddenAndTree,
}

pub struct GpuStepOutput {
    pub last_hidden: Vec<f32>,         // (b_run * emb_dim)
    pub tree_logits: Option<Vec<f32>>, // Some(b_run * 255) when StepHeadMode::HiddenAndTree
}

#[derive(Clone, Copy, Debug)]
struct WsLayout {
    h: u32,
    x_norm: u32,
    q: u32,
    k: u32,
    v: u32,
    attn_cat: u32,
    gate: u32,
    up: u32,
    rope_pos: u32,
    branch_meta: u32,
    last_out: u32,
    head_mid: u32,
    head_logits: u32,
    repack_staging: u32,
    total_floats: usize,
}

impl WsLayout {
    fn compute(emb_dim: usize, ffn_dim: usize) -> Self {
        let t = MAX_STEP_TOKENS;
        let mut cur = 0usize;
        let mut alloc = |len: usize| -> u32 {
            let aligned = (cur + 3) & !3;
            cur = aligned + len.max(4);
            aligned as u32
        };
        let h = alloc(t * emb_dim);
        let x_norm = alloc(t * emb_dim);
        let q = alloc(t * emb_dim);
        let k = alloc(t * emb_dim);
        let v = alloc(t * emb_dim);
        let attn_cat = alloc(t * emb_dim);
        let gate = alloc(t * ffn_dim);
        let up = alloc(t * ffn_dim);
        let rope_pos = alloc(t * 3);
        let branch_meta = alloc(t * 2);
        let last_out = alloc(t * emb_dim);
        let head_mid = alloc(t * emb_dim);
        let head_logits = alloc(t * 608);
        // Reserve 2 MiB (524,288 floats) for zero-CPU-readback KV slot compaction (`kv_buf -> ws -> kv_buf`)
        let repack_staging = alloc(524_288);
        Self {
            h,
            x_norm,
            q,
            k,
            v,
            attn_cat,
            gate,
            up,
            rope_pos,
            branch_meta,
            last_out,
            head_mid,
            head_logits,
            repack_staging,
            total_floats: cur,
        }
    }
}

struct ParamPacker {
    w_u32: Vec<u32>,
    f32_vec: Vec<f32>,
    linears: HashMap<String, GpuLinearHandle>,
    f32_offs: HashMap<String, u32>,
}

impl ParamPacker {
    fn new() -> Self {
        Self {
            w_u32: Vec::new(),
            f32_vec: Vec::new(),
            linears: HashMap::new(),
            f32_offs: HashMap::new(),
        }
    }

    fn push_f32(&mut self, name: &str, slice: &[f32]) -> u32 {
        let off = self.f32_vec.len() as u32;
        self.f32_vec.extend_from_slice(slice);
        self.f32_offs.insert(name.to_string(), off);
        off
    }

    fn push_linear(&mut self, bundle: &ModelBundle, prefix: &str) -> Result<GpuLinearHandle, String> {
        let lv = bundle.linear_view(prefix)?;
        debug_assert_eq!(lv.weight_bytes.len() % 4, 0);
        let words = lv.weight_bytes.len() / 4;
        let cur_chunk = self.w_u32.len() / W_CHUNK_U32;
        let end_chunk = (self.w_u32.len() + words.saturating_sub(1)) / W_CHUNK_U32;
        if words > 0 && end_chunk > cur_chunk {
            let next_start = (cur_chunk + 1) * W_CHUNK_U32;
            self.w_u32.resize(next_start, 0u32);
        }
        let w_u32_off = self.w_u32.len() as u32;
        let w_slice: &[u32] = bytemuck::cast_slice(lv.weight_bytes);
        self.w_u32.extend_from_slice(w_slice);

        let bias_off = if let Some(b) = lv.bias {
            let bo = self.f32_vec.len() as u32;
            self.f32_vec.extend_from_slice(b);
            bo
        } else {
            0xFFFF_FFFF
        };

        let h = GpuLinearHandle {
            mode: lv.mode,
            out_cols: lv.out_features as u32,
            orig_in_cols: lv.orig_cols as u32,
            pad_in_cols: lv.pad_cols as u32,
            blocks_per_row: lv.blocks_per_row as u32,
            w_u32_off,
            bias_off,
        };
        self.linears.insert(prefix.to_string(), h);
        Ok(h)
    }
}

#[derive(Clone, Copy)]
enum PipeKind {
    Gemm,
    Gemv,
    RmsNormMod,
    RopeKvWrite,
    BatchedKvAttn,
    SwigluMul,
}

pub struct WgpuDecoderBackend {
    pub adapter_name: String,
    pub backend_api: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    p_gemm: wgpu::ComputePipeline,
    p_gemv: wgpu::ComputePipeline,
    p_rmsnorm: wgpu::ComputePipeline,
    p_rope_kv: wgpu::ComputePipeline,
    p_kv_attn: wgpu::ComputePipeline,
    p_swiglu: wgpu::ComputePipeline,
    f32_params_buf: wgpu::Buffer,
    mod_dynamic_off: u32,
    mod_b_stride: u32,
    linears: HashMap<String, GpuLinearHandle>,
    f32_offs: HashMap<String, u32>,
    ws_layout: WsLayout,
    kv_buf: wgpu::Buffer,
    ws_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    uniform_cap_slots: usize,
    readback_buf: wgpu::Buffer,
    readback_cap_bytes: u64,
    bind_group: wgpu::BindGroup,
    num_layers: usize,
    emb_dim: usize,
    num_head: usize,
    head_dim: usize,
    ffn_dim: usize,
    xyz_res: f32,
}

impl WgpuDecoderBackend {
    pub async fn new_async(bundle: &ModelBundle) -> Result<Self, String> {
        let bb = &bundle.header.backbone;
        if bb.head_dim > 64 {
            return Err(format!("Unsupported head_dim={} > 64 for WebGPU kernel", bb.head_dim));
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "No WebGPU / Metal / DX12 / Vulkan GPU adapter found".to_string())?;

        let info = adapter.get_info();
        let al = adapter.limits();
        let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(al.clone());
        limits.max_storage_buffers_per_shader_stage = al.max_storage_buffers_per_shader_stage;
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_buffer_size = al.max_buffer_size;
        limits.max_compute_workgroup_storage_size = al.max_compute_workgroup_storage_size;
        limits.max_compute_invocations_per_workgroup = al.max_compute_invocations_per_workgroup;
        limits.max_compute_workgroup_size_x = al.max_compute_workgroup_size_x;
        limits.max_compute_workgroup_size_y = al.max_compute_workgroup_size_y;
        limits.max_compute_workgroups_per_dimension = al.max_compute_workgroups_per_dimension;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("uni3dar-rs-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to request WebGPU device: {e}"))?;

        let mut packer = ParamPacker::new();
        packer.push_f32("bb.norm.weight", bundle.f32_slice("bb.norm.weight")?);
        for l in 0..bb.num_layers {
            let lp = format!("bb.layers.{l}");
            packer.push_f32(&format!("{lp}.attn_norm.weight"), bundle.f32_slice(&format!("{lp}.attn_norm.weight"))?);
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                packer.push_linear(bundle, &format!("{lp}.attn.{proj}"))?;
            }
            packer.push_f32(&format!("{lp}.ffn_norm.weight"), bundle.f32_slice(&format!("{lp}.ffn_norm.weight"))?);
            for fn_name in ["w1", "w2", "w3"] {
                packer.push_linear(bundle, &format!("{lp}.ffn.{fn_name}"))?;
            }
        }
        packer.push_linear(bundle, "bb.tree_head.dense")?;
        packer.push_linear(bundle, "bb.tree_head.out")?;
        for ax in 0..4 {
            packer.push_linear(bundle, &format!("bb.atom_head.pred.{ax}.dense"))?;
            packer.push_linear(bundle, &format!("bb.atom_head.pred.{ax}.out"))?;
        }

        if packer.w_u32.len() > 4 * W_CHUNK_U32 {
            return Err(format!(
                "Packed weight words ({}) exceed 4x126 MiB GPU binding capacity",
                packer.w_u32.len()
            ));
        }

        // Dynamic modulation layout: `(num_layers + 1)` blocks of `MAX_MOD_BRANCHES * 2 * emb_dim` floats
        let mod_dynamic_off = packer.f32_vec.len() as u32;
        let mod_b_stride = (MAX_MOD_BRANCHES * 2 * bb.emb_dim) as u32;
        let total_dyn_floats = (bb.num_layers + 1) * (mod_b_stride as usize);
        packer.f32_vec.resize(packer.f32_vec.len() + total_dyn_floats, 0.0);

        let make_w_chunk = |chunk_idx: usize| -> wgpu::Buffer {
            let s = chunk_idx * W_CHUNK_U32;
            let e = ((chunk_idx + 1) * W_CHUNK_U32).min(packer.w_u32.len());
            let bytes: &[u8] = if s < e {
                bytemuck::cast_slice(&packer.w_u32[s..e])
            } else {
                &[0u8; 16]
            };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("uni3dar-w-blob-{chunk_idx}")),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let w_buf_0 = make_w_chunk(0);
        let w_buf_1 = make_w_chunk(1);
        let w_buf_2 = make_w_chunk(2);
        let w_buf_3 = make_w_chunk(3);
        // Free host u32 staging immediately so browser Worker RAM stays minimal!
        packer.w_u32 = Vec::new();

        let f32_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uni3dar-f32-params"),
            contents: bytemuck::cast_slice(&packer.f32_vec),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        packer.f32_vec = Vec::new();

        let kv_floats = bb.num_layers * 2 * KV_TOTAL_SLOT_TOKENS * bb.emb_dim;
        let kv_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uni3dar-kv-cache"),
            size: (kv_floats * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ws_layout = WsLayout::compute(bb.emb_dim, bb.ffn_dim);
        let ws_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uni3dar-ws"),
            size: (ws_layout.total_floats * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_cap_slots = 512usize;
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uni3dar-uniforms"),
            size: (uniform_cap_slots * (UNIFORM_ALIGN as usize)) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_cap_bytes = 512 * 1024u64;
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uni3dar-readback"),
            size: readback_cap_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("uni3dar-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uni3dar-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<OpParams>() as u64),
                    },
                    count: None,
                },
                st_ro_entry(1),
                st_ro_entry(2),
                st_ro_entry(3),
                st_ro_entry(4),
                st_ro_entry(5),
                st_rw_entry(6),
                st_rw_entry(7),
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("uni3dar-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let mk_pipe = |ep: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let p_gemm = mk_pipe("gemm_fused");
        let p_gemv = mk_pipe("gemv_batched");
        let p_rmsnorm = mk_pipe("rmsnorm_mod");
        let p_rope_kv = mk_pipe("rope_kv_write");
        let p_kv_attn = mk_pipe("batched_kv_attn");
        let p_swiglu = mk_pipe("swiglu_mul");

        if let Some(val_err) = device.pop_error_scope().await {
            return Err(format!("WebGPU shader/pipeline validation failed: {val_err}"));
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uni3dar-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<OpParams>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_buf_0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_buf_1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: w_buf_2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: w_buf_3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: f32_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: kv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: ws_buf.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            adapter_name: info.name,
            backend_api: format!("{:?}", info.backend),
            device,
            queue,
            p_gemm,
            p_gemv,
            p_rmsnorm,
            p_rope_kv,
            p_kv_attn,
            p_swiglu,
            f32_params_buf,
            mod_dynamic_off,
            mod_b_stride,
            linears: packer.linears,
            f32_offs: packer.f32_offs,
            ws_layout,
            kv_buf,
            ws_buf,
            uniform_buf,
            uniform_cap_slots,
            readback_buf,
            readback_cap_bytes,
            bind_group,
            num_layers: bb.num_layers,
            emb_dim: bb.emb_dim,
            num_head: bb.num_head,
            head_dim: bb.head_dim,
            ffn_dim: bb.ffn_dim,
            xyz_res: bb.xyz_resolution,
        })
    }

    /// Upload per-branch AdaLN-Zero `(scale, shift)` modulations ONCE per decoding phase (`0` per-step uploads!).
    pub fn upload_branch_modulations(&self, conds: &[&PrecomputedCond]) -> Result<(), String> {
        let b_run = conds.len();
        if b_run == 0 || b_run > MAX_MOD_BRANCHES {
            return Err(format!("Invalid branch count {b_run} (max {MAX_MOD_BRANCHES})"));
        }
        let d = self.emb_dim;
        let stride = self.mod_b_stride as usize;
        let mut buf = vec![0.0f32; (self.num_layers + 1) * stride];
        for l in 0..self.num_layers {
            let base = l * stride;
            for (bi, c) in conds.iter().enumerate() {
                let dst = base + bi * 2 * d;
                buf[dst..dst + d].copy_from_slice(&c.layer_scale[l]);
                buf[dst + d..dst + 2 * d].copy_from_slice(&c.layer_shift[l]);
            }
        }
        let out_base = self.num_layers * stride;
        for (bi, c) in conds.iter().enumerate() {
            let dst = out_base + bi * 2 * d;
            buf[dst..dst + d].copy_from_slice(&c.out_scale);
            buf[dst + d..dst + 2 * d].copy_from_slice(&c.out_shift);
        }
        self.queue.write_buffer(
            &self.f32_params_buf,
            (self.mod_dynamic_off as u64) * 4,
            bytemuck::cast_slice(&buf),
        );
        Ok(())
    }

    /// Copy winning KV-cache slots from `(old_max_slots, old_max_kv)` layout into `(new_max_slots, new_max_kv)` layout
    /// entirely on GPU (`kv_buf -> ws_buf -> kv_buf`) with zero CPU readback!
    /// `copies` is `&[(src_slot, dst_slot, kv_len)]`.
    pub fn repack_kv_slots(
        &self,
        copies: &[(usize, usize, usize)],
        old_max_slots: usize,
        old_max_kv: usize,
        new_max_slots: usize,
        new_max_kv: usize,
    ) -> Result<(), String> {
        if copies.is_empty() {
            return Ok(());
        }
        if new_max_slots * new_max_kv > KV_TOTAL_SLOT_TOKENS {
            return Err(format!(
                "New KV layout ({new_max_slots} x {new_max_kv}) exceeds KV_TOTAL_SLOT_TOKENS={KV_TOTAL_SLOT_TOKENS}"
            ));
        }
        let d = self.emb_dim;
        let staging_base_bytes = (self.ws_layout.repack_staging as u64) * 4;
        let staging_cap_bytes = 524_288u64 * 4; // 2 MiB per encoder pass

        let old_slot_stride = old_max_kv * d;
        let old_kind_stride = old_max_slots * old_slot_stride;
        let old_layer_stride = 2 * old_kind_stride;

        let new_slot_stride = new_max_kv * d;
        let new_kind_stride = new_max_slots * new_slot_stride;
        let new_layer_stride = 2 * new_kind_stride;

        // Collect all `(src_byte_off, dst_byte_off, byte_len)` segments and execute in <= 2 MiB staging batches.
        // Importantly: we FIRST stage all reads from old layout before overwriting any overlapping region, OR since
        // `2 MiB` holds 1 full layer (`2 * 8 * 128 * 768 * 4 = 6.29 MiB` -> 1 kind/subset per pass), wait: could
        // `new_layer_stride` overlap a DIFFERENT layer's `old_layer_stride`?
        // Look! `old_max_slots * old_max_kv` and `new_max_slots * new_max_kv`: if we set BOTH `old_max_slots * old_max_kv`
        // AND `new_max_slots * new_max_kv` to have the EXACT SAME `kind_stride` (e.g. `16 * 104 == 1664 == 4 * 416 == 8 * 208 == 2 * 832`!),
        // then `old_layer_stride == new_layer_stride` AND `old_kind_stride == new_kind_stride`!!
        // That means Layer `l`, Kind `k` NEVER overlaps any other `(l', k')`!! Each `(layer, kind)` is 100% independent!
        debug_assert_eq!(
            old_max_slots * old_max_kv,
            new_max_slots * new_max_kv,
            "old and new KV layouts must preserve slot*kv product so layers never overlap in-place"
        );

        for l in 0..self.num_layers {
            for kind in 0..2usize {
                let total_copy_bytes: u64 = copies.iter().map(|&(_, _, kl)| (kl * d * 4) as u64).sum();
                if total_copy_bytes == 0 {
                    continue;
                }
                if total_copy_bytes > staging_cap_bytes {
                    return Err("KV repack per-(layer,kind) exceeds 2 MiB staging buffer".to_string());
                }
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("uni3dar-kv-repack"),
                });
                let mut st_cur = staging_base_bytes;
                for &(src_s, _, kl) in copies {
                    let b_len = (kl * d * 4) as u64;
                    if b_len == 0 {
                        continue;
                    }
                    let src_off = ((l * old_layer_stride + kind * old_kind_stride + src_s * old_slot_stride) * 4) as u64;
                    enc.copy_buffer_to_buffer(&self.kv_buf, src_off, &self.ws_buf, st_cur, b_len);
                    st_cur += b_len;
                }
                st_cur = staging_base_bytes;
                for &(_, dst_s, kl) in copies {
                    let b_len = (kl * d * 4) as u64;
                    if b_len == 0 {
                        continue;
                    }
                    let dst_off = ((l * new_layer_stride + kind * new_kind_stride + dst_s * new_slot_stride) * 4) as u64;
                    enc.copy_buffer_to_buffer(&self.ws_buf, st_cur, &self.kv_buf, dst_off, b_len);
                    st_cur += b_len;
                }
                self.queue.submit(Some(enc.finish()));
            }
        }
        Ok(())
    }

    /// Run 1 batched decoding step across `b_run` branches (each contributing `seq_q` query tokens) in ONE GPU `CommandEncoder`!
    /// - `x_tokens`: `(b_run * seq_q * emb_dim)`
    /// - `phy_positions`: `(b_run * seq_q)` physical `[x, y, z]` coordinates in Å
    /// - `branch_slot_and_prev_len`: `(b_run,)` pairs `(slot_id, prev_kv_len)`
    pub async fn forward_step_batch_async(
        &mut self,
        x_tokens: &[f32],
        phy_positions: &[[f32; 3]],
        branch_slot_and_prev_len: &[(u32, u32)],
        seq_q: usize,
        max_slots: usize,
        max_kv_tokens: usize,
        head_mode: StepHeadMode,
    ) -> Result<GpuStepOutput, String> {
        let b_run = branch_slot_and_prev_len.len();
        let total_toks = b_run * seq_q;
        let d = self.emb_dim;
        if total_toks == 0 || total_toks > MAX_STEP_TOKENS {
            return Err(format!("Invalid total_toks={total_toks} (max {MAX_STEP_TOKENS})"));
        }
        if max_slots * max_kv_tokens > KV_TOTAL_SLOT_TOKENS {
            return Err(format!(
                "KV layout ({max_slots} x {max_kv_tokens}) exceeds KV_TOTAL_SLOT_TOKENS={KV_TOTAL_SLOT_TOKENS}"
            ));
        }
        debug_assert_eq!(x_tokens.len(), total_toks * d);
        debug_assert_eq!(phy_positions.len(), total_toks);

        let wl = self.ws_layout;
        // Upload token embeddings, grid positions (`phy_pos / xyz_res`), and `[slot_id, prev_len]` metadata
        self.queue.write_buffer(&self.ws_buf, (wl.h as u64) * 4, bytemuck::cast_slice(x_tokens));

        let inv_res = 1.0 / self.xyz_res;
        let mut rope_grid = Vec::with_capacity(total_toks * 3);
        for p in phy_positions {
            rope_grid.push(p[0] * inv_res);
            rope_grid.push(p[1] * inv_res);
            rope_grid.push(p[2] * inv_res);
        }
        self.queue
            .write_buffer(&self.ws_buf, (wl.rope_pos as u64) * 4, bytemuck::cast_slice(&rope_grid));

        let mut meta_u32 = Vec::with_capacity(b_run * 2);
        let mut max_n_keys_uniform = seq_q as u32;
        for &(slot_id, prev_len) in branch_slot_and_prev_len {
            if (slot_id as usize) >= max_slots || (prev_len as usize) + seq_q > max_kv_tokens {
                return Err(format!(
                    "KV slot/len overflow: slot={slot_id}/{max_slots}, prev_len={prev_len}+{seq_q}/{max_kv_tokens}"
                ));
            }
            max_n_keys_uniform = max_n_keys_uniform.max(prev_len + (seq_q as u32));
            meta_u32.push(slot_id);
            meta_u32.push(prev_len);
        }
        self.queue
            .write_buffer(&self.ws_buf, (wl.branch_meta as u64) * 4, bytemuck::cast_slice(&meta_u32));

        let mut dispatches: Vec<(PipeKind, OpParams, (u32, u32, u32))> = Vec::with_capacity(self.num_layers * 10 + 4);
        let push_gemm = |dispatches: &mut Vec<_>,
                         lin: &GpuLinearHandle,
                         rows: u32,
                         x_off: u32,
                         y_off: u32,
                         act_mode: u32,
                         epi_mode: u32,
                         res_off: u32,
                         adaln_off: u32,
                         seq_q_f: f32| {
            let (kind, wg_x, wg_y) = if rows <= 8 {
                (PipeKind::Gemv, (lin.out_cols + 3) / 4, 1)
            } else {
                (PipeKind::Gemm, (lin.out_cols + 63) / 64, (rows + 63) / 64)
            };
            dispatches.push((
                kind,
                OpParams {
                    p0: rows,
                    p1: lin.out_cols,
                    p2: lin.orig_in_cols,
                    p3: lin.pad_in_cols,
                    p4: lin.mode as u32,
                    p5: act_mode,
                    p6: epi_mode,
                    p7: x_off,
                    p8: y_off,
                    p9: lin.w_u32_off,
                    p10: lin.bias_off,
                    p11: lin.blocks_per_row,
                    f0: seq_q_f,
                    f1: 0.0,
                    u14: res_off,
                    u15: adaln_off,
                },
                (wg_x, wg_y, 1),
            ));
        };

        let tot_u = total_toks as u32;
        let sq_u = seq_q as u32;
        let d_u = d as u32;
        let ffn_u = self.ffn_dim as u32;
        let nh_u = self.num_head as u32;
        let hd_u = self.head_dim as u32;

        for l in 0..self.num_layers {
            let lp = format!("bb.layers.{l}");
            let an_w = self.f32_offs[&format!("{lp}.attn_norm.weight")];
            dispatches.push((
                PipeKind::RmsNormMod,
                OpParams {
                    p0: tot_u,
                    p1: d_u,
                    p2: wl.h,
                    p3: wl.x_norm,
                    p4: an_w,
                    p5: 0,
                    ..Default::default()
                },
                (tot_u, 1, 1),
            ));

            let q_lin = self.linears[&format!("{lp}.attn.q_proj")];
            let k_lin = self.linears[&format!("{lp}.attn.k_proj")];
            let v_lin = self.linears[&format!("{lp}.attn.v_proj")];
            let o_lin = self.linears[&format!("{lp}.attn.out_proj")];
            push_gemm(&mut dispatches, &q_lin, tot_u, wl.x_norm, wl.q, 0, 0, 0, 0, 0.0);
            push_gemm(&mut dispatches, &k_lin, tot_u, wl.x_norm, wl.k, 0, 0, 0, 0, 0.0);
            push_gemm(&mut dispatches, &v_lin, tot_u, wl.x_norm, wl.v, 0, 0, 0, 0, 0.0);

            dispatches.push((
                PipeKind::RopeKvWrite,
                OpParams {
                    p0: tot_u,
                    p1: sq_u,
                    p2: nh_u,
                    p3: hd_u,
                    p4: wl.q,
                    p5: wl.k,
                    p6: wl.v,
                    p7: wl.rope_pos,
                    p8: wl.branch_meta,
                    p9: l as u32,
                    p10: max_slots as u32,
                    p11: max_kv_tokens as u32,
                    ..Default::default()
                },
                (tot_u, 1, 1),
            ));

            dispatches.push((
                PipeKind::BatchedKvAttn,
                OpParams {
                    p0: tot_u,
                    p1: sq_u,
                    p2: nh_u,
                    p3: hd_u,
                    p4: wl.q,
                    p5: wl.attn_cat,
                    p6: wl.branch_meta,
                    p7: l as u32,
                    p8: max_slots as u32,
                    p9: max_kv_tokens as u32,
                    p10: max_n_keys_uniform,
                    ..Default::default()
                },
                (nh_u, tot_u, 1),
            ));

            // h = h + out_proj(attn_cat)
            push_gemm(&mut dispatches, &o_lin, tot_u, wl.attn_cat, wl.h, 0, 1, wl.h, 0, 0.0);

            let fn_w = self.f32_offs[&format!("{lp}.ffn_norm.weight")];
            dispatches.push((
                PipeKind::RmsNormMod,
                OpParams {
                    p0: tot_u,
                    p1: d_u,
                    p2: wl.h,
                    p3: wl.x_norm,
                    p4: fn_w,
                    p5: 0,
                    ..Default::default()
                },
                (tot_u, 1, 1),
            ));

            let w1_lin = self.linears[&format!("{lp}.ffn.w1")];
            let w3_lin = self.linears[&format!("{lp}.ffn.w3")];
            let w2_lin = self.linears[&format!("{lp}.ffn.w2")];
            push_gemm(&mut dispatches, &w1_lin, tot_u, wl.x_norm, wl.gate, 0, 0, 0, 0, 0.0);
            push_gemm(&mut dispatches, &w3_lin, tot_u, wl.x_norm, wl.up, 0, 0, 0, 0, 0.0);

            let n_ffn_elems = tot_u * ffn_u;
            dispatches.push((
                PipeKind::SwigluMul,
                OpParams {
                    p0: n_ffn_elems,
                    p1: wl.gate,
                    p2: wl.up,
                    ..Default::default()
                },
                ((n_ffn_elems + 63) / 64, 1, 1),
            ));

            // h = (h + w2(gate)) * layer_scale[l] + layer_shift[l]
            let l_mod_off = self.mod_dynamic_off + (l as u32) * self.mod_b_stride;
            push_gemm(
                &mut dispatches,
                &w2_lin,
                tot_u,
                wl.gate,
                wl.h,
                0,
                2,
                wl.h,
                l_mod_off,
                sq_u as f32,
            );
        }

        // Extract last token (`seq_q - 1`) of each branch -> RMSNorm + out_mod into `wl.last_out`
        let out_norm_w = self.f32_offs["bb.norm.weight"];
        let out_mod_off = self.mod_dynamic_off + (self.num_layers as u32) * self.mod_b_stride;
        let b_u = b_run as u32;
        dispatches.push((
            PipeKind::RmsNormMod,
            OpParams {
                p0: b_u,
                p1: d_u,
                p2: wl.h,
                p3: wl.last_out,
                p4: out_norm_w,
                p5: 1,
                p6: out_mod_off,
                p7: sq_u,
                ..Default::default()
            },
            (b_u, 1, 1),
        ));

        if head_mode == StepHeadMode::HiddenAndTree {
            let td = self.linears["bb.tree_head.dense"];
            let to = self.linears["bb.tree_head.out"];
            push_gemm(&mut dispatches, &td, b_u, wl.last_out, wl.head_mid, 2, 0, 0, 0, 0.0);
            push_gemm(&mut dispatches, &to, b_u, wl.head_mid, wl.head_logits, 0, 0, 0, 0, 0.0);
        }

        debug_assert!(dispatches.len() <= self.uniform_cap_slots);
        let mut u_bytes = vec![0u8; dispatches.len() * (UNIFORM_ALIGN as usize)];
        for (idx, (_, prm, _)) in dispatches.iter().enumerate() {
            let dst =
                &mut u_bytes[idx * (UNIFORM_ALIGN as usize)..idx * (UNIFORM_ALIGN as usize) + std::mem::size_of::<OpParams>()];
            dst.copy_from_slice(bytemuck::bytes_of(prm));
        }
        self.queue.write_buffer(&self.uniform_buf, 0, &u_bytes);

        let hid_bytes = (b_run * d * 4) as u64;
        let tree_bytes = if head_mode == StepHeadMode::HiddenAndTree {
            (b_run * 255 * 4) as u64
        } else {
            0
        };
        let total_rb = hid_bytes + tree_bytes;
        if total_rb > self.readback_cap_bytes {
            self.readback_cap_bytes = total_rb.next_power_of_two();
            self.readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("uni3dar-readback"),
                size: self.readback_cap_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("uni3dar-step-encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("uni3dar-step-pass"),
                timestamp_writes: None,
            });
            for (idx, (kind, _, wg)) in dispatches.iter().enumerate() {
                let pipe = match kind {
                    PipeKind::Gemm => &self.p_gemm,
                    PipeKind::Gemv => &self.p_gemv,
                    PipeKind::RmsNormMod => &self.p_rmsnorm,
                    PipeKind::RopeKvWrite => &self.p_rope_kv,
                    PipeKind::BatchedKvAttn => &self.p_kv_attn,
                    PipeKind::SwigluMul => &self.p_swiglu,
                };
                cpass.set_pipeline(pipe);
                cpass.set_bind_group(0, &self.bind_group, &[(idx as u32) * UNIFORM_ALIGN]);
                cpass.dispatch_workgroups(wg.0, wg.1, wg.2);
            }
        }
        encoder.copy_buffer_to_buffer(&self.ws_buf, (wl.last_out as u64) * 4, &self.readback_buf, 0, hid_bytes);
        if tree_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &self.ws_buf,
                (wl.head_logits as u64) * 4,
                &self.readback_buf,
                hid_bytes,
                tree_bytes,
            );
        }
        self.queue.submit(Some(encoder.finish()));

        map_buffer_read_async(&self.device, &self.readback_buf, 0..total_rb).await?;
        let mapped = self.readback_buf.slice(0..total_rb).get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&mapped);
        let last_hidden = floats[..b_run * d].to_vec();
        let tree_logits = if tree_bytes > 0 {
            Some(floats[b_run * d..b_run * d + b_run * 255].to_vec())
        } else {
            None
        };
        drop(mapped);
        self.readback_buf.unmap();

        Ok(GpuStepOutput {
            last_hidden,
            tree_logits,
        })
    }
}

fn st_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn st_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

struct MapState {
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    waker: Option<Waker>,
}

struct BufferMapFuture<'a> {
    device: &'a wgpu::Device,
    state: Arc<Mutex<MapState>>,
}

impl<'a> Future for BufferMapFuture<'a> {
    type Output = Result<(), String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = self.device.poll(wgpu::Maintain::Wait);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = self.device.poll(wgpu::Maintain::Poll);
        }
        let mut st = self.state.lock().unwrap();
        if let Some(res) = st.result.take() {
            Poll::Ready(res.map_err(|e| format!("GPU buffer map_async failed: {e}")))
        } else {
            st.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

async fn map_buffer_read_async(
    device: &wgpu::Device,
    buf: &wgpu::Buffer,
    range: std::ops::Range<u64>,
) -> Result<(), String> {
    let state = Arc::new(Mutex::new(MapState {
        result: None,
        waker: None,
    }));
    let st_cb = Arc::clone(&state);
    buf.slice(range).map_async(wgpu::MapMode::Read, move |res| {
        let mut s = st_cb.lock().unwrap();
        s.result = Some(res);
        if let Some(w) = s.waker.take() {
            w.wake();
        }
    });
    BufferMapFuture { device, state }.await
}
