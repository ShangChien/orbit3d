//! `wgpu` (`Metal / Vulkan / DX12 / Browser WebGPU`) accelerator for `HyFF-L2`
//! (`Presto`, `Vivace`, `Moderato`), executing the heavy `83`-element routed Expert MLPs,
//! Radial Filter MLPs, Scalar Message MLPs, and Bottleneck Linear projections on the GPU
//! while keeping `f64` atomic energy offset reductions lossless on the host!
//!
//! Also exposes `HyffEngine` (`Auto / Gpu / Cpu`) implementing both `PotentialBackend` (native)
//! and `AsyncPotentialBackend` (Browser WebGPU WASM + Native).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use mlip_core::{
    AsyncPotentialBackend, MolecularSystem, NeighborCache, PotentialBackend, PotentialEval,
};
use wgpu::util::DeviceExt;

use crate::cpu::{eval_hyff_core, HyffCpuEngine};
use crate::model::HyffWeights;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShaderParams {
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32,
    p5: u32,
    f0: f32,
    f1: f32,
}

struct GpuBlockBuffers {
    rf0_w: wgpu::Buffer,
    rf0_b: wgpu::Buffer,
    rf2_w: wgpu::Buffer,
    rf2_b: wgpu::Buffer,
    expert_w0: wgpu::Buffer,
    expert_b0: wgpu::Buffer,
    expert_w2: wgpu::Buffer,
    expert_b2: wgpu::Buffer,
}

pub struct HyffGpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: String,
    bgl_lin_fwd: wgpu::BindGroupLayout,
    pipe_lin_fwd: wgpu::ComputePipeline,
    bgl_lin_bwd: wgpu::BindGroupLayout,
    pipe_lin_bwd: wgpu::ComputePipeline,
    bgl_act: wgpu::BindGroupLayout,
    pipe_silu_bwd: wgpu::ComputePipeline,
    bgl_exp_fwd: wgpu::BindGroupLayout,
    pipe_exp_w0_fwd: wgpu::ComputePipeline,
    pipe_exp_w2_fwd: wgpu::ComputePipeline,
    bgl_exp_bwd: wgpu::BindGroupLayout,
    pipe_exp_w2_bwd: wgpu::ComputePipeline,
    pipe_exp_w0_bwd: wgpu::ComputePipeline,
    dummy_f32: wgpu::Buffer,
    blocks: Vec<GpuBlockBuffers>,
}

impl HyffGpuContext {
    pub async fn create(weights: &HyffWeights) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "No WebGPU/Metal/Vulkan adapter available".to_string())?;

        let info = adapter.get_info();
        let adapter_info = format!("{} ({:?})", info.name, info.backend);

        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("hyff-rs-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("wgpu request_device failed: {e}"))?;

        // Validate WGSL shader compilation under strict W3C WebGPU validation scope!
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hyff-rs-shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        let ro_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uni_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl_lin_fwd = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_lin_fwd"),
            entries: &[uni_entry, ro_entry(1), ro_entry(2), ro_entry(3), rw_entry(4)],
        });
        let bgl_lin_bwd = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_lin_bwd"),
            entries: &[uni_entry, ro_entry(1), ro_entry(2), rw_entry(3)],
        });
        let bgl_act = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_act"),
            entries: &[uni_entry, ro_entry(1), rw_entry(2)],
        });
        let bgl_exp_fwd = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_exp_fwd"),
            entries: &[
                uni_entry,
                ro_entry(1),
                ro_entry(2),
                ro_entry(3),
                ro_entry(4),
                ro_entry(5),
                ro_entry(6),
                rw_entry(7),
                rw_entry(8),
            ],
        });
        let bgl_exp_bwd = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_exp_bwd"),
            entries: &[
                uni_entry,
                ro_entry(1),
                ro_entry(2),
                ro_entry(3),
                ro_entry(4),
                ro_entry(5),
                rw_entry(6),
                rw_entry(7),
            ],
        });

        let make_pipe = |bgl: &wgpu::BindGroupLayout, entry: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipe_lin_fwd = make_pipe(&bgl_lin_fwd, "linear_fwd");
        let pipe_lin_bwd = make_pipe(&bgl_lin_bwd, "linear_bwd");
        let pipe_silu_bwd = make_pipe(&bgl_act, "silu_bwd_inplace");
        let pipe_exp_w0_fwd = make_pipe(&bgl_exp_fwd, "expert_w0_fwd");
        let pipe_exp_w2_fwd = make_pipe(&bgl_exp_fwd, "expert_w2_fwd");
        let pipe_exp_w2_bwd = make_pipe(&bgl_exp_bwd, "expert_w2_bwd");
        let pipe_exp_w0_bwd = make_pipe(&bgl_exp_bwd, "expert_w0_bwd");

        if let Some(err) = device.pop_error_scope().await {
            return Err(format!("WGSL pipeline validation error: {err}"));
        }

        let mk_buf = |label: &str, slice: &[f32]| {
            let safe = if slice.is_empty() { &[0.0f32][..] } else { slice };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(safe),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        let dummy_f32 = mk_buf("dummy_f32", &[0.0f32]);
        let mut blocks = Vec::with_capacity(weights.blocks.len());
        for (b, blk) in weights.blocks.iter().enumerate() {
            blocks.push(GpuBlockBuffers {
                rf0_w: mk_buf(&format!("b{b}_rf0_w"), &blk.rf0_w),
                rf0_b: mk_buf(&format!("b{b}_rf0_b"), &blk.rf0_b),
                rf2_w: mk_buf(&format!("b{b}_rf2_w"), &blk.rf2_w),
                rf2_b: mk_buf(&format!("b{b}_rf2_b"), &blk.rf2_b),
                expert_w0: mk_buf(&format!("b{b}_exp_w0"), &blk.expert_w0),
                expert_b0: mk_buf(&format!("b{b}_exp_b0"), &blk.expert_b0),
                expert_w2: mk_buf(&format!("b{b}_exp_w2"), &blk.expert_w2),
                expert_b2: mk_buf(&format!("b{b}_exp_b2"), &blk.expert_b2),
            });
        }

        Ok(Self {
            device,
            queue,
            adapter_info,
            bgl_lin_fwd,
            pipe_lin_fwd,
            bgl_lin_bwd,
            pipe_lin_bwd,
            bgl_act,
            pipe_silu_bwd,
            bgl_exp_fwd,
            pipe_exp_w0_fwd,
            pipe_exp_w2_fwd,
            bgl_exp_bwd,
            pipe_exp_w2_bwd,
            pipe_exp_w0_bwd,
            dummy_f32,
            blocks,
        })
    }

    async fn readback_f32(&self, src: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, String> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let byte_len = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hyff_staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hyff_readback_enc"),
            });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_oneshot();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await
            .map_err(|_| "GPU map_async channel dropped".to_string())?
            .map_err(|e| format!("GPU map_async failed: {e}"))?;

        let data = slice.get_mapped_range();
        let out = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// Batched single-submit GPU execution of all `I` blocks' Radial Filter MLPs (`E x 32 -> E x R_H -> E x M_b*C`)
    /// in ONE command encoder + ONE staging readback!
    pub async fn eval_all_radial_filters_fwd(
        &self,
        weights: &HyffWeights,
        phi_rbf: &[f32],
        n_edges: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
        let n_int = weights.cfg.interactions;
        if n_edges == 0 {
            return Ok((vec![Vec::new(); n_int], vec![Vec::new(); n_int]));
        }
        let k_rbf = weights.cfg.n_rbf;
        let r_h = weights.cfg.radial_hidden;
        let c_dim = weights.cfg.edge_hidden;

        let buf_phi = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("phi_rbf"),
                contents: bytemuck::cast_slice(phi_rbf),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mut total_floats = 0usize;
        let mut pre_lens = Vec::with_capacity(n_int);
        let mut out_lens = Vec::with_capacity(n_int);
        for blk in &weights.blocks {
            let mc = blk.msg_chunks * c_dim;
            pre_lens.push(n_edges * r_h);
            out_lens.push(n_edges * mc);
            total_floats += n_edges * r_h + n_edges * mc;
        }

        let packed_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rf_packed"),
            size: (total_floats.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rf_all_blocks_enc"),
            });

        let mut byte_cursor = 0u64;
        for (b, blk) in weights.blocks.iter().enumerate() {
            let mc = blk.msg_chunks * c_dim;
            let gb = &self.blocks[b];

            let buf_pre = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n_edges * r_h * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let buf_act = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n_edges * r_h * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let buf_wrad = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n_edges * mc * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Save pre-activations
            self.dispatch_lin_fwd(
                &mut enc,
                &buf_phi,
                &gb.rf0_w,
                &gb.rf0_b,
                &buf_pre,
                n_edges as u32,
                k_rbf as u32,
                r_h as u32,
                true,
                false,
            );
            // Activated hidden
            self.dispatch_lin_fwd(
                &mut enc,
                &buf_phi,
                &gb.rf0_w,
                &gb.rf0_b,
                &buf_act,
                n_edges as u32,
                k_rbf as u32,
                r_h as u32,
                true,
                true,
            );
            // Output w_rad
            self.dispatch_lin_fwd(
                &mut enc,
                &buf_act,
                &gb.rf2_w,
                &gb.rf2_b,
                &buf_wrad,
                n_edges as u32,
                r_h as u32,
                mc as u32,
                true,
                false,
            );

            let pre_bytes = (pre_lens[b] * 4) as u64;
            let out_bytes = (out_lens[b] * 4) as u64;
            enc.copy_buffer_to_buffer(&buf_pre, 0, &packed_out, byte_cursor, pre_bytes);
            byte_cursor += pre_bytes;
            enc.copy_buffer_to_buffer(&buf_wrad, 0, &packed_out, byte_cursor, out_bytes);
            byte_cursor += out_bytes;
        }

        self.queue.submit(Some(enc.finish()));
        let flat = self.readback_f32(&packed_out, total_floats).await?;

        let mut rf_pres = Vec::with_capacity(n_int);
        let mut w_rads = Vec::with_capacity(n_int);
        let mut cur = 0usize;
        for b in 0..n_int {
            rf_pres.push(flat[cur..cur + pre_lens[b]].to_vec());
            cur += pre_lens[b];
            w_rads.push(flat[cur..cur + out_lens[b]].to_vec());
            cur += out_lens[b];
        }
        Ok((rf_pres, w_rads))
    }

    /// GPU execution of 83-element routed Expert MLP forward for block `b`.
    pub async fn eval_expert_fwd(
        &self,
        weights: &HyffWeights,
        b: usize,
        experts_u32: &[u32],
        z_in: &[f32],
        n_atoms: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let blk = &weights.blocks[b];
        let gb = &self.blocks[b];
        let d_in = weights.cfg.hidden + 5 * weights.cfg.tensor_hidden;
        let e_h = weights.cfg.expert_hidden;
        let o_b = blk.out_dim;

        let buf_idx = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(experts_u32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_z = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(z_in),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_pre = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n_atoms * e_h * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_u = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n_atoms * o_b * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let packed = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((n_atoms * (e_h + o_b)) * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uni = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&ShaderParams {
                    p0: n_atoms as u32,
                    p1: d_in as u32,
                    p2: e_h as u32,
                    p3: o_b as u32,
                    p4: 0,
                    p5: 0,
                    f0: 0.0,
                    f1: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_exp_fwd,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_idx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gb.expert_w0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gb.expert_b0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gb.expert_w2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: gb.expert_b2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_pre.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buf_u.as_entire_binding(),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipe_exp_w0_fwd);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(((n_atoms * e_h + 63) / 64) as u32, 1, 1);

            pass.set_pipeline(&self.pipe_exp_w2_fwd);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(((n_atoms * o_b + 63) / 64) as u32, 1, 1);
        }
        let pre_bytes = (n_atoms * e_h * 4) as u64;
        let u_bytes = (n_atoms * o_b * 4) as u64;
        enc.copy_buffer_to_buffer(&buf_pre, 0, &packed, 0, pre_bytes);
        enc.copy_buffer_to_buffer(&buf_u, 0, &packed, pre_bytes, u_bytes);
        self.queue.submit(Some(enc.finish()));

        let flat = self.readback_f32(&packed, n_atoms * (e_h + o_b)).await?;
        Ok((
            flat[..n_atoms * e_h].to_vec(),
            flat[n_atoms * e_h..].to_vec(),
        ))
    }

    /// GPU execution of 83-element routed Expert MLP backward + Radial Filter MLP backward in ONE encoder!
    pub async fn eval_expert_and_radial_bwd(
        &self,
        weights: &HyffWeights,
        b: usize,
        experts_u32: &[u32],
        d_u_out: &[f32],
        exp_pre: &[f32],
        d_w_rad: &[f32],
        rf_pre: &[f32],
        n_atoms: usize,
        n_edges: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let blk = &weights.blocks[b];
        let gb = &self.blocks[b];
        let d_in = weights.cfg.hidden + 5 * weights.cfg.tensor_hidden;
        let e_h = weights.cfg.expert_hidden;
        let o_b = blk.out_dim;
        let k_rbf = weights.cfg.n_rbf;
        let r_h = weights.cfg.radial_hidden;
        let mc = blk.msg_chunks * weights.cfg.edge_hidden;

        let buf_idx = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(experts_u32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_du = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(d_u_out),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_epre = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(exp_pre),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_dh = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n_atoms * e_h * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buf_dz = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n_atoms * d_in * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uni_exp = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&ShaderParams {
                    p0: n_atoms as u32,
                    p1: d_in as u32,
                    p2: e_h as u32,
                    p3: o_b as u32,
                    p4: 0,
                    p5: 0,
                    f0: 0.0,
                    f1: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg_exp = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_exp_bwd,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uni_exp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_idx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_du.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_epre.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gb.expert_w0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gb.expert_w2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_dh.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_dz.as_entire_binding(),
                },
            ],
        });

        let packed_len = n_atoms * d_in + n_edges * k_rbf;
        let packed = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (packed_len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipe_exp_w2_bwd);
            pass.set_bind_group(0, &bg_exp, &[]);
            pass.dispatch_workgroups(((n_atoms * e_h + 63) / 64) as u32, 1, 1);

            pass.set_pipeline(&self.pipe_exp_w0_bwd);
            pass.set_bind_group(0, &bg_exp, &[]);
            pass.dispatch_workgroups(((n_atoms * d_in + 63) / 64) as u32, 1, 1);
        }

        let dz_bytes = (n_atoms * d_in * 4) as u64;
        enc.copy_buffer_to_buffer(&buf_dz, 0, &packed, 0, dz_bytes);

        if n_edges > 0 {
            let buf_dwrad = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(d_w_rad),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let buf_rfpre = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(rf_pre),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let buf_drf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n_edges * r_h * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let buf_dphi = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (n_edges * k_rbf * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            self.dispatch_lin_bwd(
                &mut enc,
                &buf_dwrad,
                &gb.rf2_w,
                &buf_drf,
                n_edges as u32,
                r_h as u32,
                mc as u32,
                false,
            );
            self.dispatch_silu_bwd_inplace(&mut enc, &buf_rfpre, &buf_drf, (n_edges * r_h) as u32);
            self.dispatch_lin_bwd(
                &mut enc,
                &buf_drf,
                &gb.rf0_w,
                &buf_dphi,
                n_edges as u32,
                k_rbf as u32,
                r_h as u32,
                false,
            );
            enc.copy_buffer_to_buffer(&buf_dphi, 0, &packed, dz_bytes, (n_edges * k_rbf * 4) as u64);
        }

        self.queue.submit(Some(enc.finish()));
        let flat = self.readback_f32(&packed, packed_len).await?;
        Ok((
            flat[..n_atoms * d_in].to_vec(),
            flat[n_atoms * d_in..].to_vec(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_lin_fwd(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        b: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: u32,
        in_dim: u32,
        out_dim: u32,
        has_bias: bool,
        apply_silu: bool,
    ) {
        let total = rows * out_dim;
        if total == 0 {
            return;
        }
        let uni = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&ShaderParams {
                    p0: rows,
                    p1: in_dim,
                    p2: out_dim,
                    p3: u32::from(has_bias),
                    p4: u32::from(apply_silu),
                    p5: 0,
                    f0: 0.0,
                    f1: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_lin_fwd,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: if has_bias {
                        b.as_entire_binding()
                    } else {
                        self.dummy_f32.as_entire_binding()
                    },
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: y.as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipe_lin_fwd);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((total + 63) / 64, 1, 1);
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_lin_bwd(
        &self,
        enc: &mut wgpu::CommandEncoder,
        dy: &wgpu::Buffer,
        w: &wgpu::Buffer,
        dx: &wgpu::Buffer,
        rows: u32,
        in_dim: u32,
        out_dim: u32,
        accumulate: bool,
    ) {
        let total = rows * in_dim;
        if total == 0 {
            return;
        }
        let uni = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&ShaderParams {
                    p0: rows,
                    p1: in_dim,
                    p2: out_dim,
                    p3: u32::from(accumulate),
                    p4: 0,
                    p5: 0,
                    f0: 0.0,
                    f1: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_lin_bwd,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dx.as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipe_lin_bwd);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((total + 63) / 64, 1, 1);
    }

    fn dispatch_silu_bwd_inplace(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pre: &wgpu::Buffer,
        dy_inout: &wgpu::Buffer,
        len: u32,
    ) {
        if len == 0 {
            return;
        }
        let uni = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&ShaderParams {
                    p0: len,
                    p1: 0,
                    p2: 0,
                    p3: 0,
                    p4: 0,
                    p5: 0,
                    f0: 0.0,
                    f1: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_act,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pre.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dy_inout.as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipe_silu_bwd);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((len + 63) / 64, 1, 1);
    }
}

/// Minimal dependency-free oneshot channel for `wgpu::BufferSlice::map_async` on both Native and WASM.
mod oneshot_util {
    use std::cell::RefCell;
    use std::future::Future;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};

    struct State<T> {
        val: Option<T>,
        waker: Option<Waker>,
        closed: bool,
    }

    pub struct Sender<T>(Arc<Mutex<State<T>>>);
    pub struct Receiver<T>(Arc<Mutex<State<T>>>);

    impl<T> Sender<T> {
        pub fn send(self, v: T) -> Result<(), ()> {
            let mut st = self.0.lock().unwrap();
            st.val = Some(v);
            if let Some(w) = st.waker.take() {
                w.wake();
            }
            Ok(())
        }
    }

    impl<T> Drop for Sender<T> {
        fn drop(&mut self) {
            if let Ok(mut st) = self.0.lock() {
                st.closed = true;
                if let Some(w) = st.waker.take() {
                    w.wake();
                }
            }
        }
    }

    impl<T> Future for Receiver<T> {
        type Output = Result<T, ()>;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            let mut st = self.0.lock().unwrap();
            if let Some(v) = st.val.take() {
                Poll::Ready(Ok(v))
            } else if st.closed {
                Poll::Ready(Err(()))
            } else {
                st.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }

    #[allow(dead_code)]
    type _Unused = Rc<RefCell<()>>;

    pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
        let s = Arc::new(Mutex::new(State {
            val: None,
            waker: None,
            closed: false,
        }));
        (Sender(Arc::clone(&s)), Receiver(s))
    }
}

fn futures_oneshot<T>() -> (oneshot_util::Sender<T>, oneshot_util::Receiver<T>) {
    oneshot_util::channel()
}

/// Unified `HyFF-L2` Engine supporting `Auto` (smart GPU dispatch on large molecules / Moderato +
/// zero-latency CPU path on small molecules), explicit `Gpu` (`wgpu` Metal/WebGPU), and `Cpu`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HyffExecMode {
    Auto,
    Gpu,
    Cpu,
}

pub struct HyffEngine {
    pub weights: Arc<HyffWeights>,
    pub cpu: HyffCpuEngine,
    pub gpu: Option<HyffGpuContext>,
    pub mode: HyffExecMode,
    neighbor_cache: NeighborCache,
    label: String,
}

impl HyffEngine {
    pub async fn create(raw_bundle: &[u8], mode: HyffExecMode) -> Result<Self, String> {
        let weights = Arc::new(HyffWeights::from_bytes(raw_bundle)?);
        let gpu = if mode != HyffExecMode::Cpu {
            match HyffGpuContext::create(&weights).await {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    if mode == HyffExecMode::Gpu {
                        return Err(e);
                    }
                    None
                }
            }
        } else {
            None
        };

        let backend_tag = match (&gpu, mode) {
            (Some(g), HyffExecMode::Gpu) => format!("WebGPU/wgpu: {}", g.adapter_info),
            (Some(g), HyffExecMode::Auto) => format!("Auto (WebGPU {} + SIMD CPU)", g.adapter_info),
            _ => "Pure Rust CPU".to_string(),
        };
        let label = format!("HyFF-L2 {} [{}]", weights.cfg.tier, backend_tag);
        let cutoff = weights.cfg.cutoff;
        let cpu = HyffCpuEngine::new(Arc::clone(&weights));

        Ok(Self {
            weights,
            cpu,
            gpu,
            mode,
            neighbor_cache: NeighborCache::new(cutoff, 0.5),
            label,
        })
    }

    pub fn backend_info(&self) -> &str {
        &self.label
    }

    fn should_use_gpu(&self, n_atoms: usize) -> bool {
        match (&self.gpu, self.mode) {
            (None, _) | (_, HyffExecMode::Cpu) => false,
            (Some(_), HyffExecMode::Gpu) => true,
            // In Auto mode: use wgpu when molecule + model width amortizes map_async readback (~1.5 ms)
            (Some(_), HyffExecMode::Auto) => {
                (self.weights.cfg.hidden >= 128 && n_atoms >= 32) || n_atoms >= 96
            }
        }
    }

    /// Full `wgpu`-accelerated forward + analytical reverse-mode VJP evaluation.
    async fn eval_with_gpu(
        &mut self,
        system: &MolecularSystem,
        need_forces: bool,
        need_descriptor: bool,
    ) -> Result<PotentialEval, String> {
        let gpu = self.gpu.as_ref();
        eval_hyff_core(
            &self.weights,
            &mut self.neighbor_cache,
            system,
            need_forces,
            need_descriptor,
            gpu,
        )
        .await
    }
}

impl AsyncPotentialBackend for HyffEngine {
    fn model_name(&self) -> &str {
        &self.label
    }

    fn cutoff_a(&self) -> f64 {
        self.weights.cfg.cutoff
    }

    async fn eval_async(
        &mut self,
        system: &MolecularSystem,
        need_forces: bool,
        need_descriptor: bool,
    ) -> Result<PotentialEval, String> {
        if self.should_use_gpu(system.n_atoms()) {
            self.eval_with_gpu(system, need_forces, need_descriptor)
                .await
        } else {
            self.cpu.eval(system, need_forces, need_descriptor)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl PotentialBackend for HyffEngine {
    fn model_name(&self) -> &str {
        &self.label
    }

    fn cutoff_a(&self) -> f64 {
        self.weights.cfg.cutoff
    }

    fn eval(
        &mut self,
        system: &MolecularSystem,
        need_forces: bool,
        need_descriptor: bool,
    ) -> Result<PotentialEval, String> {
        pollster::block_on(self.eval_async(system, need_forces, need_descriptor))
    }
}
