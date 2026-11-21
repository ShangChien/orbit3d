use crate::config::{ForwardOutput, PrecomputedModulations, SelfCondState};
use crate::cpu_backend::{build_node_and_edge_inputs, remove_com_in_place, GraphTopology};
use crate::format::{ModelBundle, QuantMode};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use wgpu::util::DeviceExt;

const UNIFORM_ALIGN: u32 = 256;

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

#[derive(Clone, Copy, Debug)]
struct GpuGvpHandle {
    wh_off: u32,
    wcp_off: u32,
    wu_off: u32,
    feats_lin: GpuLinearHandle,
    gate_lin: GpuLinearHandle,
}

#[derive(Clone, Copy, Debug)]
struct WorkspaceOffsets {
    n_cap: usize,
    x: u32,
    h: u32,
    v: u32,
    e: u32,
    unit_diff: u32,
    rbf: u32,
    node_s_a: u32,
    node_s_b: u32,
    node_v_a: u32,
    node_v_b: u32,
    edge_s_a: u32,
    edge_s_b: u32,
    edge_v_a: u32,
    edge_v_b: u32,
    s_cat: u32,
    vu_buf: u32,
    gate_buf: u32,
    p_src: u32,
    p_dst: u32,
    out_node_logits: u32,
    out_p_lat: u32,
    out_e_logits: u32,
    total_floats: usize,
}

impl WorkspaceOffsets {
    fn compute(bundle: &ModelBundle, n_cap: usize) -> Self {
        let cfg = &bundle.header.config;
        let n = n_cap.max(4);
        let n_up = n * (n - 1) / 2;
        let e_cnt = (2 * n_up).max(4);
        let s = cfg.n_hidden_scalars;
        let ed = cfg.n_hidden_edge_feats;
        let v = cfg.n_vec_channels;
        let r = cfg.rbf_dim;
        let n_cp = cfg.n_cp_feats;
        let dim_h0 = (1 + v + if cfg.use_dst_feats { v } else { 0 }).max(v);
        let wu0 = dim_h0 + n_cp;
        let node_in_d = cfg.node_in_dim();
        let edge_in_d = cfg.edge_in_dim();
        let a_num = cfg.n_atom_types();
        let c_num = cfg.n_charges;
        let b_num = cfg.n_bond_types;

        let mut cur = 0usize;
        let mut alloc = |len: usize| -> u32 {
            let aligned = (cur + 3) & !3;
            cur = aligned + len.max(4);
            aligned as u32
        };

        let x = alloc(n * 3);
        let h = alloc(n * s);
        let v_off = alloc(n * v * 3);
        let e = alloc(e_cnt * ed);
        let unit_diff = alloc(e_cnt * 3);
        let rbf = alloc(e_cnt * r);

        let node_s_len = n * s.max(node_in_d);
        let node_s_a = alloc(node_s_len);
        let node_s_b = alloc(node_s_len);

        let nv_ch = dim_h0.max(2 * n_cp).max(v).max(1);
        let node_v_a = alloc(n * nv_ch * 3);
        let node_v_b = alloc(n * nv_ch * 3);

        let edge_s_len = e_cnt * s.max(edge_in_d).max(ed);
        let edge_s_a = alloc(edge_s_len);
        let edge_s_b = alloc(edge_s_len);

        let edge_v_a = alloc(e_cnt * v * 3);
        let edge_v_b = alloc(e_cnt * v * 3);

        let max_cat_cols = (s + r + ed + wu0 + s).max(ed + r);
        let s_cat = alloc(e_cnt.max(n) * max_cat_cols);
        let vu_buf = alloc(e_cnt.max(n) * v * 3);
        let gate_buf = alloc(e_cnt.max(n) * v);

        let p_src = alloc(n * s.max(ed));
        let p_dst = alloc(n * ed);

        let out_node_logits = alloc(n * (a_num + c_num));
        let out_p_lat = alloc(n * 34);
        let out_e_logits = alloc(n_up.max(2) * b_num);

        Self {
            n_cap: n,
            x,
            h,
            v: v_off,
            e,
            unit_diff,
            rbf,
            node_s_a,
            node_s_b,
            node_v_a,
            node_v_b,
            edge_s_a,
            edge_s_b,
            edge_v_a,
            edge_v_b,
            s_cat,
            vu_buf,
            gate_buf,
            p_src,
            p_dst,
            out_node_logits,
            out_p_lat,
            out_e_logits,
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

    fn push_f32(&mut self, name: &str, vals: &[f32]) -> u32 {
        let off = self.f32_vec.len() as u32;
        self.f32_vec.extend_from_slice(vals);
        self.f32_offs.insert(name.to_string(), off);
        off
    }

    fn push_linear(&mut self, bundle: &ModelBundle, prefix: &str) -> Result<GpuLinearHandle, String> {
        if let Some(&h) = self.linears.get(prefix) {
            return Ok(h);
        }
        let lv = bundle.linear_view(prefix)?;
        while self.w_u32.len() % 4 != 0 {
            self.w_u32.push(0);
        }
        let w_u32_off = self.w_u32.len() as u32;
        let w_words: &[u32] = bytemuck::cast_slice(lv.weight_bytes);
        self.w_u32.extend_from_slice(w_words);
        let bias_off = if let Some(b) = lv.bias {
            self.push_f32(&format!("{prefix}.bias"), b)
        } else {
            0xFFFF_FFFF
        };
        let handle = GpuLinearHandle {
            mode: lv.mode,
            out_cols: lv.out_features as u32,
            orig_in_cols: lv.orig_cols as u32,
            pad_in_cols: lv.pad_cols as u32,
            blocks_per_row: lv.blocks_per_row as u32,
            w_u32_off,
            bias_off,
        };
        self.linears.insert(prefix.to_string(), handle);
        Ok(handle)
    }

    fn push_ln(&mut self, bundle: &ModelBundle, prefix: &str) -> Result<(u32, u32), String> {
        let w = self.push_f32(&format!("{prefix}.weight"), bundle.f32_slice(&format!("{prefix}.weight"))?);
        let b = self.push_f32(&format!("{prefix}.bias"), bundle.f32_slice(&format!("{prefix}.bias"))?);
        Ok((w, b))
    }

    fn push_std_gvp(&mut self, bundle: &ModelBundle, prefix: &str, has_cp: bool) -> Result<GpuGvpHandle, String> {
        let wh_off = self.push_f32(&format!("{prefix}.Wh"), bundle.f32_slice(&format!("{prefix}.Wh"))?);
        let wcp_off = if has_cp {
            self.push_f32(&format!("{prefix}.Wcp"), bundle.f32_slice(&format!("{prefix}.Wcp"))?)
        } else {
            0
        };
        let wu_off = self.push_f32(&format!("{prefix}.Wu"), bundle.f32_slice(&format!("{prefix}.Wu"))?);
        let feats_lin = self.push_linear(bundle, &format!("{prefix}.to_feats_out"))?;
        let gate_lin = self.push_linear(bundle, &format!("{prefix}.vec_gate"))?;
        Ok(GpuGvpHandle {
            wh_off,
            wcp_off,
            wu_off,
            feats_lin,
            gate_lin,
        })
    }
}

pub struct WgpuBackend {
    pub adapter_name: String,
    pub backend_api: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    p_gemm: wgpu::ComputePipeline,
    p_ln_mod: wgpu::ComputePipeline,
    p_geom: wgpu::ComputePipeline,
    p_gvp_std: wgpu::ComputePipeline,
    p_em0_node: wgpu::ComputePipeline,
    p_em0_edge: wgpu::ComputePipeline,
    p_gate: wgpu::ComputePipeline,
    p_agg_norm: wgpu::ComputePipeline,
    p_sym_edge: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    weights_buf: wgpu::Buffer,
    f32_params_buf: wgpu::Buffer,
    mod_dynamic_f32_off: u32,
    packer: ParamPacker,
    gvps: HashMap<String, GpuGvpHandle>,
    // Cached dynamic workspace & bind group
    ws_layout: WorkspaceOffsets,
    ws_buf: wgpu::Buffer,
    graph_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    uniform_cap_slots: usize,
    readback_buf: wgpu::Buffer,
    readback_cap_bytes: u64,
    bind_group: wgpu::BindGroup,
    cached_topo_n: usize,
}

impl WgpuBackend {
    pub async fn new_async(bundle: &ModelBundle, initial_n_cap: usize) -> Result<Self, String> {
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
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter_limits.max_buffer_size;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("flowmol-rs-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to request WebGPU device: {e}"))?;

        let cfg = &bundle.header.config;
        let has_cp = cfg.n_cp_feats > 0;
        let mut packer = ParamPacker::new();
        let mut gvps = HashMap::new();

        packer.push_linear(bundle, "scalar_embedding.0")?;
        packer.push_linear(bundle, "scalar_embedding.2")?;
        packer.push_ln(bundle, "scalar_embedding.4")?;
        packer.push_linear(bundle, "fragment_cond_proj")?;

        packer.push_linear(bundle, "edge_embedding.0")?;
        packer.push_linear(bundle, "edge_embedding.2")?;
        packer.push_ln(bundle, "edge_embedding.4")?;

        for b_idx in 0..cfg.n_molecule_updates {
            let bp = format!("blocks.{b_idx}");
            for c_idx in 0..cfg.convs_per_update {
                let cp = format!("{bp}.convs.{c_idx}");
                packer.push_f32(&format!("{cp}.em0.Wh_src"), bundle.f32_slice(&format!("{cp}.em0.Wh_src"))?);
                if has_cp {
                    packer.push_f32(&format!("{cp}.em0.Wcp_src"), bundle.f32_slice(&format!("{cp}.em0.Wcp_src"))?);
                }
                // Contiguous pack: Wh_unit, Wcp_unit (if any), Wu
                let u_pack_off = packer.f32_vec.len() as u32;
                packer.f32_vec.extend_from_slice(bundle.f32_slice(&format!("{cp}.em0.Wh_unit"))?);
                if has_cp {
                    packer.f32_vec.extend_from_slice(bundle.f32_slice(&format!("{cp}.em0.Wcp_unit"))?);
                }
                packer.f32_vec.extend_from_slice(bundle.f32_slice(&format!("{cp}.em0.Wu"))?);
                packer.f32_offs.insert(format!("{cp}.em0.unit_pack"), u_pack_off);

                packer.push_linear(bundle, &format!("{cp}.em0.s_src"))?;
                packer.push_linear(bundle, &format!("{cp}.em0.s_edge_sh"))?;
                packer.push_linear(bundle, &format!("{cp}.em0.vec_gate"))?;

                for m_idx in 1..cfg.n_message_gvps {
                    let k = format!("{cp}.em.{m_idx}");
                    let g = packer.push_std_gvp(bundle, &k, has_cp)?;
                    gvps.insert(k, g);
                }
                packer.push_ln(bundle, &format!("{cp}.msg_ln"))?;
                for u_idx in 0..cfg.n_update_gvps {
                    let k = format!("{cp}.nu.{u_idx}");
                    let g = packer.push_std_gvp(bundle, &k, has_cp)?;
                    gvps.insert(k, g);
                }
                packer.push_ln(bundle, &format!("{cp}.upd_ln"))?;
            }
            for p_idx in 0..cfg.n_pos_update_gvps {
                let k = format!("{bp}.pu.{p_idx}");
                let g = packer.push_std_gvp(bundle, &k, has_cp)?;
                gvps.insert(k, g);
            }
            packer.push_linear(bundle, &format!("{bp}.eu.w_src"))?;
            packer.push_linear(bundle, &format!("{bp}.eu.w_dst"))?;
            packer.push_linear(bundle, &format!("{bp}.eu.w_er"))?;
            packer.push_linear(bundle, &format!("{bp}.eu.lin1"))?;
            packer.push_ln(bundle, &format!("{bp}.eu.ln"))?;
        }

        packer.push_linear(bundle, "node_output_head.0")?;
        packer.push_linear(bundle, "node_output_head.2")?;
        if bundle.header.tensors.contains_key("periodic_logit_proj.weight") {
            packer.push_linear(bundle, "periodic_logit_proj")?;
        }
        packer.push_linear(bundle, "to_edge_logits.0")?;
        packer.push_linear(bundle, "to_edge_logits.2")?;

        // Reserve dynamic modulation slots at the tail of `f32_params` (`(L + 1) * 2 * S` floats)
        let mod_dynamic_f32_off = packer.f32_vec.len() as u32;
        let mod_floats = (cfg.n_molecule_updates + 1) * 2 * cfg.n_hidden_scalars;
        packer.f32_vec.resize(packer.f32_vec.len() + mod_floats, 0.0);

        if packer.w_u32.is_empty() {
            packer.w_u32.push(0);
        }
        let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flowmol-weights-blob"),
            contents: bytemuck::cast_slice(&packer.w_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let f32_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flowmol-f32-params"),
            contents: bytemuck::cast_slice(&packer.f32_vec),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flowmol-shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("flowmol-bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flowmol-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let mk_pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let p_gemm = mk_pipe("gemm_fused");
        let p_ln_mod = mk_pipe("layernorm_mod");
        let p_geom = mk_pipe("edge_geom_rbf");
        let p_gvp_std = mk_pipe("gvp_vec_std");
        let p_em0_node = mk_pipe("gvp_em0_node_pre");
        let p_em0_edge = mk_pipe("gvp_em0_edge");
        let p_gate = mk_pipe("gvp_gate_apply");
        let p_agg_norm = mk_pipe("agg_res_gvp_norm");
        let p_sym_edge = mk_pipe("sym_upper_edges");

        let n_cap = initial_n_cap.max(32);
        let ws_layout = WorkspaceOffsets::compute(bundle, n_cap);
        let ws_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flowmol-ws"),
            size: (ws_layout.total_floats * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let max_edges = n_cap * (n_cap - 1);
        let graph_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flowmol-graph-idx"),
            size: ((3 * max_edges).max(16) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_cap_slots = 512usize;
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flowmol-uniforms"),
            size: (uniform_cap_slots * (UNIFORM_ALIGN as usize)) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback_cap_bytes = 256 * 1024u64;
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flowmol-readback"),
            size: readback_cap_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = Self::make_bind_group(&device, &bgl, &uniform_buf, &weights_buf, &f32_params_buf, &graph_buf, &ws_buf);

        Ok(Self {
            adapter_name: info.name,
            backend_api: format!("{:?}", info.backend),
            device,
            queue,
            p_gemm,
            p_ln_mod,
            p_geom,
            p_gvp_std,
            p_em0_node,
            p_em0_edge,
            p_gate,
            p_agg_norm,
            p_sym_edge,
            bgl,
            weights_buf,
            f32_params_buf,
            mod_dynamic_f32_off,
            packer,
            gvps,
            ws_layout,
            ws_buf,
            graph_buf,
            uniform_buf,
            uniform_cap_slots,
            readback_buf,
            readback_cap_bytes,
            bind_group,
            cached_topo_n: 0,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_blocking(bundle: &ModelBundle, initial_n_cap: usize) -> Result<Self, String> {
        pollster::block_on(Self::new_async(bundle, initial_n_cap))
    }

    fn make_bind_group(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
        weights_buf: &wgpu::Buffer,
        f32_params_buf: &wgpu::Buffer,
        graph_buf: &wgpu::Buffer,
        ws_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flowmol-bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: uniform_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<OpParams>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: f32_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: graph_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ws_buf.as_entire_binding(),
                },
            ],
        })
    }

    fn ensure_capacity(&mut self, bundle: &ModelBundle, topo: &GraphTopology) {
        if topo.n_nodes > self.ws_layout.n_cap {
            let new_cap = topo.n_nodes.next_power_of_two().max(topo.n_nodes);
            self.ws_layout = WorkspaceOffsets::compute(bundle, new_cap);
            self.ws_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flowmol-ws"),
                size: (self.ws_layout.total_floats * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let max_e = new_cap * (new_cap - 1);
            self.graph_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flowmol-graph-idx"),
                size: ((3 * max_e).max(16) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.bind_group = Self::make_bind_group(
                &self.device,
                &self.bgl,
                &self.uniform_buf,
                &self.weights_buf,
                &self.f32_params_buf,
                &self.graph_buf,
                &self.ws_buf,
            );
            self.cached_topo_n = 0;
        }
        if self.cached_topo_n != topo.n_nodes {
            let mut g_idx = Vec::with_capacity(3 * topo.n_edges);
            g_idx.extend(topo.src.iter().map(|&x| x as u32));
            g_idx.extend(topo.dst.iter().map(|&x| x as u32));
            g_idx.extend_from_slice(&topo.dst_incoming);
            if !g_idx.is_empty() {
                self.queue.write_buffer(&self.graph_buf, 0, bytemuck::cast_slice(&g_idx));
            }
            self.cached_topo_n = topo.n_nodes;
        }
    }

    pub async fn forward_async(
        &mut self,
        bundle: &ModelBundle,
        topo: &GraphTopology,
        x_t: &[[f32; 3]],
        a_t: &[usize],
        c_t: &[usize],
        e_t_upper: &[usize],
        t_val: f32,
        sc: Option<&SelfCondState>,
        mods: Option<&PrecomputedModulations>,
        node_cond_flags: Option<&[[f32; 2]]>,
    ) -> Result<ForwardOutput, String> {
        self.ensure_capacity(bundle, topo);
        let cfg = &bundle.header.config;
        let wl = self.ws_layout;
        let n = topo.n_nodes as u32;
        let n_edges = topo.n_edges as u32;
        let n_up = topo.n_up as u32;
        let s_dim = cfg.n_hidden_scalars as u32;
        let e_dim = cfg.n_hidden_edge_feats as u32;
        let v_dim = cfg.n_vec_channels as u32;
        let n_cp = cfg.n_cp_feats as u32;
        let rbf_dim = cfg.rbf_dim as u32;
        let a_num = cfg.n_atom_types() as u32;
        let c_num = cfg.n_charges as u32;
        let b_num = cfg.n_bond_types as u32;

        // Build compact input embeddings on CPU (< 0.05 ms) and upload to ws
        let (node_in, edge_in, v_init) =
            build_node_and_edge_inputs(bundle, topo, x_t, a_t, c_t, e_t_upper, t_val, sc)?;
        let x_flat: &[f32] = bytemuck::cast_slice(x_t);
        let v_flat: &[f32] = bytemuck::cast_slice(&v_init);

        self.queue.write_buffer(&self.ws_buf, (wl.x as u64) * 4, bytemuck::cast_slice(x_flat));
        self.queue.write_buffer(&self.ws_buf, (wl.node_s_a as u64) * 4, bytemuck::cast_slice(&node_in));
        self.queue.write_buffer(&self.ws_buf, (wl.v as u64) * 4, bytemuck::cast_slice(v_flat));
        if !edge_in.is_empty() {
            self.queue.write_buffer(&self.ws_buf, (wl.edge_s_a as u64) * 4, bytemuck::cast_slice(&edge_in));
        }

        if let Some(m) = mods {
            let mut flat = Vec::with_capacity((cfg.n_molecule_updates + 1) * 2 * cfg.n_hidden_scalars);
            for l in 0..cfg.n_molecule_updates {
                flat.extend_from_slice(&m.layer_scale[l]);
                flat.extend_from_slice(&m.layer_shift[l]);
            }
            flat.extend_from_slice(&m.out_scale);
            flat.extend_from_slice(&m.out_shift);
            self.queue.write_buffer(
                &self.f32_params_buf,
                (self.mod_dynamic_f32_off as u64) * 4,
                bytemuck::cast_slice(&flat),
            );
        }

        // Record entire 12-layer schedule of uniform parameters + dispatches
        #[derive(Clone, Copy)]
        enum PipeKind {
            Gemm,
            LnMod,
            Geom,
            GvpStd,
            Em0Node,
            Em0Edge,
            Gate,
            AggNorm,
            SymEdge,
        }
        let mut dispatches: Vec<(PipeKind, OpParams, (u32, u32, u32))> = Vec::with_capacity(260);

        let push_gemm = |list: &mut Vec<_>,
                         lin: GpuLinearHandle,
                         rows: u32,
                         x_off: u32,
                         y_off: u32,
                         act: u32,
                         add_mode: u32,
                         add_a: u32,
                         add_b: u32| {
            if rows == 0 {
                return;
            }
            let prm = OpParams {
                p0: rows,
                p1: lin.out_cols,
                p2: lin.orig_in_cols,
                p3: lin.pad_in_cols,
                p4: lin.mode as u32,
                p5: act,
                p6: add_mode,
                p7: x_off,
                p8: y_off,
                p9: lin.w_u32_off,
                p10: lin.bias_off,
                p11: lin.blocks_per_row,
                f0: 0.0,
                f1: 0.0,
                u14: add_a,
                u15: add_b,
            };
            list.push((PipeKind::Gemm, prm, ((lin.out_cols + 63) / 64, (rows + 63) / 64, 1)));
        };

        let push_std_gvp = |list: &mut Vec<_>,
                            g: GpuGvpHandle,
                            rows: u32,
                            v_in: u32,
                            v_out: u32,
                            f_in: u32,
                            s_in: u32,
                            v_in_off: u32,
                            s_out: u32,
                            v_out_off: u32,
                            gate_mode: u32| {
            if rows == 0 {
                return;
            }
            list.push((
                PipeKind::GvpStd,
                OpParams {
                    p0: rows,
                    p1: v_in,
                    p2: v_out,
                    p3: f_in,
                    p4: n_cp,
                    p5: s_in,
                    p6: v_in_off,
                    p7: wl.s_cat,
                    p8: wl.vu_buf,
                    p9: g.wh_off,
                    p10: g.wcp_off,
                    p11: g.wu_off,
                    ..Default::default()
                },
                (rows, 1, 1),
            ));
            push_gemm(list, g.feats_lin, rows, wl.s_cat, s_out, 1, 0, 0, 0);
            push_gemm(list, g.gate_lin, rows, s_out, wl.gate_buf, 0, 0, 0, 0);
            list.push((
                PipeKind::Gate,
                OpParams {
                    p0: rows * v_out,
                    p1: wl.gate_buf,
                    p2: wl.vu_buf,
                    p3: v_out_off,
                    p4: gate_mode,
                    ..Default::default()
                },
                (((rows * v_out) + 63) / 64, 1, 1),
            ));
        };

        // 1. Scalar embedding: node_s_a -> node_s_b -> h -> LayerNorm
        push_gemm(&mut dispatches, self.packer.linears["scalar_embedding.0"], n, wl.node_s_a, wl.node_s_b, 1, 0, 0, 0);
        push_gemm(&mut dispatches, self.packer.linears["scalar_embedding.2"], n, wl.node_s_b, wl.h, 1, 0, 0, 0);
        dispatches.push((
            PipeKind::LnMod,
            OpParams {
                p0: n,
                p1: s_dim,
                p2: wl.h,
                p3: self.packer.f32_offs["scalar_embedding.4.weight"],
                p4: self.packer.f32_offs["scalar_embedding.4.bias"],
                p5: 0,
                ..Default::default()
            },
            (n, 1, 1),
        ));
        if let Some(flags) = node_cond_flags {
            let flat: Vec<f32> = flags.iter().flat_map(|f| [f[0], f[1]]).collect();
            self.queue.write_buffer(&self.ws_buf, (wl.p_src as u64) * 4, bytemuck::cast_slice(&flat));
            push_gemm(&mut dispatches, self.packer.linears["fragment_cond_proj"], n, wl.p_src, wl.h, 0, 3, wl.h, 0);
        }

        // 2. Edge embedding: edge_s_a -> edge_s_b -> e -> LayerNorm
        if n_edges > 0 {
            push_gemm(&mut dispatches, self.packer.linears["edge_embedding.0"], n_edges, wl.edge_s_a, wl.edge_s_b, 1, 0, 0, 0);
            push_gemm(&mut dispatches, self.packer.linears["edge_embedding.2"], n_edges, wl.edge_s_b, wl.e, 1, 0, 0, 0);
            dispatches.push((
                PipeKind::LnMod,
                OpParams {
                    p0: n_edges,
                    p1: e_dim,
                    p2: wl.e,
                    p3: self.packer.f32_offs["edge_embedding.4.weight"],
                    p4: self.packer.f32_offs["edge_embedding.4.bias"],
                    p5: 0,
                    ..Default::default()
                },
                (n_edges, 1, 1),
            ));
            dispatches.push((
                PipeKind::Geom,
                OpParams {
                    p0: n_edges,
                    p1: rbf_dim,
                    p2: wl.x,
                    p3: wl.unit_diff,
                    p4: wl.rbf,
                    p5: 0,
                    f0: cfg.rbf_dmax,
                    ..Default::default()
                },
                ((n_edges + 63) / 64, 1, 1),
            ));
        }

        let dim_h0 = (1 + v_dim).max(v_dim);
        let msg_norm_inv = 1.0 / cfg.message_norm_factor(topo.n_nodes);

        for blk_idx in 0..cfg.n_molecule_updates {
            let bp = format!("blocks.{blk_idx}");
            if mods.is_some() {
                let sc_off = self.mod_dynamic_f32_off + (blk_idx as u32) * 2 * s_dim;
                dispatches.push((
                    PipeKind::LnMod,
                    OpParams {
                        p0: n,
                        p1: s_dim,
                        p2: wl.h,
                        p5: 1,
                        p6: sc_off,
                        p7: sc_off + s_dim,
                        ..Default::default()
                    },
                    (n, 1, 1),
                ));
            }

            for conv_idx in 0..cfg.convs_per_update {
                let cp = format!("{bp}.convs.{conv_idx}");
                // Pre-project node vectors & scalars in O(N)
                dispatches.push((
                    PipeKind::Em0Node,
                    OpParams {
                        p0: n,
                        p1: v_dim,
                        p2: dim_h0,
                        p3: n_cp,
                        p4: wl.v,
                        p5: wl.node_v_a,
                        p6: wl.node_v_b,
                        p7: self.packer.f32_offs[&format!("{cp}.em0.Wh_src")],
                        p8: *self.packer.f32_offs.get(&format!("{cp}.em0.Wcp_src")).unwrap_or(&0),
                        ..Default::default()
                    },
                    (n, 1, 1),
                ));
                push_gemm(
                    &mut dispatches,
                    self.packer.linears[&format!("{cp}.em0.s_src")],
                    n,
                    wl.h,
                    wl.p_src,
                    0,
                    0,
                    0,
                    0,
                );

                // Per-edge vector assembly + fused GEMM (s_edge_sh + p_src[src] -> SiLU) + gate
                if n_edges > 0 {
                    dispatches.push((
                        PipeKind::Em0Edge,
                        OpParams {
                            p0: n_edges,
                            p1: v_dim,
                            p2: dim_h0,
                            p3: n_cp,
                            p4: rbf_dim,
                            p5: e_dim,
                            p6: wl.unit_diff,
                            p7: wl.rbf,
                            p8: wl.e,
                            p9: wl.node_v_a,
                            p10: wl.node_v_b,
                            p11: wl.s_cat,
                            u14: wl.vu_buf,
                            u15: self.packer.f32_offs[&format!("{cp}.em0.unit_pack")],
                            ..Default::default()
                        },
                        (n_edges, 1, 1),
                    ));
                    push_gemm(
                        &mut dispatches,
                        self.packer.linears[&format!("{cp}.em0.s_edge_sh")],
                        n_edges,
                        wl.s_cat,
                        wl.edge_s_a,
                        1, // SiLU
                        1, // + p_src[src]
                        wl.p_src,
                        0,
                    );
                    push_gemm(
                        &mut dispatches,
                        self.packer.linears[&format!("{cp}.em0.vec_gate")],
                        n_edges,
                        wl.edge_s_a,
                        wl.gate_buf,
                        0,
                        0,
                        0,
                        0,
                    );
                    dispatches.push((
                        PipeKind::Gate,
                        OpParams {
                            p0: n_edges * v_dim,
                            p1: wl.gate_buf,
                            p2: wl.vu_buf,
                            p3: wl.edge_v_a,
                            p4: 0,
                            ..Default::default()
                        },
                        (((n_edges * v_dim) + 63) / 64, 1, 1),
                    ));
                }

                let mut cur_es = wl.edge_s_a;
                let mut nxt_es = wl.edge_s_b;
                let mut cur_ev = wl.edge_v_a;
                let mut nxt_ev = wl.edge_v_b;
                for m_idx in 1..cfg.n_message_gvps {
                    let g = self.gvps[&format!("{cp}.em.{m_idx}")];
                    push_std_gvp(&mut dispatches, g, n_edges, v_dim, v_dim, s_dim, cur_es, cur_ev, nxt_es, nxt_ev, 0);
                    std::mem::swap(&mut cur_es, &mut nxt_es);
                    std::mem::swap(&mut cur_ev, &mut nxt_ev);
                }

                // Incoming-edge aggregation + GVPLayerNorm
                dispatches.push((
                    PipeKind::AggNorm,
                    OpParams {
                        p0: n,
                        p1: s_dim,
                        p2: v_dim,
                        p3: n_edges,
                        p4: wl.h,
                        p5: wl.v,
                        p6: cur_es,
                        p7: cur_ev,
                        p8: self.packer.f32_offs[&format!("{cp}.msg_ln.weight")],
                        p9: self.packer.f32_offs[&format!("{cp}.msg_ln.bias")],
                        p10: 0,
                        f0: msg_norm_inv,
                        ..Default::default()
                    },
                    (n, 1, 1),
                ));

                // Copy (h, v) -> (node_s_a, node_v_a) and run node_update GVPs
                dispatches.push((
                    PipeKind::AggNorm,
                    OpParams {
                        p0: n,
                        p1: s_dim,
                        p2: v_dim,
                        p4: wl.h,
                        p5: wl.v,
                        p6: wl.node_s_a,
                        p7: wl.node_v_a,
                        p10: 2,
                        ..Default::default()
                    },
                    (n, 1, 1),
                ));
                let mut cur_ns = wl.node_s_a;
                let mut nxt_ns = wl.node_s_b;
                let mut cur_nv = wl.node_v_a;
                let mut nxt_nv = wl.node_v_b;
                for u_idx in 0..cfg.n_update_gvps {
                    let g = self.gvps[&format!("{cp}.nu.{u_idx}")];
                    push_std_gvp(&mut dispatches, g, n, v_dim, v_dim, s_dim, cur_ns, cur_nv, nxt_ns, nxt_nv, 0);
                    std::mem::swap(&mut cur_ns, &mut nxt_ns);
                    std::mem::swap(&mut cur_nv, &mut nxt_nv);
                }
                dispatches.push((
                    PipeKind::AggNorm,
                    OpParams {
                        p0: n,
                        p1: s_dim,
                        p2: v_dim,
                        p4: wl.h,
                        p5: wl.v,
                        p6: cur_ns,
                        p7: cur_nv,
                        p8: self.packer.f32_offs[&format!("{cp}.upd_ln.weight")],
                        p9: self.packer.f32_offs[&format!("{cp}.upd_ln.bias")],
                        p10: 1,
                        ..Default::default()
                    },
                    (n, 1, 1),
                ));
            }

            // Position updater
            dispatches.push((
                PipeKind::AggNorm,
                OpParams {
                    p0: n,
                    p1: s_dim,
                    p2: v_dim,
                    p4: wl.h,
                    p5: wl.v,
                    p6: wl.node_s_a,
                    p7: wl.node_v_a,
                    p10: 2,
                    ..Default::default()
                },
                (n, 1, 1),
            ));
            let mut cur_ns = wl.node_s_a;
            let mut nxt_ns = wl.node_s_b;
            let mut cur_nv = wl.node_v_a;
            let mut nxt_nv = wl.node_v_b;
            for p_idx in 0..cfg.n_pos_update_gvps {
                let is_last = p_idx + 1 == cfg.n_pos_update_gvps;
                let g = self.gvps[&format!("{bp}.pu.{p_idx}")];
                if is_last {
                    push_std_gvp(&mut dispatches, g, n, v_dim, 1, s_dim, cur_ns, cur_nv, nxt_ns, wl.x, 2);
                } else {
                    push_std_gvp(&mut dispatches, g, n, v_dim, v_dim, s_dim, cur_ns, cur_nv, nxt_ns, nxt_nv, 0);
                    std::mem::swap(&mut cur_ns, &mut nxt_ns);
                    std::mem::swap(&mut cur_nv, &mut nxt_nv);
                }
            }

            // Recompute edge geometry AND pack [e, rbf] into s_cat in ONE pass!
            if n_edges > 0 {
                dispatches.push((
                    PipeKind::Geom,
                    OpParams {
                        p0: n_edges,
                        p1: rbf_dim,
                        p2: wl.x,
                        p3: wl.unit_diff,
                        p4: wl.rbf,
                        p5: 1, // pack [e, rbf] -> s_cat
                        p6: wl.e,
                        p7: wl.s_cat,
                        p8: e_dim,
                        f0: cfg.rbf_dmax,
                        ..Default::default()
                    },
                    ((n_edges + 63) / 64, 1, 1),
                ));

                // EdgeUpdate (Node-Before-Gather pre-split!)
                push_gemm(&mut dispatches, self.packer.linears[&format!("{bp}.eu.w_src")], n, wl.h, wl.p_src, 0, 0, 0, 0);
                push_gemm(&mut dispatches, self.packer.linears[&format!("{bp}.eu.w_dst")], n, wl.h, wl.p_dst, 0, 0, 0, 0);
                push_gemm(
                    &mut dispatches,
                    self.packer.linears[&format!("{bp}.eu.w_er")],
                    n_edges,
                    wl.s_cat,
                    wl.edge_s_a,
                    1, // SiLU
                    2, // + p_src[src] + p_dst[dst]
                    wl.p_src,
                    wl.p_dst,
                );
                push_gemm(
                    &mut dispatches,
                    self.packer.linears[&format!("{bp}.eu.lin1")],
                    n_edges,
                    wl.edge_s_a,
                    wl.e,
                    1, // SiLU
                    4, // e = e + silu(lin1)
                    wl.e,
                    0,
                );
                dispatches.push((
                    PipeKind::LnMod,
                    OpParams {
                        p0: n_edges,
                        p1: e_dim,
                        p2: wl.e,
                        p3: self.packer.f32_offs[&format!("{bp}.eu.ln.weight")],
                        p4: self.packer.f32_offs[&format!("{bp}.eu.ln.bias")],
                        p5: 0,
                        ..Default::default()
                    },
                    (n_edges, 1, 1),
                ));
            }
        }

        if mods.is_some() {
            let out_off = self.mod_dynamic_f32_off + (cfg.n_molecule_updates as u32) * 2 * s_dim;
            dispatches.push((
                PipeKind::LnMod,
                OpParams {
                    p0: n,
                    p1: s_dim,
                    p2: wl.h,
                    p5: 1,
                    p6: out_off,
                    p7: out_off + s_dim,
                    ..Default::default()
                },
                (n, 1, 1),
            ));
        }

        // Output heads
        push_gemm(&mut dispatches, self.packer.linears["node_output_head.0"], n, wl.h, wl.node_s_a, 1, 0, 0, 0);
        push_gemm(&mut dispatches, self.packer.linears["node_output_head.2"], n, wl.node_s_a, wl.out_node_logits, 0, 0, 0, 0);
        let has_plog = self.packer.linears.contains_key("periodic_logit_proj");
        if has_plog {
            push_gemm(&mut dispatches, self.packer.linears["periodic_logit_proj"], n, wl.h, wl.out_p_lat, 0, 0, 0, 0);
        }
        if n_up > 0 {
            dispatches.push((
                PipeKind::SymEdge,
                OpParams {
                    p0: n_up * e_dim,
                    p1: wl.e,
                    p2: wl.edge_s_a,
                    ..Default::default()
                },
                (((n_up * e_dim) + 63) / 64, 1, 1),
            ));
            push_gemm(&mut dispatches, self.packer.linears["to_edge_logits.0"], n_up, wl.edge_s_a, wl.edge_s_b, 1, 0, 0, 0);
            push_gemm(&mut dispatches, self.packer.linears["to_edge_logits.2"], n_up, wl.edge_s_b, wl.out_e_logits, 0, 0, 0, 0);
        }

        // Ensure dynamic uniform buffer & readback staging buffer are big enough
        if dispatches.len() > self.uniform_cap_slots {
            self.uniform_cap_slots = dispatches.len().next_power_of_two();
            self.uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flowmol-uniforms"),
                size: (self.uniform_cap_slots * (UNIFORM_ALIGN as usize)) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.bind_group = Self::make_bind_group(
                &self.device,
                &self.bgl,
                &self.uniform_buf,
                &self.weights_buf,
                &self.f32_params_buf,
                &self.graph_buf,
                &self.ws_buf,
            );
        }

        let mut u_bytes = vec![0u8; dispatches.len() * (UNIFORM_ALIGN as usize)];
        for (idx, (_, prm, _)) in dispatches.iter().enumerate() {
            let dst = &mut u_bytes[idx * (UNIFORM_ALIGN as usize)..idx * (UNIFORM_ALIGN as usize) + std::mem::size_of::<OpParams>()];
            dst.copy_from_slice(bytemuck::bytes_of(prm));
        }
        self.queue.write_buffer(&self.uniform_buf, 0, &u_bytes);

        let x_bytes = (n as u64) * 3 * 4;
        let nl_bytes = (n as u64) * ((a_num + c_num) as u64) * 4;
        let pl_bytes = if has_plog { (n as u64) * 34 * 4 } else { 0 };
        let el_bytes = (n_up as u64) * (b_num as u64) * 4;
        let total_rb = x_bytes + nl_bytes + pl_bytes + el_bytes;
        if total_rb > self.readback_cap_bytes {
            self.readback_cap_bytes = total_rb.next_power_of_two();
            self.readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flowmol-readback"),
                size: self.readback_cap_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        // Record ONE single CommandEncoder for the entire 12-layer network!
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flowmol-step-encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flowmol-forward-pass"),
                timestamp_writes: None,
            });
            for (idx, (kind, _, wg)) in dispatches.iter().enumerate() {
                let pipe = match kind {
                    PipeKind::Gemm => &self.p_gemm,
                    PipeKind::LnMod => &self.p_ln_mod,
                    PipeKind::Geom => &self.p_geom,
                    PipeKind::GvpStd => &self.p_gvp_std,
                    PipeKind::Em0Node => &self.p_em0_node,
                    PipeKind::Em0Edge => &self.p_em0_edge,
                    PipeKind::Gate => &self.p_gate,
                    PipeKind::AggNorm => &self.p_agg_norm,
                    PipeKind::SymEdge => &self.p_sym_edge,
                };
                cpass.set_pipeline(pipe);
                cpass.set_bind_group(0, &self.bind_group, &[(idx as u32) * UNIFORM_ALIGN]);
                cpass.dispatch_workgroups(wg.0, wg.1, wg.2);
            }
        }

        let mut rb_off = 0u64;
        encoder.copy_buffer_to_buffer(&self.ws_buf, (wl.x as u64) * 4, &self.readback_buf, rb_off, x_bytes);
        rb_off += x_bytes;
        encoder.copy_buffer_to_buffer(&self.ws_buf, (wl.out_node_logits as u64) * 4, &self.readback_buf, rb_off, nl_bytes);
        rb_off += nl_bytes;
        if pl_bytes > 0 {
            encoder.copy_buffer_to_buffer(&self.ws_buf, (wl.out_p_lat as u64) * 4, &self.readback_buf, rb_off, pl_bytes);
            rb_off += pl_bytes;
        }
        if el_bytes > 0 {
            encoder.copy_buffer_to_buffer(&self.ws_buf, (wl.out_e_logits as u64) * 4, &self.readback_buf, rb_off, el_bytes);
        }

        self.queue.submit(Some(encoder.finish()));

        // Map staging readback buffer cleanly on both Native and Browser WASM
        map_buffer_read_async(&self.device, &self.readback_buf, 0..total_rb).await?;
        let mapped = self.readback_buf.slice(0..total_rb).get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&mapped);

        let n_u = topo.n_nodes;
        let a_u = a_num as usize;
        let c_u = c_num as usize;
        let b_u = b_num as usize;
        let mut cursor = 0usize;

        let mut x = vec![[0.0f32; 3]; n_u];
        for i in 0..n_u {
            x[i] = [floats[cursor + 3 * i], floats[cursor + 3 * i + 1], floats[cursor + 3 * i + 2]];
        }
        cursor += n_u * 3;

        let node_logits = &floats[cursor..cursor + n_u * (a_u + c_u)];
        cursor += n_u * (a_u + c_u);

        let mut a_logits = vec![0.0f32; n_u * a_u];
        let mut c_logits = vec![0.0f32; n_u * c_u];
        for i in 0..n_u {
            a_logits[i * a_u..(i + 1) * a_u].copy_from_slice(&node_logits[i * (a_u + c_u)..i * (a_u + c_u) + a_u]);
            c_logits[i * c_u..(i + 1) * c_u]
                .copy_from_slice(&node_logits[i * (a_u + c_u) + a_u..(i + 1) * (a_u + c_u)]);
        }

        if has_plog {
            let ptable = bundle.f32_slice("periodic_table_feats")?;
            let p_lat = &floats[cursor..cursor + n_u * 34];
            cursor += n_u * 34;
            for i in 0..n_u {
                let pl = &p_lat[i * 34..(i + 1) * 34];
                for z in 0..a_u {
                    let pt_r = &ptable[z * 34..(z + 1) * 34];
                    let mut dot = 0.0f32;
                    for d in 0..34 {
                        dot += pl[d] * pt_r[d];
                    }
                    a_logits[i * a_u + z] += dot;
                }
            }
        }

        let e_logits = floats[cursor..cursor + topo.n_up * b_u].to_vec();
        drop(mapped);
        self.readback_buf.unmap();

        remove_com_in_place(&mut x);
        Ok(ForwardOutput {
            x,
            a_logits,
            c_logits,
            e_logits,
        })
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

async fn map_buffer_read_async(device: &wgpu::Device, buf: &wgpu::Buffer, range: std::ops::Range<u64>) -> Result<(), String> {
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
