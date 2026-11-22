//! `wgpu` (`Metal / Vulkan / DX12 / Browser WebGPU`) accelerator and unified `Dpa4Engine`
//! (`Auto / Gpu / Cpu`) for DeePMD `DPA-4` (`SeZM`: `Nano`, `Mini`, `Neo`, `Air`, `Plus`).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use mlip_core::{
    AsyncPotentialBackend, MolecularSystem, NeighborCache, PotentialEval, PropertyPrediction,
};
#[cfg(not(target_arch = "wasm32"))]
use mlip_core::PotentialBackend;
use wgpu::util::DeviceExt;

use crate::cpu::{eval_dpa4_core, Dpa4CpuEngine};
use crate::model::Dpa4Weights;

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

pub struct Dpa4GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: String,
    bgl_mm: wgpu::BindGroupLayout,
    pipe_mm: wgpu::ComputePipeline,
    bgl_to_grid: wgpu::BindGroupLayout,
    pipe_to_grid: wgpu::ComputePipeline,
    bgl_from_grid: wgpu::BindGroupLayout,
    pipe_from_grid: wgpu::ComputePipeline,
    buf_to_grid: wgpu::Buffer,
    buf_from_grid: wgpu::Buffer,
    /// `[block_idx][layer_idx] -> (buf_w_m0_f, buf_w_m1_f)`
    pub so2_bufs: Vec<Vec<(wgpu::Buffer, wgpu::Buffer)>>,
}

impl Dpa4GpuContext {
    pub async fn create(weights: &Dpa4Weights) -> Result<Self, String> {
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
                    label: Some("dpa4-rs-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("wgpu request_device failed: {e}"))?;

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dpa4-rs-shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

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

        let bgl_mm = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_mm"),
            entries: &[uni_entry, ro_entry(1), ro_entry(2), rw_entry(3)],
        });
        let bgl_to_grid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_to_grid"),
            entries: &[
                uni_entry,
                ro_entry(1),
                ro_entry(2),
                ro_entry(3),
                rw_entry(4),
                rw_entry(5),
                rw_entry(6),
            ],
        });
        let bgl_from_grid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_from_grid"),
            entries: &[uni_entry, ro_entry(1), ro_entry(2), rw_entry(3)],
        });

        let mk_pipe = |bgl: &wgpu::BindGroupLayout, entry: &str| {
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

        let pipe_mm = mk_pipe(&bgl_mm, "batched_matmul");
        let pipe_to_grid = mk_pipe(&bgl_to_grid, "so3_to_grid_prod");
        let pipe_from_grid = mk_pipe(&bgl_from_grid, "so3_from_grid");

        if let Some(err) = device.pop_error_scope().await {
            return Err(format!("DPA-4 WGSL pipeline validation error: {err}"));
        }

        let mk_buf = |label: &str, slice: &[f32]| {
            let safe = if slice.is_empty() { &[0.0f32][..] } else { slice };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(safe),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        let buf_to_grid = mk_buf("to_grid_mat", &weights.to_grid_mat);
        let buf_from_grid = mk_buf("from_grid_mat", &weights.from_grid_mat);

        let mut so2_bufs = Vec::with_capacity(weights.blocks.len());
        for (b, blk) in weights.blocks.iter().enumerate() {
            let mut layers = Vec::with_capacity(blk.so2_layers.len());
            for (k, sl) in blk.so2_layers.iter().enumerate() {
                layers.push((
                    mk_buf(&format!("b{b}_l{k}_wm0"), &sl.w_m0_f),
                    mk_buf(&format!("b{b}_l{k}_wm1"), &sl.w_m1_f),
                ));
            }
            so2_bufs.push(layers);
        }

        Ok(Self {
            device,
            queue,
            adapter_info,
            bgl_mm,
            pipe_mm,
            bgl_to_grid,
            pipe_to_grid,
            bgl_from_grid,
            pipe_from_grid,
            buf_to_grid,
            buf_from_grid,
            so2_bufs,
        })
    }

    async fn readback_f32(&self, src: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, String> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let byte_len = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dpa4_staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await
            .map_err(|_| "GPU map_async dropped".to_string())?
            .map_err(|e| format!("GPU map_async error: {e}"))?;

        let data = slice.get_mapped_range();
        let out = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// Execute both `m=0` `(F, E, W0) x (F, W0, W0)` and `|m|=1` `(F, E, 2*W1) x (F, 2*W1, 2*W1)`
    /// `SO2Linear` batched GEMMs (forward or transpose-VJP) in ONE GPU command encoder!
    #[allow(clippy::too_many_arguments)]
    pub async fn run_so2_linear_pair(
        &self,
        b_idx: usize,
        l_idx: usize,
        x_m0: &[f32],
        x_m1: &[f32],
        n_focus: usize,
        n_edges: usize,
        w0: usize,
        two_w1: usize,
        trans_w: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if n_edges == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let (ref bw0, ref bw1) = self.so2_bufs[b_idx][l_idx];
        let len0 = n_focus * n_edges * w0;
        let len1 = n_focus * n_edges * two_w1;

        let buf_x0 = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(x_m0),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_y0 = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (len0 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let packed = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((len0 + len1).max(1) * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.dispatch_mm(
            &mut enc,
            &buf_x0,
            bw0,
            &buf_y0,
            n_focus as u32,
            n_edges as u32,
            w0 as u32,
            w0 as u32,
            trans_w,
        );
        enc.copy_buffer_to_buffer(&buf_y0, 0, &packed, 0, (len0 * 4) as u64);

        if two_w1 > 0 {
            let buf_x1 = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(x_m1),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let buf_y1 = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (len1 * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.dispatch_mm(
                &mut enc,
                &buf_x1,
                bw1,
                &buf_y1,
                n_focus as u32,
                n_edges as u32,
                two_w1 as u32,
                two_w1 as u32,
                trans_w,
            );
            enc.copy_buffer_to_buffer(&buf_y1, 0, &packed, (len0 * 4) as u64, (len1 * 4) as u64);
        }

        self.queue.submit(Some(enc.finish()));
        let flat = self.readback_f32(&packed, len0 + len1).await?;
        Ok((flat[..len0].to_vec(), flat[len0..].to_vec()))
    }

    /// Execute `SO(3)` Lebedev-Haar grid quadratic product (`T_from((T_to L) * (T_to R))`)
    /// for `F=1` `EquivariantFFN` / `output_ffn` on the GPU in ONE command encoder!
    pub async fn run_so3_grid_quad_f1(
        &self,
        left: &[f32],
        right: &[f32],
        n_atoms: usize,
        kd: usize,
        n_grid: usize,
        ch: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let coeff_len = n_atoms * kd * ch;
        let grid_len = n_atoms * n_grid * ch;
        if coeff_len == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let buf_l = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(left),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_r = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(right),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_lg = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (grid_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_rg = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (grid_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_vg = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (grid_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buf_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (coeff_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let total_out = coeff_len + 2 * grid_len;
        let packed = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (total_out * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uni = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&ShaderParams {
                    p0: n_atoms as u32,
                    p1: kd as u32,
                    p2: n_grid as u32,
                    p3: ch as u32,
                    p4: 0,
                    p5: 0,
                    f0: 0.0,
                    f1: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg_to = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_to_grid,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_l.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_to_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_lg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_rg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_vg.as_entire_binding(),
                },
            ],
        });

        let bg_from = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_from_grid,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_vg.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_from_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipe_to_grid);
            pass.set_bind_group(0, &bg_to, &[]);
            pass.dispatch_workgroups(((grid_len + 63) / 64) as u32, 1, 1);

            pass.set_pipeline(&self.pipe_from_grid);
            pass.set_bind_group(0, &bg_from, &[]);
            pass.dispatch_workgroups(((coeff_len + 63) / 64) as u32, 1, 1);
        }

        let b0 = (coeff_len * 4) as u64;
        let bg_bytes = (grid_len * 4) as u64;
        enc.copy_buffer_to_buffer(&buf_out, 0, &packed, 0, b0);
        enc.copy_buffer_to_buffer(&buf_lg, 0, &packed, b0, bg_bytes);
        enc.copy_buffer_to_buffer(&buf_rg, 0, &packed, b0 + bg_bytes, bg_bytes);
        self.queue.submit(Some(enc.finish()));

        let flat = self.readback_f32(&packed, total_out).await?;
        Ok((
            flat[..coeff_len].to_vec(),
            flat[coeff_len..coeff_len + grid_len].to_vec(),
            flat[coeff_len + grid_len..].to_vec(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_mm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
        batch: u32,
        rows: u32,
        k_dim: u32,
        cols: u32,
        trans_b: bool,
    ) {
        let total = batch * rows * cols;
        if total == 0 {
            return;
        }
        let uni = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&ShaderParams {
                    p0: batch,
                    p1: rows,
                    p2: k_dim,
                    p3: cols,
                    p4: u32::from(trans_b),
                    p5: 0,
                    f0: 0.0,
                    f1: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl_mm,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c.as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipe_mm);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((total + 63) / 64, 1, 1);
    }
}

mod oneshot {
    use std::future::Future;
    use std::pin::Pin;
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

    pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
        let s = Arc::new(Mutex::new(State {
            val: None,
            waker: None,
            closed: false,
        }));
        (Sender(Arc::clone(&s)), Receiver(s))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dpa4ExecMode {
    Auto,
    Gpu,
    Cpu,
}

pub struct Dpa4Engine {
    pub weights: Arc<Dpa4Weights>,
    pub cpu: Dpa4CpuEngine,
    pub gpu: Option<Dpa4GpuContext>,
    pub mode: Dpa4ExecMode,
    neighbor_cache: NeighborCache,
    label: String,
}

impl Dpa4Engine {
    pub async fn create(raw_bundle: &[u8], mode: Dpa4ExecMode) -> Result<Self, String> {
        let weights = Arc::new(Dpa4Weights::from_bytes(raw_bundle)?);
        let gpu = if mode != Dpa4ExecMode::Cpu {
            match Dpa4GpuContext::create(&weights).await {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    if mode == Dpa4ExecMode::Gpu {
                        return Err(e);
                    }
                    None
                }
            }
        } else {
            None
        };

        let backend_tag = match (&gpu, mode) {
            (Some(g), Dpa4ExecMode::Gpu) => format!("WebGPU/wgpu: {}", g.adapter_info),
            (Some(g), Dpa4ExecMode::Auto) => format!("Auto (WebGPU {} + SIMD CPU)", g.adapter_info),
            _ => "Pure Rust CPU".to_string(),
        };
        let label = format!("DPA-4 {} [{}]", weights.cfg.tier, backend_tag);
        let rcut = weights.cfg.rcut;
        let cpu = Dpa4CpuEngine::new(Arc::clone(&weights));

        Ok(Self {
            weights,
            cpu,
            gpu,
            mode,
            neighbor_cache: NeighborCache::new(rcut, 0.5),
            label,
        })
    }

    pub fn backend_info(&self) -> &str {
        &self.label
    }

    /// Evaluate optional bundled `_BootstrapAdapters` 3D property oracle on a `(N, C)` descriptor.
    pub fn predict_properties(
        &self,
        descriptor: &[f32],
        coords_a: &[[f64; 3]],
    ) -> Option<Result<PropertyPrediction, String>> {
        self.weights
            .oracle
            .as_ref()
            .map(|o| o.predict(descriptor, coords_a))
    }

    fn should_use_gpu(&self, n_atoms: usize) -> bool {
        match (&self.gpu, self.mode) {
            (None, _) | (_, Dpa4ExecMode::Cpu) => false,
            (Some(_), Dpa4ExecMode::Gpu) => true,
            (Some(_), Dpa4ExecMode::Auto) => {
                (self.weights.cfg.lmax >= 3 && n_atoms >= 28) || n_atoms >= 80
            }
        }
    }
}

impl AsyncPotentialBackend for Dpa4Engine {
    fn model_name(&self) -> &str {
        &self.label
    }

    fn cutoff_a(&self) -> f64 {
        self.weights.cfg.rcut
    }

    async fn eval_async(
        &mut self,
        system: &MolecularSystem,
        need_forces: bool,
        need_descriptor: bool,
    ) -> Result<PotentialEval, String> {
        let gpu_ref = if self.should_use_gpu(system.n_atoms()) {
            self.gpu.as_ref()
        } else {
            None
        };
        eval_dpa4_core(
            &self.weights,
            &mut self.neighbor_cache,
            system,
            need_forces,
            need_descriptor,
            gpu_ref,
        )
        .await
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl PotentialBackend for Dpa4Engine {
    fn model_name(&self) -> &str {
        &self.label
    }

    fn cutoff_a(&self) -> f64 {
        self.weights.cfg.rcut
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
