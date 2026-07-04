//! Per-device miner: owns the four compute pipelines, the per-shard scratchpad
//! / state / output buffers, and drives one full cn/gpu pipeline pass per
//! iteration.

use crate::vk::{as_bytes, Buffer, Gpu};
use anyhow::{Context, Result};
use ash::vk;
use std::sync::Arc;

const SPV_CN0: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cn0.comp.spv"));
const SPV_CN00: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cn00.comp.spv"));
const SPV_CN1: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cn1.comp.spv"));
const SPV_CN2: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cn2.comp.spv"));
const SPV_CN2_DBG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cn2_dbg.comp.spv"));

pub const MEMORY: u64 = 2 * 1024 * 1024;
const STATE_BYTES: u64 = 200; // 25 * u64
const OUTPUT_LEN: u64 = 256; // uints; [255] = count

/// Total cn1 iterations per hash; the per-slice `iter_count`s must sum to this
/// (mirrors `ITERATIONS` in cn1.comp).
const CN1_ITERATIONS: u32 = 49152;
/// Per-hash slice-resume slot in the cn1 save buffer (vec4 vs + uint s, padded
/// to 32 B; mirrors `SaveSlot` in cn1.comp).
const CN1_SAVE_BYTES: u64 = 32;
/// Auto mode: starting slice count, and the per-slice GPU time above which the
/// count is doubled. Windows TDR resets the GPU when a dispatch cannot be
/// preempted for ~2 s, so slices must stay well under that on slow cards.
const CN1_AUTO_SLICES: u32 = 16;
const CN1_MAX_SLICES: u32 = 512;
const CN1_TARGET_SLICE_MS: f64 = 200.0;

#[repr(C)]
#[derive(Clone, Copy)]
struct Cn0Push {
    nonce_base: u64,
    num_threads: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Cn1Push {
    iter_start: u32,
    iter_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Cn2Push {
    target: u64,
}

struct Pipe {
    module: vk::ShaderModule,
    ds_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Pipe {
    fn new(
        gpu: &Gpu,
        spv: &[u8],
        binding_count: u32,
        push_size: u32,
        req_subgroup: Option<u32>,
    ) -> Result<Self> {
        let device = &gpu.device;

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..binding_count)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let ds_layout = unsafe { device.create_descriptor_set_layout(&dsl_ci, None) }?;

        let set_layouts = [ds_layout];
        let mut pl_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pc_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_size)];
        if push_size > 0 {
            pl_ci = pl_ci.push_constant_ranges(&pc_ranges);
        }
        let layout = unsafe { device.create_pipeline_layout(&pl_ci, None) }?;

        let module = gpu.create_shader_module(spv)?;
        let entry = std::ffi::CString::new("main").unwrap();
        // Optionally pin the wavefront size (see `WavePref`). Must outlive the
        // pipeline-create call, so it is declared here regardless.
        let mut req_info = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(req_subgroup.unwrap_or(0));
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry);
        if req_subgroup.is_some() {
            stage = stage.push_next(&mut req_info);
        }
        let cp_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[cp_ci], None)
        }
        .map_err(|(_, e)| e)?[0];

        Ok(Self {
            module,
            ds_layout,
            layout,
            pipeline,
        })
    }

    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.ds_layout, None);
            device.destroy_shader_module(self.module, None);
        }
    }
}

struct Shard {
    scratch: Buffer,
    states: Buffer,
    output: Buffer,
    cn1_save: Buffer,
    ds_cn0: vk::DescriptorSet,
    ds_cn00: vk::DescriptorSet,
    ds_cn1: vk::DescriptorSet,
    ds_cn2: vk::DescriptorSet,
    nonce_base: u64,
}

pub struct Miner {
    gpu: Arc<Gpu>,
    pub tps: u32,
    pub num_shards: u32,
    shards: Vec<Shard>,
    input: Buffer,
    cn0: Pipe,
    cn00: Pipe,
    cn1: Pipe,
    cn2: Pipe,
    desc_pool: vk::DescriptorPool,
    cmd_pool: vk::CommandPool,
    /// One command buffer per queue submission (grown on demand): cn1 slices
    /// are submitted individually so no single kernel-driver job outlives the
    /// GPU watchdog (amdgpu's job timeout covers a whole submission; Windows
    /// TDR preempts at dispatch boundaries — per-slice submits satisfy both).
    cmds: Vec<vk::CommandBuffer>,
    fence: vk::Fence,
    /// cn1 dispatch slices per pass (see `CN1_AUTO_SLICES`).
    cn1_slices: u32,
    /// Auto mode: double `cn1_slices` when a slice runs too long.
    cn1_slices_auto: bool,
    #[allow(dead_code)]
    debug: bool,
    cn2_dbg: Option<Pipe>,
    dbg_bufs: Vec<Buffer>, // one per shard, only in debug mode
}

// Each Miner is driven by exactly one thread; the mapped pointers it holds are
// stable for the lifetime of the mapping.
unsafe impl Send for Miner {}

fn buffer_info(b: &Buffer) -> [vk::DescriptorBufferInfo; 1] {
    [vk::DescriptorBufferInfo::default()
        .buffer(b.buffer)
        .offset(0)
        .range(vk::WHOLE_SIZE)]
}

impl Miner {
    pub fn new(
        gpu: Arc<Gpu>,
        tps: u32,
        num_shards: u32,
        debug: bool,
        wave: Option<u32>,
        cn1_slices: u32,
    ) -> Result<Self> {
        assert!(tps % 64 == 0, "threads-per-shard must be a multiple of 64");
        let device = gpu.device.clone();

        let cn1_slices_auto = cn1_slices == 0;
        let cn1_slices = if cn1_slices_auto {
            CN1_AUTO_SLICES
        } else {
            cn1_slices.clamp(1, CN1_MAX_SLICES)
        };

        // Only the cross-lane cooperative kernels (cn1/cn2) benefit from a pinned
        // wavefront; cn0/cn00 have no in-loop barriers, so leave them at the
        // driver default to avoid constraining their occupancy.
        let cn0 = Pipe::new(&gpu, SPV_CN0, 2, std::mem::size_of::<Cn0Push>() as u32, None)?;
        let cn00 = Pipe::new(&gpu, SPV_CN00, 2, 0, None)?;
        let cn1 = Pipe::new(&gpu, SPV_CN1, 3, std::mem::size_of::<Cn1Push>() as u32, wave)?;
        let cn2 = Pipe::new(&gpu, SPV_CN2, 3, std::mem::size_of::<Cn2Push>() as u32, wave)?;
        let cn2_dbg = if debug {
            Some(Pipe::new(&gpu, SPV_CN2_DBG, 4, std::mem::size_of::<Cn2Push>() as u32, wave)?)
        } else {
            None
        };

        // Descriptor pool sized for all shards.
        let sets_per_shard = if debug { 5 } else { 4 };
        let bufs_per_shard = if debug { 10 + 4 } else { 10 };
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(bufs_per_shard * num_shards)];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(sets_per_shard * num_shards)
            .pool_sizes(&pool_sizes);
        let desc_pool = unsafe { device.create_descriptor_pool(&pool_ci, None) }?;

        let input = gpu.create_buffer(128, vk::BufferUsageFlags::STORAGE_BUFFER, true)?;

        let mut shards = Vec::with_capacity(num_shards as usize);
        let mut dbg_bufs = Vec::new();
        for _ in 0..num_shards {
            // In debug mode make scratch/states host-visible so the self-test
            // can read intermediate stage outputs back.
            let scratch = gpu.create_buffer(
                MEMORY * tps as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                debug,
            )?;
            let states = gpu.create_buffer(
                STATE_BYTES * tps as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                debug,
            )?;
            let output = gpu.create_buffer(
                OUTPUT_LEN * 4,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                true,
            )?;
            let cn1_save = gpu.create_buffer(
                CN1_SAVE_BYTES * tps as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                false,
            )?;

            let alloc = |layout: vk::DescriptorSetLayout| -> Result<vk::DescriptorSet> {
                let layouts = [layout];
                let ai = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool)
                    .set_layouts(&layouts);
                Ok(unsafe { device.allocate_descriptor_sets(&ai) }?[0])
            };
            let ds_cn0 = alloc(cn0.ds_layout)?;
            let ds_cn00 = alloc(cn00.ds_layout)?;
            let ds_cn1 = alloc(cn1.ds_layout)?;
            let ds_cn2 = alloc(cn2.ds_layout)?;

            // Write descriptors.
            let write = |set: vk::DescriptorSet, binding: u32, b: &Buffer| {
                let info = buffer_info(b);
                let w = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&info);
                unsafe { device.update_descriptor_sets(&[w], &[]) };
            };
            write(ds_cn0, 0, &input);
            write(ds_cn0, 1, &states);
            write(ds_cn00, 0, &scratch);
            write(ds_cn00, 1, &states);
            write(ds_cn1, 0, &scratch);
            write(ds_cn1, 1, &states);
            write(ds_cn1, 2, &cn1_save);
            write(ds_cn2, 0, &scratch);
            write(ds_cn2, 1, &states);
            write(ds_cn2, 2, &output);

            if let Some(dbg) = &cn2_dbg {
                let dbuf = gpu.create_buffer(
                    (tps as u64) * 8,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    true,
                )?;
                let ds = alloc(dbg.ds_layout)?;
                write(ds, 0, &scratch);
                write(ds, 1, &states);
                write(ds, 2, &output);
                write(ds, 3, &dbuf);
                // reuse ds_cn2 slot to hold the debug set instead
                dbg_bufs.push(dbuf);
                shards.push(Shard {
                    scratch,
                    states,
                    output,
                    cn1_save,
                    ds_cn0,
                    ds_cn00,
                    ds_cn1,
                    ds_cn2: ds, // debug set
                    nonce_base: 0,
                });
                continue;
            }

            shards.push(Shard {
                scratch,
                states,
                output,
                cn1_save,
                ds_cn0,
                ds_cn00,
                ds_cn1,
                ds_cn2,
                nonce_base: 0,
            });
        }

        let cmd_pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(gpu.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_ci, None) }?;
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;

        Ok(Self {
            gpu,
            tps,
            num_shards,
            shards,
            input,
            cn0,
            cn00,
            cn1,
            cn2,
            desc_pool,
            cmd_pool,
            cmds: Vec::new(),
            fence,
            cn1_slices,
            cn1_slices_auto,
            debug,
            cn2_dbg,
            dbg_bufs,
        })
    }

    /// Total hashes computed per iteration across all shards.
    pub fn hashes_per_iter(&self) -> u64 {
        self.tps as u64 * self.num_shards as u64
    }

    /// Current cn1 slice count (may grow in auto mode).
    pub fn cn1_slices(&self) -> u32 {
        self.cn1_slices
    }

    /// Upload the 128-byte input blob (with the 0x01 pad already applied).
    pub fn set_input(&self, input: &[u8; 128]) {
        unsafe { self.input.write_bytes(0, input) };
    }

    /// Grow the command-buffer pool to at least `n` buffers.
    fn ensure_cmds(&mut self, n: usize) -> Result<()> {
        if self.cmds.len() < n {
            let ai = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count((n - self.cmds.len()) as u32);
            let mut fresh = unsafe { self.gpu.device.allocate_command_buffers(&ai) }?;
            self.cmds.append(&mut fresh);
        }
        Ok(())
    }

    fn barrier(&self, cmd: vk::CommandBuffer) {
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        unsafe {
            self.gpu.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mb],
                &[],
                &[],
            );
        }
    }

    /// Run one full pipeline pass. `nonce_start` is the base nonce
    /// (extraNonce + startNonce); each shard is offset by `tps`. Returns the
    /// GPU-flagged candidate nonces (still need CPU verification).
    pub fn run_iteration(&mut self, nonce_start: u64, target: u64) -> Result<Vec<u64>> {
        self.run_stages(nonce_start, target, 4)
    }

    /// Like `run_iteration` but records only the first `stages` kernels
    /// (1=cn0, 2=+cn00, 3=+cn1, 4=+cn2). Used by the self-test to isolate a
    /// diverging stage.
    pub fn run_stages(&mut self, nonce_start: u64, target: u64, stages: u32) -> Result<Vec<u64>> {
        let device = self.gpu.device.clone();

        // Assign per-shard nonce bases and clear outputs on the host.
        for (i, shard) in self.shards.iter_mut().enumerate() {
            shard.nonce_base = nonce_start + (i as u64) * self.tps as u64;
            unsafe {
                std::ptr::write_bytes(shard.output.mapped, 0, (OUTPUT_LEN * 4) as usize);
            }
        }

        // One command buffer per queue submission: cn0/cn00 ride with the
        // first cn1 slice, cn2 with the last (stage-major within each buffer
        // so shards overlap). Each cn1 slice is submitted as its own job so
        // no single submission outlives the OS GPU watchdog — Windows TDR
        // (~2 s) preempts at dispatch boundaries, but amdgpu's job timeout
        // covers a whole submission, so slicing inside one submit is not
        // enough on Linux.
        let slices = if stages >= 3 {
            self.cn1_slices.clamp(1, CN1_ITERATIONS)
        } else {
            1
        };
        let base_count = CN1_ITERATIONS / slices;
        let extra = CN1_ITERATIONS % slices; // first `extra` slices run one more
        let num_cmds = slices as usize;
        self.ensure_cmds(num_cmds)?;

        unsafe {
            let bi = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            let mut start = 0u32;
            for ci in 0..num_cmds {
                let cmd = self.cmds[ci];
                device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
                device.begin_command_buffer(cmd, &bi)?;

                if ci == 0 {
                    // Stage cn0
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.cn0.pipeline);
                    for shard in &self.shards {
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            self.cn0.layout,
                            0,
                            &[shard.ds_cn0],
                            &[],
                        );
                        let push = Cn0Push {
                            nonce_base: shard.nonce_base,
                            num_threads: self.tps,
                            _pad: 0,
                        };
                        device.cmd_push_constants(
                            cmd,
                            self.cn0.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            as_bytes(&push),
                        );
                        device.cmd_dispatch(cmd, self.tps / 64, 1, 1);
                    }

                    if stages >= 2 {
                        self.barrier(cmd);

                        // Stage cn00
                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            self.cn00.pipeline,
                        );
                        for shard in &self.shards {
                            device.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::COMPUTE,
                                self.cn00.layout,
                                0,
                                &[shard.ds_cn00],
                                &[],
                            );
                            device.cmd_dispatch(cmd, self.tps, 1, 1);
                        }
                    }
                }

                if stages >= 3 {
                    // Make the previous stage/slice's writes visible (pipeline
                    // barriers order across submissions on the same queue).
                    self.barrier(cmd);

                    // Stage cn1, slice `ci`. Slice N+1 reads the scratchpad and
                    // save-state written by slice N; shards overlap in a slice.
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.cn1.pipeline);
                    let count = base_count + u32::from((ci as u32) < extra);
                    let push = Cn1Push {
                        iter_start: start,
                        iter_count: count,
                    };
                    device.cmd_push_constants(
                        cmd,
                        self.cn1.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        as_bytes(&push),
                    );
                    for shard in &self.shards {
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            self.cn1.layout,
                            0,
                            &[shard.ds_cn1],
                            &[],
                        );
                        device.cmd_dispatch(cmd, self.tps / 4, 1, 1);
                    }
                    start += count;
                }

                if stages >= 4 && ci == num_cmds - 1 {
                    self.barrier(cmd);

                    // Stage cn2
                    let (cn2_pipe, cn2_layout) = match &self.cn2_dbg {
                        Some(p) => (p.pipeline, p.layout),
                        None => (self.cn2.pipeline, self.cn2.layout),
                    };
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, cn2_pipe);
                    for shard in &self.shards {
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            cn2_layout,
                            0,
                            &[shard.ds_cn2],
                            &[],
                        );
                        let push = Cn2Push { target };
                        device.cmd_push_constants(
                            cmd,
                            cn2_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            as_bytes(&push),
                        );
                        device.cmd_dispatch(cmd, 1, self.tps / 8, 1);
                    }
                }

                device.end_command_buffer(cmd)?;
            }

            let pass_start = std::time::Instant::now();
            for ci in 0..num_cmds {
                let cbs = [self.cmds[ci]];
                let submit = vk::SubmitInfo::default().command_buffers(&cbs);
                let fence = if ci == num_cmds - 1 {
                    self.fence
                } else {
                    vk::Fence::null()
                };
                device
                    .queue_submit(self.gpu.queue, &[submit], fence)
                    .context("queue_submit failed")?;
            }

            // Wait for the GPU without pegging a CPU core. Some drivers implement
            // an infinite `vkWaitForFences` as a spin loop, which burns a whole
            // core for the multi-second duration of a pass. Poll with a zero
            // timeout and yield the core between checks instead.
            loop {
                match device.wait_for_fences(&[self.fence], true, 0) {
                    Ok(()) => break,
                    Err(vk::Result::TIMEOUT) => {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Err(e) => return Err(e).context("wait_for_fences failed"),
                }
            }
            device.reset_fences(&[self.fence])?;

            // Auto slice sizing: the pass time is an upper bound on the cn1
            // slice time (cn1 dominates a full pass), so keep doubling while a
            // slice could exceed the TDR-safe target. Only full passes are
            // meaningful; partial (self-test) passes skip cn1 entirely or run
            // it whole, and adapting there is harmless either way.
            if self.cn1_slices_auto && stages >= 3 {
                let per_slice_ms =
                    pass_start.elapsed().as_secs_f64() * 1000.0 / self.cn1_slices as f64;
                if per_slice_ms > CN1_TARGET_SLICE_MS && self.cn1_slices < CN1_MAX_SLICES {
                    self.cn1_slices = (self.cn1_slices * 2).min(CN1_MAX_SLICES);
                    log::info!(
                        "cn1 slice ~{per_slice_ms:.0} ms > {CN1_TARGET_SLICE_MS:.0} ms; \
                         raising cn1 slices to {} (GPU watchdog headroom)",
                        self.cn1_slices
                    );
                }
            }
        }

        // Collect flagged candidates.
        let mut out = Vec::new();
        for shard in &self.shards {
            let mut raw = [0u8; (OUTPUT_LEN * 4) as usize];
            unsafe { shard.output.read_bytes(0, &mut raw) };
            let count = u32::from_le_bytes(raw[255 * 4..255 * 4 + 4].try_into().unwrap());
            let count = count.min(255) as usize;
            for i in 0..count {
                let lane = u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
                out.push(shard.nonce_base + lane as u64);
            }
        }
        Ok(out)
    }

    /// Debug-only: read a lane's 25-word state (states buffer is host-visible
    /// in debug mode).
    pub fn debug_state(&self, shard: usize, lane: usize) -> [u64; 25] {
        let mut raw = [0u8; 200];
        unsafe { self.shards[shard].states.read_bytes(lane * 200, &mut raw) };
        let mut st = [0u64; 25];
        for i in 0..25 {
            st[i] = u64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap());
        }
        st
    }

    /// Debug-only: read `words` 32-bit words from a lane's scratchpad.
    pub fn debug_scratch(&self, shard: usize, lane: usize, words: usize) -> Vec<u32> {
        let byte_off = lane * MEMORY as usize;
        let mut raw = vec![0u8; words * 4];
        unsafe { self.shards[shard].scratch.read_bytes(byte_off, &mut raw) };
        (0..words)
            .map(|i| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect()
    }

    /// Debug-only: read back the per-lane PoW values of the last iteration
    /// (requires the miner to have been built with `debug = true`).
    pub fn read_debug_hashes(&self, shard: usize) -> Vec<u64> {
        let n = self.tps as usize;
        let mut raw = vec![0u8; n * 8];
        unsafe { self.dbg_bufs[shard].read_bytes(0, &mut raw) };
        (0..n)
            .map(|i| u64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap()))
            .collect()
    }
}

impl Drop for Miner {
    fn drop(&mut self) {
        let device = &self.gpu.device;
        unsafe {
            let _ = device.device_wait_idle();
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.cmd_pool, None);
            device.destroy_descriptor_pool(self.desc_pool, None);
        }
        self.cn0.destroy(device);
        self.cn00.destroy(device);
        self.cn1.destroy(device);
        self.cn2.destroy(device);
        if let Some(p) = &self.cn2_dbg {
            p.destroy(device);
        }
        for b in &self.dbg_bufs {
            self.gpu.destroy_buffer(b);
        }
        for s in &self.shards {
            self.gpu.destroy_buffer(&s.scratch);
            self.gpu.destroy_buffer(&s.states);
            self.gpu.destroy_buffer(&s.output);
            self.gpu.destroy_buffer(&s.cn1_save);
        }
        self.gpu.destroy_buffer(&self.input);
    }
}
