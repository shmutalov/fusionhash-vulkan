//! Isolated GPU-vs-CPU comparison of `single_compute` (the cn/gpu FP core),
//! so a rounding divergence can be bisected without the full pipeline.

use crate::cnhash;
use crate::vk::{as_bytes, Gpu};
use anyhow::Result;
use ash::vk;
use std::sync::Arc;

const SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sctest.comp.spv"));

struct Xorshift(u64);
impl Xorshift {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    fn next_f32_unit(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

pub fn run(gpu: Arc<Gpu>, count: usize) -> Result<()> {
    let device = gpu.device.clone();

    // Generate records.
    let mut rng = Xorshift(0x1234_5678_9abc_def0);
    let mut v = vec![0i32; count * 16];
    let mut rc = vec![0f32; count * 8];
    for i in 0..count {
        for k in 0..16 {
            v[i * 16 + k] = rng.next_u32() as i32;
        }
        // rnd_c: mix of zero and realistic small positive values.
        let scale = match i % 4 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            _ => 4.0,
        };
        for j in 0..4 {
            rc[i * 8 + j] = rng.next_f32_unit() * scale;
        }
        rc[i * 8 + 4] = cnhash_ccnt(i % 16);
    }

    // Buffers.
    let in_v = gpu.create_buffer((v.len() * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, true)?;
    let in_rc = gpu.create_buffer((rc.len() * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, true)?;
    let out_i = gpu.create_buffer((count * 4 * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, true)?;
    let out_f = gpu.create_buffer((count * 4 * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, true)?;
    unsafe {
        in_v.write_bytes(0, bytemuck_slice(&v));
        in_rc.write_bytes(0, bytemuck_slice(&rc));
    }

    // Pipeline.
    let bindings: Vec<_> = (0..4)
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let dsl = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }?;
    let set_layouts = [dsl];
    let pcr = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(4)];
    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&pcr),
            None,
        )
    }?;
    let module = gpu.create_shader_module(SPV)?;
    let entry = std::ffi::CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry);
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
            None,
        )
    }
    .map_err(|(_, e)| e)?[0];

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(4)];
    let desc_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes),
            None,
        )
    }?;
    let ds = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(&set_layouts),
        )
    }?[0];
    for (b, buf) in [&in_v, &in_rc, &out_i, &out_f].iter().enumerate() {
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(buf.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let w = vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(b as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info);
        unsafe { device.update_descriptor_sets(&[w], &[]) };
    }

    let cmd_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(gpu.queue_family),
            None,
        )
    }?;
    let cmd = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }?[0];
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;

    unsafe {
        device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[ds], &[]);
        let c = count as u32;
        device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, as_bytes(&c));
        device.cmd_dispatch(cmd, (count as u32).div_ceil(64), 1, 1);
        device.end_command_buffer(cmd)?;
        let cbs = [cmd];
        device.queue_submit(gpu.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
    }

    // Read back and compare.
    let mut gi = vec![0u8; count * 16];
    let mut gf = vec![0u8; count * 16];
    unsafe {
        out_i.read_bytes(0, &mut gi);
        out_f.read_bytes(0, &mut gf);
    }

    let mut int_mismatch = 0;
    let mut sum_mismatch = 0;
    let mut shown = 0;
    for i in 0..count {
        let vv: [i32; 16] = v[i * 16..i * 16 + 16].try_into().unwrap();
        let rnd_c = [rc[i * 8], rc[i * 8 + 1], rc[i * 8 + 2], rc[i * 8 + 3]];
        let cnt = rc[i * 8 + 4];
        let (cpu_i, cpu_sum) = cnhash::sc_test(&vv, rnd_c, cnt);

        let mut gi4 = [0i32; 4];
        let mut gs4 = [0f32; 4];
        for j in 0..4 {
            gi4[j] = i32::from_le_bytes(gi[i * 16 + j * 4..i * 16 + j * 4 + 4].try_into().unwrap());
            let bits = u32::from_le_bytes(gf[i * 16 + j * 4..i * 16 + j * 4 + 4].try_into().unwrap());
            gs4[j] = f32::from_bits(bits);
        }

        let sum_bad = (0..4).any(|j| cpu_sum[j].to_bits() != gs4[j].to_bits());
        let int_bad = cpu_i != gi4;
        if sum_bad {
            sum_mismatch += 1;
        }
        if int_bad {
            int_mismatch += 1;
        }
        if (sum_bad || int_bad) && shown < 6 {
            shown += 1;
            log::error!("record {i}: cnt={cnt} rnd_c={rnd_c:?}");
            log::error!("  cpu sum bits = {:08x?}", cpu_sum.map(|x| x.to_bits()));
            log::error!("  gpu sum bits = {:08x?}", gs4.map(|x| x.to_bits()));
            log::error!("  cpu int = {cpu_i:08x?}");
            log::error!("  gpu int = {gi4:08x?}");
        }
    }

    // cleanup
    unsafe {
        device.device_wait_idle().ok();
        device.destroy_fence(fence, None);
        device.destroy_command_pool(cmd_pool, None);
        device.destroy_descriptor_pool(desc_pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(dsl, None);
        device.destroy_shader_module(module, None);
    }
    for b in [&in_v, &in_rc, &out_i, &out_f] {
        gpu.destroy_buffer(b);
    }

    log::info!(
        "microtest: {count} records, sum mismatches={sum_mismatch}, int mismatches={int_mismatch}"
    );
    if sum_mismatch == 0 && int_mismatch == 0 {
        log::info!("single_compute matches bit-exactly");
        Ok(())
    } else {
        anyhow::bail!("single_compute diverges")
    }
}

fn cnhash_ccnt(i: usize) -> f32 {
    const C: [f32; 16] = [
        1.34375, 1.28125, 1.359375, 1.3671875, 1.4296875, 1.3984375, 1.3828125, 1.3046875,
        1.4140625, 1.2734375, 1.2578125, 1.2890625, 1.3203125, 1.3515625, 1.3359375, 1.4609375,
    ];
    C[i]
}

fn bytemuck_slice<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
