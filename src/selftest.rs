//! GPU-vs-CPU bit-exactness self test. Runs the pipeline stage by stage and
//! reports the first stage that diverges from the CPU reference. Any FP or bit
//! divergence changes the final hash completely, so matching proves exactness.

use crate::autotune::Crdiv;
use crate::cnhash;
use crate::miner::{Miner, MEMORY};
use crate::vk::Gpu;
use anyhow::Result;
use std::sync::Arc;

pub fn run(gpu: Arc<Gpu>, lanes: usize, wave: Option<u32>, crdiv: Crdiv) -> Result<()> {
    let tps = 64u32;
    let name = gpu.pdev.name.clone();
    log::info!(
        "running self-test on {name} (tps={tps}, wave={}, divide={})",
        wave.map_or_else(|| "driver".to_string(), |w| w.to_string()),
        crdiv.name(),
    );

    let mut miner = Miner::new(gpu, tps, 1, true, wave, 0, crdiv)?;

    let mut input = [0u8; 128];
    for i in 0..76 {
        input[i] = ((i * 7 + 3) & 0xff) as u8;
    }
    input[76] = 0x01;
    miner.set_input(&input);

    let nonce_start: u64 = 0x0000_1122_3344_5566;
    let base = nonce_start;
    let lane = 0usize;
    let nonce = base + lane as u64;
    let cpu = cnhash::debug_pipeline(&input, nonce);

    // Stage 1 — cn0 (keccak absorb)
    miner.run_stages(nonce_start, 0, 1)?;
    let gpu_state = miner.debug_state(0, lane);
    if gpu_state != cpu.state {
        for i in 0..25 {
            if gpu_state[i] != cpu.state[i] {
                log::error!(
                    "cn0 diverges at word {i}: cpu=0x{:016x} gpu=0x{:016x}",
                    cpu.state[i],
                    gpu_state[i]
                );
            }
        }
        anyhow::bail!("SELF-TEST FAILED at stage cn0 (Keccak absorb)");
    }
    log::info!("stage cn0  (Keccak absorb)     OK");

    // Stage 2 — cn00 (explode)
    miner.run_stages(nonce_start, 0, 2)?;
    let words = (MEMORY / 4) as usize;
    let gpu_scr = miner.debug_scratch(0, lane, words);
    if let Some(idx) = first_diff(&gpu_scr, &cpu.scratch_after_explode) {
        log::error!(
            "cn00 diverges at word {idx}: cpu=0x{:08x} gpu=0x{:08x}",
            cpu.scratch_after_explode[idx],
            gpu_scr[idx]
        );
        anyhow::bail!("SELF-TEST FAILED at stage cn00 (explode)");
    }
    log::info!("stage cn00 (scratchpad explode) OK");

    // Stage 3 — cn1 (FP core); check a few lanes for early localization
    miner.run_stages(nonce_start, 0, 3)?;
    let ncheck = lanes.min(tps as usize).min(4);
    let mut cn1_bad = false;
    for l in 0..ncheck {
        let cpu_l = cnhash::debug_pipeline(&input, base + l as u64);
        let gpu_scr = miner.debug_scratch(0, l, words);
        if let Some(idx) = first_diff(&gpu_scr, &cpu_l.scratch_after_cn1) {
            log::error!(
                "cn1 lane {l} diverges at word {idx}: cpu=0x{:08x} gpu=0x{:08x}",
                cpu_l.scratch_after_cn1[idx],
                gpu_scr[idx]
            );
            cn1_bad = true;
        }
    }
    if cn1_bad {
        anyhow::bail!("SELF-TEST FAILED at stage cn1 (FP32 core)");
    }
    log::info!("stage cn1  (FP32 memory core)  OK");

    // Stage 4 — cn2 (implode + final keccak), compare several lanes
    miner.run_stages(nonce_start, 0, 4)?;
    let gpu_hashes = miner.read_debug_hashes(0);
    let n = lanes.min(tps as usize);
    let cpu_hashes: Vec<u64> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..n)
            .map(|l| {
                let inp = input;
                s.spawn(move || cnhash::fusion_hash(&inp, base + l as u64))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut mismatches = 0;
    for l in 0..n {
        if cpu_hashes[l] != gpu_hashes[l] {
            mismatches += 1;
            log::error!(
                "cn2 lane {l}: cpu=0x{:016x} gpu=0x{:016x}",
                cpu_hashes[l],
                gpu_hashes[l]
            );
        }
    }
    if mismatches != 0 {
        anyhow::bail!("SELF-TEST FAILED at stage cn2 — {mismatches}/{n} lanes mismatched");
    }
    log::info!("stage cn2  (AES implode+final)  OK");
    log::info!("SELF-TEST PASSED — full pipeline matches bit-exactly on {n} lanes");
    Ok(())
}

fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().min(b.len());
    (0..n).find(|&i| a[i] != b[i])
}
