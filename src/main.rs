mod autotune;
mod cnhash;
mod config;
mod microtest;
mod miner;
mod selftest;
mod stratum;
mod vk;

use crate::config::Config;
use crate::miner::{Miner, MEMORY};
use crate::stratum::{MockPool, Pool, StratumPool};
use crate::vk::{Gpu, Instance, PhysicalDevice, WavePref};
use anyhow::{bail, Result};
use clap::Parser;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    let cfg = Config::parse();
    let instance = Instance::new()?;
    let mut devices = instance.enumerate()?;

    if !cfg.all {
        devices.retain(|d| d.is_gpu_vendor());
    }

    if cfg.info {
        println!("Available Vulkan compute devices:");
        for (i, d) in devices.iter().enumerate() {
            println!(
                "[{}] {} (vendor=0x{:04x} device=0x{:04x} VRAM={:.1} GiB CUs={} subgroup={} range={}..={} ctrl={} driver={})",
                i + 1,
                d.name,
                d.vendor_id,
                d.device_id,
                d.device_local_mem as f64 / (1u64 << 30) as f64,
                d.compute_units,
                d.subgroup_size,
                d.min_subgroup_size,
                d.max_subgroup_size,
                d.subgroup_size_control,
                d.driver_info,
            );
        }
        return Ok(());
    }

    // Device selection (1-based).
    let sel = cfg.selected_indices();
    let chosen: Vec<PhysicalDevice> = if sel.is_empty() {
        devices.clone()
    } else {
        sel.iter()
            .filter_map(|&i| devices.get(i - 1).cloned())
            .collect()
    };
    if chosen.is_empty() {
        bail!("no matching Vulkan devices (use --all or --info to inspect)");
    }

    if cfg.microtest {
        let gpu = Gpu::new(instance.clone(), chosen[0].clone())?;
        return microtest::run(gpu, 1_000_000);
    }

    if cfg.selftest {
        let gpu = Gpu::new(instance.clone(), chosen[0].clone())?;
        let wave = resolve_wave(&gpu, &cfg.wave);
        let crdiv = autotune::select_crdiv(&gpu, &cfg.crdiv)?;
        return selftest::run(gpu, 64, wave, crdiv);
    }

    // Pool.
    let pool: Arc<dyn Pool> = if cfg.mock {
        Arc::new(MockPool::new())
    } else {
        Arc::new(StratumPool::connect(&cfg.pool, cfg.user.clone(), cfg.pass.clone())?)
    };
    log::info!("pool: {}", pool.url());

    // Spin up a miner thread per device.
    let mut reporters: Vec<Reporter> = Vec::new();
    for (idx, pd) in chosen.iter().enumerate() {
        let (tps, num_shards) = autotune::select_layout(&cfg, pd);
        if MEMORY * tps as u64 > pd.max_alloc {
            bail!(
                "tps={} needs {} MiB/shard which exceeds the device max allocation of {} MiB; lower --tps",
                tps,
                MEMORY * tps as u64 / (1024 * 1024),
                pd.max_alloc / (1024 * 1024)
            );
        }
        let gpu = Gpu::new(instance.clone(), pd.clone())?;
        let wave = resolve_wave(&gpu, &cfg.wave);
        let crdiv = autotune::select_crdiv(&gpu, &cfg.crdiv)?;
        let miner = Miner::new(gpu, tps, num_shards, false, wave, cfg.cn1_slices, crdiv)?;
        let total = miner.hashes_per_iter();
        log::info!(
            "device [{}] {}: tps={}{} shards={} threads={} scratch={:.2} GiB wave={} divide={} cn1_slices={}{}",
            idx + 1,
            pd.name,
            tps,
            if cfg.tps == 0 { " (auto)" } else { "" },
            num_shards,
            total,
            (MEMORY * total) as f64 / (1u64 << 30) as f64,
            wave.map_or_else(|| "driver".to_string(), |w| w.to_string()),
            crdiv.name(),
            miner.cn1_slices(),
            if cfg.cn1_slices == 0 { " (auto)" } else { "" },
        );

        let hr = Arc::new(AtomicU64::new(0));
        reporters.push(Reporter {
            name: pd.name.clone(),
            bus: pd.pci_bus,
            hashrate: hr.clone(),
        });
        let pool = pool.clone();
        let dev = idx + 1;
        std::thread::spawn(move || {
            if let Err(e) = mine_loop(miner, pool, hr, dev) {
                log::error!("device [{dev}] miner stopped: {e:#}");
            }
        });
    }

    // Hashrate reporting + optional stats file.
    let start = Instant::now();
    let stats_file = cfg.stats_file.clone();
    loop {
        std::thread::sleep(Duration::from_secs(10));
        let mut total = 0u64;
        for r in &reporters {
            let v = r.hashrate.load(Ordering::Relaxed);
            total += v;
            log::info!("{}: {:.3} kH/s", r.name, v as f64 / 1000.0);
        }
        if reporters.len() > 1 {
            log::info!("total: {:.3} kH/s", total as f64 / 1000.0);
        }
        if !stats_file.is_empty() {
            let (acc, rej) = pool.shares();
            write_stats(&stats_file, &reporters, start.elapsed().as_secs(), acc, rej);
        }
    }
}

struct Reporter {
    name: String,
    bus: u32,
    hashrate: Arc<AtomicU64>,
}

fn write_stats(path: &str, reporters: &[Reporter], uptime: u64, accepted: u64, rejected: u64) {
    // kH/s throughout (hs_units = khs on the HiveOS side).
    let total_khs: f64 = reporters
        .iter()
        .map(|r| r.hashrate.load(Ordering::Relaxed) as f64 / 1000.0)
        .sum();
    let gpus: Vec<String> = reporters
        .iter()
        .map(|r| {
            format!(
                "{{\"bus\":{},\"name\":{},\"khs\":{:.3}}}",
                r.bus,
                serde_json::to_string(&r.name).unwrap_or_else(|_| "\"\"".into()),
                r.hashrate.load(Ordering::Relaxed) as f64 / 1000.0
            )
        })
        .collect();
    let json = format!(
        "{{\"algo\":\"cn/gpu\",\"uptime\":{uptime},\"khs\":{total_khs:.3},\"accepted\":{accepted},\"rejected\":{rejected},\"gpus\":[{}]}}",
        gpus.join(",")
    );
    // Write atomically-ish: temp file then rename.
    let tmp = format!("{path}.tmp");
    if std::fs::write(&tmp, json).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// Turn the `--wave` string into a concrete required subgroup size for cn1/cn2,
/// warning if a forced size cannot be honoured by the device.
fn resolve_wave(gpu: &Gpu, s: &str) -> Option<u32> {
    let pref = match s.trim() {
        "" | "auto" => WavePref::Auto,
        // Explicitly leave the driver to choose (no required size pinned).
        "driver" | "off" | "none" => return None,
        "32" => WavePref::Force(32),
        "64" => WavePref::Force(64),
        other => {
            log::warn!("invalid --wave '{other}' (expected auto|driver|32|64), using auto");
            WavePref::Auto
        }
    };
    let resolved = gpu.required_subgroup_size(pref);
    if let (WavePref::Force(n), None) = (pref, resolved) {
        log::warn!(
            "--wave {n} not supported on {} (subgroup range {}..={}, control={}); using driver default",
            gpu.pdev.name,
            gpu.pdev.min_subgroup_size,
            gpu.pdev.max_subgroup_size,
            gpu.pdev.subgroup_size_control,
        );
    }
    resolved
}


fn mine_loop(mut miner: Miner, pool: Arc<dyn Pool>, hashrate: Arc<AtomicU64>, dev: usize) -> Result<()> {
    let per_iter = miner.hashes_per_iter();
    let mut last = Instant::now();
    let mut hashes_since = 0u64;

    loop {
        let job = match pool.current_job() {
            Some(j) => j,
            None => {
                std::thread::sleep(Duration::from_millis(500));
                continue;
            }
        };
        let input = job.input_128();
        miner.set_input(&input);

        while pool
            .current_job()
            .map(|j| j.job_id == job.job_id)
            .unwrap_or(false)
        {
            if job.received.elapsed() > Duration::from_secs(300) {
                std::thread::sleep(Duration::from_secs(5));
                continue;
            }

            let start = job.reserve(per_iter);
            let nonce_base = job.extra_nonce.wrapping_add(start);
            let candidates = miner.run_iteration(nonce_base, job.target)?;

            hashes_since += per_iter;
            let dt = last.elapsed();
            if dt >= Duration::from_secs(1) {
                let hr = hashes_since as f64 / dt.as_secs_f64();
                // simple EMA smoothing
                let prev = hashrate.load(Ordering::Relaxed) as f64;
                let smoothed = if prev == 0.0 { hr } else { prev * 0.7 + hr * 0.3 };
                hashrate.store(smoothed as u64, Ordering::Relaxed);
                last = Instant::now();
                hashes_since = 0;
            }

            // Verify + submit candidates on a background thread. The CPU
            // re-hash (`fusion_hash`) takes hundreds of ms; doing it inline
            // stalled the GPU until it finished, which is what made the pool
            // hashrate spike down whenever shares were found. Hand the work off
            // and immediately launch the next GPU pass. Mock has target 0, so it
            // never produces candidates and never spawns a thread.
            if !pool.is_mock() && !candidates.is_empty() {
                let pool = pool.clone();
                let job = job.clone();
                std::thread::spawn(move || {
                    for nonce in candidates {
                        let pow = cnhash::fusion_hash(&input, nonce);
                        if pow <= job.target {
                            log::info!("device [{dev}] share found nonce=0x{nonce:016x} pow=0x{pow:016x}");
                            pool.submit(&job, nonce);
                        } else {
                            log::debug!("device [{dev}] false positive nonce=0x{nonce:016x} pow=0x{pow:016x}");
                        }
                    }
                });
            }
        }
    }
}
