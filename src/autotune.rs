//! Per-device kernel selection. Every correctly-rounded-divide variant of the
//! FP core is embedded in the binary; at startup each device gets the fastest
//! variant that is *empirically bit-exact there*, validated by a short
//! GPU-vs-CPU run of `single_compute` (the same comparison `--microtest`
//! does, over fewer records).
//!
//! This is deliberately not keyed on vendor/driver strings: the failure mode
//! it guards against is compiler lowering (e.g. Mesa/ACO on GCN never fuses
//! fp32 fma, which silently breaks both fp32 divide variants by 1 ULP), and
//! that is a property of the installed driver build, not of the GPU name.

use crate::config::Config;
use crate::microtest;
use crate::miner::{Miner, MEMORY};
use crate::vk::{Gpu, PhysicalDevice};
use anyhow::{bail, Result};
use std::path::PathBuf;
use std::sync::Arc;

/// Correctly-rounded fp32 divide implementations, fastest first.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Crdiv {
    /// Hardware reciprocal seed + 1 Newton step + Markstein residual.
    Rcp,
    /// Bit-hack integer seed + 3 Newton steps + Markstein residual.
    Markstein,
    /// Divide in fp64 and round back. Slow (1/16-rate fp64 on GCN) but the
    /// only variant with no dependence on the driver's fp32 fma/rcp lowering.
    Fp64,
}

impl Crdiv {
    pub const ALL: [Crdiv; 3] = [Crdiv::Rcp, Crdiv::Markstein, Crdiv::Fp64];

    pub fn name(self) -> &'static str {
        match self {
            Crdiv::Rcp => "rcp",
            Crdiv::Markstein => "markstein",
            Crdiv::Fp64 => "fp64",
        }
    }

    /// cn1 SPIR-V for this variant.
    pub fn cn1_spv(self) -> &'static [u8] {
        match self {
            Crdiv::Rcp => include_bytes!(concat!(env!("OUT_DIR"), "/cn1_rcp.comp.spv")),
            Crdiv::Markstein => {
                include_bytes!(concat!(env!("OUT_DIR"), "/cn1_markstein.comp.spv"))
            }
            Crdiv::Fp64 => include_bytes!(concat!(env!("OUT_DIR"), "/cn1_fp64.comp.spv")),
        }
    }

    /// sctest (single_compute isolation) SPIR-V for this variant.
    pub fn sctest_spv(self) -> &'static [u8] {
        match self {
            Crdiv::Rcp => include_bytes!(concat!(env!("OUT_DIR"), "/sctest_rcp.comp.spv")),
            Crdiv::Markstein => {
                include_bytes!(concat!(env!("OUT_DIR"), "/sctest_markstein.comp.spv"))
            }
            Crdiv::Fp64 => include_bytes!(concat!(env!("OUT_DIR"), "/sctest_fp64.comp.spv")),
        }
    }

    fn parse(s: &str) -> Option<Crdiv> {
        match s {
            "rcp" => Some(Crdiv::Rcp),
            "markstein" => Some(Crdiv::Markstein),
            "fp64" => Some(Crdiv::Fp64),
            _ => None,
        }
    }
}

/// Lane density the layout solver aims for, per architecture.
/// GCN (wave64-only): 48/CU — xmr-stak's profiled optimum (6 waves ×
/// worksize 8). RDNA (wave32-capable, i.e. min subgroup size 32): 70/CU —
/// measured on a 7900 XT with the sliced cn1 pipeline, where 7×960 (6720
/// lanes, 70/CU) does ~6.62 kH/s vs ~6.18 at the old 48/CU target; the
/// earlier "saturates at 3–5 shards" observation predates the slicing.
fn lanes_per_cu(pd: &PhysicalDevice) -> f64 {
    if pd.min_subgroup_size == 32 {
        70.0
    } else {
        48.0
    }
}
/// Preferred threads-per-shard. Kept as the starting point so devices with
/// enough VRAM (e.g. the validated 5×960 on a 7900 XT) resolve exactly as the
/// previous fixed default did.
const DEFAULT_TPS: u32 = 960;
/// Fraction of device-local VRAM the scratchpads may occupy.
const VRAM_BUDGET: f64 = 0.80;

/// Resolve threads-per-shard and shard count for one device.
///
/// `--tps`/`--shards` (nonzero) force their respective values. In auto mode
/// the solver starts from `DEFAULT_TPS` and only shrinks tps (in steps of 64)
/// when the per-shard granularity would starve the CU-based lane target —
/// e.g. a 4 GiB RX 570 fits only 1×960 lanes at the default, but 2×768 hits
/// the 1536-lane target exactly.
pub fn select_layout(cfg: &Config, pd: &PhysicalDevice) -> (u32, u32) {
    let cu = if pd.compute_units > 0 { pd.compute_units } else { 32 };
    let target = (cu as f64 * lanes_per_cu(pd) * cfg.intensity).max(64.0);

    let shards_for = |tps: u32| -> u32 {
        let max_by_mem =
            (pd.device_local_mem as f64 * VRAM_BUDGET / (MEMORY * tps as u64) as f64) as u32;
        let desired = (target / tps as f64).round() as u32;
        desired.clamp(1, max_by_mem.max(1))
    };

    // Forced tps: legacy behaviour (shards from the CU formula unless forced).
    if cfg.tps > 0 {
        let shards = if cfg.shards > 0 { cfg.shards } else { shards_for(cfg.tps) };
        return (cfg.tps, shards);
    }

    // Largest tps the device's max allocation permits, in units of 64.
    let max_tps_alloc = (((pd.max_alloc / MEMORY) as u32) / 64 * 64).max(64);
    let tps0 = DEFAULT_TPS.min(max_tps_alloc);
    let shards0 = if cfg.shards > 0 { cfg.shards } else { shards_for(tps0) };
    let lanes0 = tps0 * shards0;
    // Close enough to the target (or shards forced): keep the default tps.
    if cfg.shards > 0 || lanes0 as f64 >= target * 0.95 {
        return (tps0, shards0);
    }

    // Starved: search smaller tps for the layout closest to the target from
    // below (score = min(lanes, target); overshoot beyond the target is not
    // rewarded). Ties keep the larger tps (fewer shards).
    let score = |lanes: u32| (lanes as f64).min(target);
    let (mut best, mut best_score) = ((tps0, shards0), score(lanes0));
    let mut tps = tps0;
    while tps > 64 {
        tps -= 64;
        let shards = shards_for(tps);
        let s = score(tps * shards);
        if s > best_score {
            best = (tps, shards);
            best_score = s;
        }
    }
    best
}

/// One warm-up plus this many timed full passes per shard-count candidate.
const TUNE_PASSES: u32 = 2;

/// A tuned result, keyed by everything that would invalidate it.
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq)]
struct TuneKey {
    device: String,
    device_id: u32,
    driver: String,
    version: String,
    crdiv: String,
    tps: u32,
    intensity: f64,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct TuneEntry {
    key: TuneKey,
    shards: u32,
    khs: f64,
}

fn tune_cache_path() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".vulkminer-tune.json")
}

fn load_tune_cache() -> Vec<TuneEntry> {
    std::fs::read_to_string(tune_cache_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_tune_entry(entry: TuneEntry) {
    let mut cache = load_tune_cache();
    cache.retain(|e| e.key != entry.key);
    cache.push(entry);
    if let Ok(s) = serde_json::to_string_pretty(&cache) {
        if let Err(e) = std::fs::write(tune_cache_path(), s) {
            log::warn!("could not write {}: {e}", tune_cache_path().display());
        }
    }
}

/// Time `TUNE_PASSES` full pipeline passes for one layout; returns kH/s.
/// The miner (and its scratchpads) is dropped before the next candidate runs.
fn measure_layout(
    gpu: &Arc<Gpu>,
    tps: u32,
    shards: u32,
    wave: Option<u32>,
    cn1_slices: u32,
    crdiv: Crdiv,
) -> Result<f64> {
    let mut miner = Miner::new(gpu.clone(), tps, shards, false, wave, cn1_slices, crdiv)?;
    let mut input = [0u8; 128];
    for (i, b) in input.iter_mut().enumerate().take(76) {
        *b = ((i * 11 + 5) & 0xff) as u8;
    }
    input[76] = 0x01;
    miner.set_input(&input);
    let lanes = miner.hashes_per_iter();
    miner.run_iteration(0, 0)?; // warm-up (also lets auto cn1 slicing settle)
    let t = std::time::Instant::now();
    for p in 0..TUNE_PASSES {
        miner.run_iteration((p as u64 + 1) * lanes, 0)?;
    }
    Ok((lanes * TUNE_PASSES as u64) as f64 / t.elapsed().as_secs_f64() / 1000.0)
}

/// Measured shard tuning: hill-climb the shard count around the formula's
/// starting point (`shards0`), one VRAM-budget step at a time, and keep the
/// fastest. Results are cached per device/driver/version/config so later
/// startups skip the sweep.
pub fn tune_shards(
    gpu: &Arc<Gpu>,
    cfg: &Config,
    tps: u32,
    shards0: u32,
    wave: Option<u32>,
    crdiv: Crdiv,
) -> Result<u32> {
    let pd = &gpu.pdev;
    let max_shards = ((pd.device_local_mem as f64 * VRAM_BUDGET
        / (MEMORY * tps as u64) as f64) as u32)
        .max(1);

    let key = TuneKey {
        device: pd.name.clone(),
        device_id: pd.device_id,
        driver: pd.driver_info.clone(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        crdiv: crdiv.name().to_string(),
        tps,
        intensity: cfg.intensity,
    };
    if let Some(e) = load_tune_cache().iter().find(|e| e.key == key) {
        let shards = e.shards.clamp(1, max_shards);
        log::info!(
            "[{}] tuned shards={} ({:.3} kH/s, cached in {})",
            pd.name,
            shards,
            e.khs,
            tune_cache_path().display()
        );
        return Ok(shards);
    }

    log::info!(
        "[{}] tuning shard count (start {}, 1..={} possible, {} timed passes each; \
         --tune=false skips this)",
        pd.name,
        shards0,
        max_shards,
        TUNE_PASSES
    );

    let mut results: Vec<(u32, f64)> = Vec::new();
    let mut measured = |n: u32, results: &mut Vec<(u32, f64)>| -> Result<f64> {
        if let Some(&(_, khs)) = results.iter().find(|(m, _)| *m == n) {
            return Ok(khs);
        }
        let khs = match measure_layout(gpu, tps, n, wave, cfg.cn1_slices, crdiv) {
            Ok(k) => k,
            Err(e) => {
                // e.g. allocation failure at the VRAM edge — score it out.
                log::warn!("[{}] shards={n}: measurement failed ({e:#})", pd.name);
                0.0
            }
        };
        log::info!("[{}]   shards={n}: {khs:.3} kH/s", pd.name);
        results.push((n, khs));
        Ok(khs)
    };

    let start = shards0.clamp(1, max_shards);
    for n in start.saturating_sub(1).max(1)..=(start + 1).min(max_shards) {
        measured(n, &mut results)?;
    }
    // Hill-climb while the best sits on the edge of the measured range.
    loop {
        let &(best, _) = results
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("at least one candidate");
        let lo = results.iter().map(|&(n, _)| n).min().unwrap();
        let hi = results.iter().map(|&(n, _)| n).max().unwrap();
        if best == hi && hi < max_shards {
            measured(hi + 1, &mut results)?;
        } else if best == lo && lo > 1 {
            measured(lo - 1, &mut results)?;
        } else {
            break;
        }
    }

    let &(best, khs) = results.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
    if khs <= 0.0 {
        bail!("[{}] every tuning candidate failed", pd.name);
    }
    log::info!("[{}] tuned shards={best} ({khs:.3} kH/s) — cached", pd.name);
    save_tune_entry(TuneEntry { key, shards: best, khs });
    Ok(best)
}

/// Records for the startup validation. The observed failure modes are dense
/// (~44% of records on a non-fusing driver), so even a few hundred records
/// would do; 16384 costs ~100 ms and makes a 0.1%-rate defect vanishingly
/// unlikely to slip through (0.999^16384 ≈ 8e-8).
const VALIDATE_RECORDS: usize = 16384;

/// Resolve the divide variant for one device: "auto" tries each variant
/// fastest-first and picks the first that is bit-exact on this device;
/// a concrete name forces that variant but still refuses to run if it
/// diverges (a diverging divide can never produce an accepted share).
pub fn select_crdiv(gpu: &Arc<Gpu>, requested: &str) -> Result<Crdiv> {
    match requested {
        "auto" => {
            for crdiv in Crdiv::ALL {
                if crdiv == Crdiv::Fp64 && !gpu.pdev.shader_float64 {
                    continue;
                }
                let t = std::time::Instant::now();
                if microtest::validate(gpu, crdiv, VALIDATE_RECORDS)? {
                    log::info!(
                        "[{}] divide={} (auto: bit-exact over {} records, {} ms)",
                        gpu.pdev.name,
                        crdiv.name(),
                        VALIDATE_RECORDS,
                        t.elapsed().as_millis()
                    );
                    return Ok(crdiv);
                }
                log::warn!(
                    "[{}] divide variant '{}' diverges from the CPU reference on \
                     this driver — trying the next one",
                    gpu.pdev.name,
                    crdiv.name()
                );
            }
            bail!(
                "no bit-exact divide variant on {} ({}) — cannot mine on this device",
                gpu.pdev.name,
                gpu.pdev.driver_info
            )
        }
        name => {
            let Some(crdiv) = Crdiv::parse(name) else {
                bail!("--crdiv must be auto, rcp, markstein or fp64 (got '{name}')");
            };
            if crdiv == Crdiv::Fp64 && !gpu.pdev.shader_float64 {
                bail!("--crdiv fp64 requested but {} lacks shaderFloat64", gpu.pdev.name);
            }
            if !microtest::validate(gpu, crdiv, VALIDATE_RECORDS)? {
                bail!(
                    "--crdiv {} diverges from the CPU reference on {} ({}) — \
                     it would mine zero accepted shares; use --crdiv auto",
                    crdiv.name(),
                    gpu.pdev.name,
                    gpu.pdev.driver_info
                );
            }
            log::info!("[{}] divide={} (forced, validated)", gpu.pdev.name, crdiv.name());
            Ok(crdiv)
        }
    }
}
