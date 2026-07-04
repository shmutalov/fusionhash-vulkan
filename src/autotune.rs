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

use crate::microtest;
use crate::vk::Gpu;
use anyhow::{bail, Result};
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
