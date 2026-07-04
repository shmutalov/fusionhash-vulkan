//! Compiles the GLSL compute shaders to SPIR-V at build time using `glslc`
//! (from the Vulkan SDK). When `glslc` is not available (e.g. a minimal Linux
//! rig or CI image), it falls back to the pre-built SPIR-V committed under
//! `shaders/spirv/`, so the crate builds with nothing but a Rust toolchain.
//!
//! The FP-core kernels (cn1 / sctest) are built once per correctly-rounded
//! divide variant; every variant is embedded in the binary and the right one
//! is selected per device at runtime (see `src/autotune.rs`).

use std::path::{Path, PathBuf};
use std::process::Command;

// (source shader, output basename) — kernels with a single build.
const SHADERS: &[(&str, &str)] = &[
    ("cn0.comp", "cn0.comp.spv"),
    ("cn00.comp", "cn00.comp.spv"),
    ("cn2.comp", "cn2.comp.spv"),
];
// cn2 debug variant (defines DEBUG_HASH)
const CN2_DBG_OUT: &str = "cn2_dbg.comp.spv";

// FP-core kernels × divide variants:
//   rcp       — hardware reciprocal seed + 1 Newton step (fastest)
//   markstein — bit-hack integer seed + 3 Newton steps (driver-independent seed)
//   fp64      — divide in fp64 and round back (needs shaderFloat64; the only
//               bit-exact option on drivers that do not fuse fp32 fma, e.g.
//               Mesa/ACO on GCN)
const CRDIV_SHADERS: &[&str] = &["cn1", "sctest"];
const CRDIV_VARIANTS: &[(&str, Option<&str>)] = &[
    ("rcp", Some("CRDIV_RCP")),
    ("markstein", None),
    ("fp64", Some("CRDIV_FP64")),
];

fn find_glslc() -> Option<PathBuf> {
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let exe = if cfg!(windows) { "glslc.exe" } else { "glslc" };
        for sub in ["Bin", "bin"] {
            let c = Path::new(&sdk).join(sub).join(exe);
            if c.exists() {
                return Some(c);
            }
        }
    }
    // Is glslc on PATH?
    let probe = if cfg!(windows) { "glslc.exe" } else { "glslc" };
    if Command::new(probe).arg("--version").output().is_ok() {
        return Some(PathBuf::from(probe));
    }
    None
}

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let prebuilt_dir = shader_dir.join("spirv");

    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let glslc = find_glslc();

    let compile = |src: &Path, dst: &Path, defines: &[&str]| -> bool {
        let Some(glslc) = &glslc else { return false };
        let mut cmd = Command::new(glslc);
        cmd.arg("--target-env=vulkan1.3")
            .arg("-fshader-stage=compute")
            .arg("-O")
            .arg("-I")
            .arg(&shader_dir);
        for d in defines {
            cmd.arg(format!("-D{d}"));
        }
        let status = cmd
            .arg(src)
            .arg("-o")
            .arg(dst)
            .status()
            .unwrap_or_else(|e| panic!("failed to launch glslc: {e}"));
        if !status.success() {
            panic!("glslc failed to compile {}", src.display());
        }
        true
    };

    let use_prebuilt = |name: &str, dst: &Path| {
        let src = prebuilt_dir.join(name);
        if !src.exists() {
            panic!(
                "glslc not found and no pre-built SPIR-V at {} — install the \
                 Vulkan SDK or commit shaders/spirv/{name}",
                src.display()
            );
        }
        std::fs::copy(&src, dst).expect("failed to copy pre-built SPIR-V");
    };

    for (src_name, out_name) in SHADERS {
        let dst = out_dir.join(out_name);
        if !compile(&shader_dir.join(src_name), &dst, &[]) {
            use_prebuilt(out_name, &dst);
        }
    }

    for base in CRDIV_SHADERS {
        let src = shader_dir.join(format!("{base}.comp"));
        for (variant, define) in CRDIV_VARIANTS {
            let out_name = format!("{base}_{variant}.comp.spv");
            let dst = out_dir.join(&out_name);
            let defines: Vec<&str> = define.iter().copied().collect();
            if !compile(&src, &dst, &defines) {
                use_prebuilt(&out_name, &dst);
            }
        }
    }

    let dbg_dst = out_dir.join(CN2_DBG_OUT);
    if !compile(&shader_dir.join("cn2.comp"), &dbg_dst, &["DEBUG_HASH"]) {
        use_prebuilt(CN2_DBG_OUT, &dbg_dst);
    }
}
