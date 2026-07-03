//! Compiles the GLSL compute shaders to SPIR-V at build time using `glslc`
//! (from the Vulkan SDK). When `glslc` is not available (e.g. a minimal Linux
//! rig or CI image), it falls back to the pre-built SPIR-V committed under
//! `shaders/spirv/`, so the crate builds with nothing but a Rust toolchain.

use std::path::{Path, PathBuf};
use std::process::Command;

// (source shader, output basename)
const SHADERS: &[(&str, &str)] = &[
    ("cn0.comp", "cn0.comp.spv"),
    ("cn00.comp", "cn00.comp.spv"),
    ("cn1.comp", "cn1.comp.spv"),
    ("cn2.comp", "cn2.comp.spv"),
    ("sctest.comp", "sctest.comp.spv"),
];
// cn2 debug variant (defines DEBUG_HASH)
const CN2_DBG_OUT: &str = "cn2_dbg.comp.spv";

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
    println!("cargo:rerun-if-env-changed=CRDIV");

    // Correctly-rounded fp32 divide variant for the cn/gpu core (cn1 / sctest):
    //   (unset)/rcp -> hardware reciprocal seed + 1 Newton step (default)
    //   markstein   -> bit-hack seed + 3 Newton steps (driver-independent seed)
    //   fp64        -> divide in fp64 and round back
    let crdiv_env = std::env::var("CRDIV").ok();
    let crdiv: Vec<&str> = match crdiv_env.as_deref() {
        Some("markstein") => vec![],
        Some("fp64") => vec!["CRDIV_FP64"],
        _ => vec!["CRDIV_RCP"],
    };
    // The committed prebuilt SPIR-V is built with the default (rcp), so only a
    // non-default override needs glslc; the default falls back cleanly (e.g. CI
    // runners without a Vulkan SDK).
    let needs_glslc = matches!(crdiv_env.as_deref(), Some("markstein") | Some("fp64"));

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
        // Only the FP-core shaders divide; the CRDIV define is inert elsewhere.
        let defines: &[&str] = if matches!(*src_name, "cn1.comp" | "sctest.comp") {
            &crdiv
        } else {
            &[]
        };
        if !compile(&shader_dir.join(src_name), &dst, defines) {
            if needs_glslc {
                panic!(
                    "CRDIV={:?} overrides the default divide but glslc (Vulkan SDK) \
                     is unavailable to recompile the shaders",
                    crdiv
                );
            }
            use_prebuilt(out_name, &dst);
        }
    }
    let dbg_dst = out_dir.join(CN2_DBG_OUT);
    if !compile(&shader_dir.join("cn2.comp"), &dbg_dst, &["DEBUG_HASH"]) {
        use_prebuilt(CN2_DBG_OUT, &dbg_dst);
    }
}
