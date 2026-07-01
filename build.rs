//! Compiles the GLSL compute shaders to SPIR-V at build time using `glslc`
//! (shipped with the Vulkan SDK). The resulting `.spv` blobs are written to
//! `OUT_DIR` and pulled into the binary with `include_bytes!`.

use std::path::{Path, PathBuf};
use std::process::Command;

const SHADERS: &[&str] = &["cn0.comp", "cn00.comp", "cn1.comp", "cn2.comp", "sctest.comp"];

fn find_glslc() -> PathBuf {
    // Prefer the compiler that ships with the installed Vulkan SDK, fall back to PATH.
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let exe = if cfg!(windows) { "glslc.exe" } else { "glslc" };
        let candidate = Path::new(&sdk).join("Bin").join(exe);
        if candidate.exists() {
            return candidate;
        }
        let candidate = Path::new(&sdk).join("bin").join(exe);
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("glslc")
}

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let glslc = find_glslc();

    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let compile = |src: &Path, dst: &Path, defines: &[&str]| {
        let mut cmd = Command::new(&glslc);
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
            .unwrap_or_else(|e| panic!("failed to launch glslc ({}): {e}", glslc.display()));
        if !status.success() {
            panic!("glslc failed to compile {}", src.display());
        }
    };

    for shader in SHADERS {
        let src = shader_dir.join(shader);
        println!("cargo:rerun-if-changed={}", src.display());
        compile(&src, &out_dir.join(format!("{shader}.spv")), &[]);
    }

    // Debug variant of cn2 that also emits the per-lane PoW value, used by the
    // GPU-vs-CPU self test.
    compile(
        &shader_dir.join("cn2.comp"),
        &out_dir.join("cn2_dbg.comp.spv"),
        &["DEBUG_HASH"],
    );
}
