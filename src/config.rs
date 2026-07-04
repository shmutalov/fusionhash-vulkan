use clap::Parser;

/// FusionHash (CryptoNight-GPU) miner with a Vulkan compute backend, tuned for
/// AMD RDNA3 (Radeon RX 7900 XT).
#[derive(Parser, Debug, Clone)]
#[command(name = "vulkminer", version, about)]
pub struct Config {
    /// List all Vulkan compute devices and exit.
    #[arg(long)]
    pub info: bool,

    /// Benchmark mode: run the pipeline against a synthetic job, no pool.
    #[arg(long, alias = "bench")]
    pub mock: bool,

    /// Run the GPU-vs-CPU bit-exactness self test and exit.
    #[arg(long)]
    pub selftest: bool,

    /// Run the single_compute micro-test (FP core isolation) and exit.
    #[arg(long)]
    pub microtest: bool,

    /// Use every Vulkan device, not just AMD/NVIDIA GPUs.
    #[arg(long)]
    pub all: bool,

    /// Pool URL (ws:// or wss://).
    #[arg(long, default_value = "ws://127.0.0.1:8546")]
    pub pool: String,

    /// Pool username / wallet.
    #[arg(long, default_value = "")]
    pub user: String,

    /// Pool password.
    #[arg(long, default_value = "x")]
    pub pass: String,

    /// Intensity factor; scales the default shard count per device.
    #[arg(long, default_value_t = 1.0)]
    pub intensity: f64,

    /// Comma-separated 1-based device indices (e.g. "1,3"). Empty = all selected.
    #[arg(long, short = 'd', default_value = "")]
    pub devices: String,

    /// Override the number of 2 GiB scratchpad shards per device (0 = auto).
    #[arg(long, default_value_t = 0)]
    pub shards: u32,

    /// Threads per shard (multiple of 64). 0 = auto: solved jointly with the
    /// shard count from the device's CU count, VRAM and max allocation
    /// (starts at 960 and shrinks only when a small-VRAM card would otherwise
    /// fall short of the per-CU lane target).
    #[arg(long, default_value_t = 0)]
    pub tps: u32,

    /// Measure candidate shard counts at startup (one warm-up + 2 timed full
    /// passes each, hill-climbing from the computed target) and keep the
    /// fastest. Results are cached per device/driver/version/config in
    /// ~/.vulkminer-tune.json, so only the first launch pays the sweep.
    /// Disable with --tune=false; a forced --shards also skips it.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set,
          num_args = 0..=1, default_missing_value = "true")]
    pub tune: bool,

    /// Correctly-rounded divide variant for the FP core: "auto" (validate each
    /// variant on the device at startup and keep the fastest bit-exact one),
    /// "rcp", "markstein" or "fp64". Forced variants are still validated —
    /// a diverging divide can never produce an accepted share.
    #[arg(long, default_value = "auto")]
    pub crdiv: String,

    /// cn1 dispatch slices per pass (0 = auto). A single dispatch running the
    /// full 49152-iteration cn1 loop takes multiple seconds on slower GPUs and
    /// trips the Windows TDR watchdog (~2 s), killing the device. Auto starts
    /// at 16 slices and doubles whenever a slice exceeds ~200 ms.
    #[arg(long, default_value_t = 0)]
    pub cn1_slices: u32,

    /// Wavefront size for the cooperative kernels (cn1/cn2): "auto" (wave64 on
    /// AMD, so each cooperative workgroup is a single wave and the barriers are
    /// free), "driver" (no pinning), "32", or "64". "32"/"driver" are mainly for
    /// A/B testing the barrier cost.
    #[arg(long, default_value = "auto")]
    pub wave: String,

    /// Write a JSON stats file periodically (for HiveOS / monitoring).
    #[arg(long, default_value = "")]
    pub stats_file: String,
}

impl Config {
    pub fn selected_indices(&self) -> Vec<usize> {
        self.devices
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .filter(|&i| i > 0)
            .collect()
    }
}
