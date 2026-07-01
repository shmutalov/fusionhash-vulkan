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

    /// Threads per shard (multiple of 64, <= ~1000 to stay under the 2 GiB
    /// allocation limit).
    #[arg(long, default_value_t = 960)]
    pub tps: u32,
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
