# HiveOS integration

`vulkminer` as a HiveOS **custom miner**, plus how to build the flight sheet.

## 1. Build the Linux binary and package

HiveOS is Ubuntu-based, so it needs a Linux binary (not the Windows `.exe`).
From the repo root:

```bash
bash hiveos/build-linux.sh     # builds ./vulkminer (linux x86_64) via Docker
bash hiveos/package.sh         # -> vulkminer-0.1.0.tar.gz
```

`build-linux.sh` uses [Dockerfile](Dockerfile) (a plain `rust` image — no Vulkan
SDK or OpenSSL needed, since `build.rs` falls back to the committed SPIR-V and
the pool client uses rustls).

Alternatively, build directly on a rig / any Linux box with Rust installed:

```bash
cargo build --release          # -> target/release/vulkminer
```

The prebuilt `vulkminer-<version>.tar.gz` is also attached to the GitHub release.

## 2. Install the custom miner in HiveOS

Host the tarball somewhere reachable (the GitHub release URL works), then on the
rig **Miners** step of the flight sheet choose **Custom** and set:

- **Miner name**: `vulkminer`
- **Installation URL**: the direct link to `vulkminer-0.1.0.tar.gz`

HiveOS downloads and unpacks it to `/hive/miners/custom/vulkminer/`. You can also
`hpkg install vulkminer-0.1.0.tar.gz` on the rig directly.

## 3. Flight sheet

Create a flight sheet:

- **Coin**: pick any (or a custom coin); it is not used for hashing.
- **Wallet**: your address, e.g. `0xEe73Ed81501Fa503FC708265A43B07dCf86A8763`
- **Pool** → *Configure in miner*:
  - **URL**: `ws://fxl.baikalmine.com:2030`  (a `host:port`, `stratum+tcp://`,
    or `ssl://` value is normalised to `ws://` / `wss://` automatically)
- **Miner**: `vulkminer` (the custom miner above)
  - **Wallet and worker template**: `%WAL%`  (just the address; this pool does
    not use a worker suffix)
  - **Pass**: `x`
  - **Extra config arguments** (optional): CLI flags, e.g.
    - `--intensity 0.5`  — lower GPU load so a connected display stays responsive
    - `-d 0`             — select a specific device
    - `--shards 4`       — fix the shard count

Apply the flight sheet to the rig. The miner reports hashrate (kH/s), accepted /
rejected shares, and per-GPU temp/fan (matched by PCI bus) back to the HiveOS
dashboard.

## Notes

- Algorithm label is `cn/gpu`. Expect ~5 kH/s per RX 7900 XT.
- Requires a working Vulkan driver on the rig (AMD RDNA/RDNA2/RDNA3 with the
  amdgpu stack). Check with `vulkminer --info`.
- The miner writes `/var/log/miner/vulkminer/vulkminer.stats.json`; `h-stats.sh`
  turns that into HiveOS stats.
