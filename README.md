# fusionhash-vulkan

A Rust re-implementation of the FusionHash (WarpMiner) GPU miner with a Vulkan
compute backend, targeting AMD RDNA3 (Radeon RX 7900 XT).

FusionHash is CryptoNight-GPU (`cn/gpu`, the xmr-stak variant): a 2 MiB
memory-hard hash whose inner loop is a chain of correctly-rounded 32-bit
floating-point operations. The upstream miner
([`0xFusionLayer/warpminer`](https://github.com/0xFusionLayer/warpminer)) runs it
in OpenCL; this port runs the same algorithm through Vulkan compute shaders
(GLSL → SPIR-V).

## Bit-exactness

The GPU must produce the same hash as the pool's reference validator. Three
things are handled explicitly:

* No FMA contraction. Every `a*b + c` in the reference is two separately rounded
  operations. The FP core variables are marked `precise` in GLSL so the driver
  emits `NoContraction` and does not fuse them.
* Correctly-rounded division. The reference divide is IEEE round-to-nearest (x86
  `divss` / `-cl-fp32-correctly-rounded-divide-sqrt`), but Vulkan only guarantees
  2.5 ULP for `OpFDiv`. The divisor here is always a normal number in `±[2,4)`
  (cn/gpu forces it), so `crdiv` uses a Markstein method — a bit-hack reciprocal
  seed, 3 Newton-Raphson steps (≤0.5 ULP for that range), then one FMA residual
  correction — which is correctly rounded and ~9% faster than the `fp64` fallback
  (`-DCRDIV_FP64`). Every op is IEEE fp32, so it matches the CPU reference
  bit-for-bit (validated over 800M+ divides by the self/micro tests).
* No reassociation in the reductions. The cn1 cross-lane float reductions are
  also `precise`; otherwise the AMD shader compiler reassociates the adds and the
  result drifts by a ULP after a few thousand iterations.

A CPU reference ([`src/cnhash.rs`](src/cnhash.rs)) mirrors the shaders
operation-for-operation and is used to (a) verify every candidate share before it
is submitted and (b) drive a GPU-vs-CPU comparison:

```
vulkminer --selftest      # runs the full pipeline on the GPU and compares
                          # every stage (cn0/cn00/cn1/cn2) to the CPU reference
vulkminer --microtest     # isolates the single_compute FP core over 200k records
```

## Pipeline

Each hash runs four compute kernels ([`shaders/`](shaders/)):

| kernel | file | role |
|--------|------|------|
| `cn0`  | `cn0.comp`  | Keccak-f[1600] absorb of `input‖nonce` |
| `cn00` | `cn00.comp` | "explode" — fill the 2 MiB scratchpad from the state |
| `cn1`  | `cn1.comp`  | FP32 memory-hard core (16 lanes / hash) |
| `cn2`  | `cn2.comp`  | AES "implode" + final Keccak + target compare |

VRAM is used through scratchpad shards: each shard is one `≤ 2 GiB` buffer
holding `--tps` lanes (default 960 ≈ 1.9 GiB). Several shards run per dispatch,
stage-major, so they overlap on the queue. On a 7900 XT the default (intensity 1)
uses 5 shards / 4800 lanes at ~5.9 kH/s.

The cn1 stage is dispatched in slices (`--cn1-slices`, default auto) rather
than as one 49152-iteration dispatch: a dispatch that runs for seconds cannot
be preempted, and Windows' TDR watchdog resets the GPU (`VK_ERROR_DEVICE_LOST`)
after ~2 s — which is exactly what happens on slower cards (e.g. an RX 580
needs ~2.5 s per pass). Each hash's tiny resume state (`s` + `vs`, 32 B) is
persisted between slices, so the split is bit-exact and costs nothing
measurable (~48 ms/slice on a 7900 XT). Auto starts at 16 slices and doubles
whenever a slice exceeds ~200 ms.

## Build

Requires the Vulkan SDK (for `glslc`, which `build.rs` invokes) and Rust ≥ 1.75.

```bash
cargo build --release
```

## Usage

```bash
# list Vulkan compute devices
vulkminer --info

# benchmark (no pool)
vulkminer --mock

# verify correctness on this GPU
vulkminer --selftest

# mine
vulkminer --pool wss://pool.example:1234 --user <wallet> --pass x

# pick devices / tune
vulkminer -d 1,2 --intensity 1.5
vulkminer --shards 5 --tps 960
```

Flags: `--pool --user --pass`, `--devices/-d`, `--intensity`, `--shards`,
`--tps`, `--tune` (default true: measure shard candidates at startup, cached
per device in `~/.vulkminer-tune.json`; `--tune=false` uses the formula only),
`--crdiv`, `--cn1-slices`, `--all` (include non-AMD/NVIDIA devices), `--info`,
`--mock`, `--selftest`, `--microtest`.

## Pool protocol

The stratum client ([`src/stratum.rs`](src/stratum.rs)) speaks WebSocket JSON-RPC
2.0 with the CryptoNote `login` / `job` / `submit` shape. The field mapping lives
in `parse_job` / `submit`; adjust there if a specific FusionLayer pool differs.
Every share is re-hashed on the CPU with the reference before submission, so a
mis-mapped target cannot produce a rejected share, only a missed one.

## Tuning notes for RDNA3

* `--tps` is capped by the 2 GiB max-allocation limit (960 lanes ≈ 1.9 GiB).
  Both tps and shards default to auto: the solver targets 70 in-flight
  hashes/CU on RDNA (measured: 7 shards ≈ 6.6 kH/s vs 5 ≈ 6.2 on a 7900 XT
  with the sliced cn1 pipeline) and 48/CU on GCN, within 80 % of VRAM.
* More shards is still not always faster — the 7900 XT peaks at 7 and dips at
  8 (power/thermal-bound).
* The correctly-rounded fp32 divide has three variants; **all are embedded in
  the binary and the right one is picked per device at startup**
  ([`src/autotune.rs`](src/autotune.rs)): each candidate is validated
  bit-for-bit against the CPU reference over 16k `single_compute` records
  (~100 ms) and the fastest exact one wins. `--crdiv rcp|markstein|fp64`
  forces a variant (still validated — a diverging divide can never produce an
  accepted share); `--microtest` reports the status of all three on a device.
  * `rcp` — seed the reciprocal from the hardware divide (`1.0/|b|`, one
    `v_rcp`, ≤2.5 ULP) and do a single Newton step; the Markstein residual
    correction pins the correctly-rounded quotient. Measured **+2.4 % on a
    7900 XT**; the auto-selection default everywhere it validates.
  * `markstein` — bit-hack reciprocal seed + 3 Newton steps. The seed is
    driver-independent (a pure integer op), so it needs no guarantees from the
    driver's `OpFDiv` lowering.
  * `fp64` — divide in fp64 and round back. Slow (1/16-rate fp64 on GCN) but
    the only variant that does not require the driver to emit *fused* fp32
    fmas. This matters in practice: **Mesa/ACO on GCN (e.g. RADV on an
    RX 570/580) never fuses `fma()`** — neither `precise` nor a
    `NoContraction` decoration changes that — which silently breaks both fp32
    variants by 1 ULP on ~44 % of divides. Auto-selection lands here on
    those drivers (measured ~0.45 kH/s vs ~0.7 fp32-theoretical on an
    RX 570).
* Cooperative-kernel wavefront size is pinned with `--wave auto|driver|32|64`
  (`auto` = wave64 on AMD, so each cn1/cn2 workgroup is a single wave and its
  barriers are free). On a 7900 XT `auto` matches the driver default; `32` is for
  A/B testing the barrier cost.

## License

GPL-3.0-or-later — the OpenCL kernels this is derived from originate from xmr-stak
(GPL-3.0).
