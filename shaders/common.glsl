// Shared primitives for the FusionHash (CryptoNight-GPU / cn/gpu) compute kernels.
//
// Ported bit-for-bit from the upstream OpenCL kernel (xmr-stak lineage). The
// numerically sensitive parts (the FP32 "single_compute" core) are reproduced
// so that the GPU result matches the canonical reference exactly:
//   * every multiply / add is rounded independently  -> variables are `precise`
//     so the compiler may not contract them into an FMA.
//   * division is IEEE round-to-nearest correctly rounded -> done in fp64 and
//     rounded back to fp32 (double-rounding is safe here: 53 >= 2*24+2).
//
// This file expects the includer to have enabled:
//   #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// and, for kernels that divide, shaderFloat64 must be enabled on the device.

#ifndef FUSIONHASH_COMMON_GLSL
#define FUSIONHASH_COMMON_GLSL

// ---------------------------------------------------------------------------
// Keccak-f[1600]
// ---------------------------------------------------------------------------

const uint64_t keccakf_rndc[24] = uint64_t[24](
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808aul,
    0x8000000080008000ul, 0x000000000000808bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000aul,
    0x000000008000808bul, 0x800000000000008bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800aul, 0x800000008000000aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul);

const int keccakf_rotc[24] = int[24](
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44);

const int keccakf_piln[24] = int[24](
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1);

uint64_t rotl64(uint64_t x, uint n) {
    return (x << n) | (x >> (64u - n));
}

// OpenCL bitselect(a, b, c): per bit, choose b where c==1 else a.
uint64_t bitselect64(uint64_t a, uint64_t b, uint64_t c) {
    return (a & ~c) | (b & c);
}

void keccak_f1600(inout uint64_t st[25]) {
    uint64_t t, bc[5];
    for (int round = 0; round < 24; ++round) {
        // Theta (fused with the first rho rotation, matching the reference).
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20] ^ rotl64(st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22], 1u);
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21] ^ rotl64(st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23], 1u);
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22] ^ rotl64(st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24], 1u);
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23] ^ rotl64(st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20], 1u);
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24] ^ rotl64(st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21], 1u);

        st[0] ^= bc[4];  st[5] ^= bc[4];  st[10] ^= bc[4]; st[15] ^= bc[4]; st[20] ^= bc[4];
        st[1] ^= bc[0];  st[6] ^= bc[0];  st[11] ^= bc[0]; st[16] ^= bc[0]; st[21] ^= bc[0];
        st[2] ^= bc[1];  st[7] ^= bc[1];  st[12] ^= bc[1]; st[17] ^= bc[1]; st[22] ^= bc[1];
        st[3] ^= bc[2];  st[8] ^= bc[2];  st[13] ^= bc[2]; st[18] ^= bc[2]; st[23] ^= bc[2];
        st[4] ^= bc[3];  st[9] ^= bc[3];  st[14] ^= bc[3]; st[19] ^= bc[3]; st[24] ^= bc[3];

        // Rho + Pi
        t = st[1];
        for (int i = 0; i < 24; ++i) {
            int j = keccakf_piln[i];
            bc[0] = st[j];
            st[j] = rotl64(t, uint(keccakf_rotc[i]));
            t = bc[0];
        }

        // Chi
        for (int i = 0; i < 25; i += 5) {
            uint64_t tmp1 = st[i], tmp2 = st[i + 1];
            st[i]     = bitselect64(st[i]     ^ st[i + 2], st[i],     st[i + 1]);
            st[i + 1] = bitselect64(st[i + 1] ^ st[i + 3], st[i + 1], st[i + 2]);
            st[i + 2] = bitselect64(st[i + 2] ^ st[i + 4], st[i + 2], st[i + 3]);
            st[i + 3] = bitselect64(st[i + 3] ^ tmp1,      st[i + 3], st[i + 4]);
            st[i + 4] = bitselect64(st[i + 4] ^ tmp2,      st[i + 4], tmp1);
        }

        // Iota
        st[0] ^= keccakf_rndc[round];
    }
}

uint64_t bswap64(uint64_t x) {
    x = ((x & 0x00FF00FF00FF00FFul) << 8)  | ((x >> 8)  & 0x00FF00FF00FF00FFul);
    x = ((x & 0x0000FFFF0000FFFFul) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFul);
    return (x << 32) | (x >> 32);
}

// ---------------------------------------------------------------------------
// FP32 "single_compute" core (cn/gpu inner function)
// ---------------------------------------------------------------------------

// Bitwise mask helpers over the fp32 representation (exact, no rounding).
vec4 fp_and(vec4 a, uint m) { return uintBitsToFloat(floatBitsToUint(a) & uvec4(m)); }
vec4 fp_or(vec4 a, uint m)  { return uintBitsToFloat(floatBitsToUint(a) | uvec4(m)); }

// IEEE correctly-rounded fp32 division (round-to-nearest-even).
//
// Two implementations, selected at compile time:
//   * default: Markstein — a fused-multiply-add residual method in pure fp32.
//     The divisor here is always a normal number in +/-[2,4) (cn/gpu forces it),
//     so no special-case handling is needed. Every op is IEEE fp32, so it is
//     deterministic and matches the CPU fp64 reference bit-for-bit (validated by
//     the self-test / micro-test).
//   * CRDIV_FP64: divide in fp64 and round back (correctly rounded by
//     construction, ~10% slower on RDNA3). Kept as a reference/fallback.
//   * CRDIV_RCP: seed the reciprocal from the hardware divide (`1.0/|b|`, one
//     v_rcp, <=2.5 ULP) instead of the 8-bit integer bit-hack, so a single
//     Newton step reaches the same ~0.5-ULP reciprocal that the bit-hack seed
//     needs three steps for. Same final correctly-rounded quotient (the Markstein
//     residual correction pins it regardless of seed), but ~4 fewer FMAs and it
//     offloads the seed onto the transcendental unit — which helps when the FP32
//     ALU is the bottleneck (RDNA1/2). Validate with --microtest / --selftest.
#if defined(CRDIV_FP64)
vec4 crdiv(vec4 a, vec4 b) {
    return vec4(
        float(double(a.x) / double(b.x)),
        float(double(a.y) / double(b.y)),
        float(double(a.z) / double(b.z)),
        float(double(a.w) / double(b.w)));
}
#elif defined(CRDIV_RCP)
vec4 crdiv(vec4 a, vec4 b) {
    vec4 ab = abs(b);
    // Hardware reciprocal seed (<=2.5 ULP) + one Newton-Raphson step. From a
    // ~22-bit seed one step reaches the fp32 rounding floor, matching the 8-bit
    // seed's three-step result.
    //
    // The Newton/Markstein chain is `precise`: SPIR-V only guarantees fma() is
    // actually fused when it carries NoContraction. Without it, Mesa/ACO on
    // GCN legally lowers fma to v_mad_f32 (double rounding), the residual
    // correction picks the wrong neighbour ~half the time, and the quotient is
    // off by 1 ULP (observed on RADV/Polaris). The seed division itself stays
    // relaxed so the driver may use its cheap rcp path.
    vec4 y = vec4(1.0) / ab;
    precise vec4 e = fma(-ab, y, vec4(1.0));
    precise vec4 y1 = fma(y, e, y);
    // Apply the divisor's sign: rb = 1/b.
    vec4 rb = uintBitsToFloat(floatBitsToUint(y1) | (floatBitsToUint(b) & uvec4(0x80000000u)));
    // Quotient + FMA residual correction -> correctly rounded a/b.
    precise vec4 q = a * rb;
    precise vec4 r = fma(-b, q, a);
    precise vec4 q1 = fma(r, rb, q);
    return q1;
}
#else
vec4 crdiv(vec4 a, vec4 b) {
    vec4 ab = abs(b);
    // Reciprocal seed (bit hack) + Newton-Raphson to ~correctly-rounded 1/|b|.
    // `precise` for the same reason as the rcp variant: the fma()s must be
    // truly fused or the Markstein correction breaks (see above).
    vec4 y0 = uintBitsToFloat(uvec4(0x7EF127EAu) - floatBitsToUint(ab));
    precise vec4 e0 = fma(-ab, y0, vec4(1.0));
    precise vec4 y1 = fma(y0, e0, y0);
    precise vec4 e1 = fma(-ab, y1, vec4(1.0));
    precise vec4 y2 = fma(y1, e1, y1);
    precise vec4 e2 = fma(-ab, y2, vec4(1.0));
    precise vec4 y3 = fma(y2, e2, y2);
    // Apply the divisor's sign: rb = 1/b.
    vec4 rb = uintBitsToFloat(floatBitsToUint(y3) | (floatBitsToUint(b) & uvec4(0x80000000u)));
    // Quotient + FMA residual correction -> correctly rounded a/b.
    precise vec4 q = a * rb;
    precise vec4 r = fma(-b, q, a);
    precise vec4 q1 = fma(r, rb, q);
    return q1;
}
#endif

precise vec4 fma_break(precise vec4 x) {
    x = fp_and(x, 0xFEFFFFFFu);
    return fp_or(x, 0x00800000u);
}

void sub_round(vec4 n0, vec4 n1, vec4 n2, vec4 n3, vec4 rnd_c,
               inout precise vec4 n, inout precise vec4 d, inout precise vec4 c) {
    n1 = n1 + c;
    precise vec4 nn = n0 * c;
    nn = n1 * (nn * nn);
    nn = fma_break(nn);
    n = n + nn;

    n3 = n3 - c;
    precise vec4 dd = n2 * c;
    dd = n3 * (dd * dd);
    dd = fma_break(dd);
    d = d + dd;

    c = c + rnd_c;
    c = c + vec4(0.734375);
    precise vec4 r = nn + dd;
    r = fp_and(r, 0x807FFFFFu);
    r = fp_or(r, 0x40000000u);
    c = c + r;
}

void round_compute(vec4 n0, vec4 n1, vec4 n2, vec4 n3, vec4 rnd_c,
                   inout precise vec4 c, inout precise vec4 r) {
    precise vec4 n = vec4(0.0);
    precise vec4 d = vec4(0.0);

    sub_round(n0, n1, n2, n3, rnd_c, n, d, c);
    sub_round(n1, n2, n3, n0, rnd_c, n, d, c);
    sub_round(n2, n3, n0, n1, rnd_c, n, d, c);
    sub_round(n3, n0, n1, n2, rnd_c, n, d, c);
    sub_round(n3, n2, n1, n0, rnd_c, n, d, c);
    sub_round(n2, n1, n0, n3, rnd_c, n, d, c);
    sub_round(n1, n0, n3, n2, rnd_c, n, d, c);
    sub_round(n0, n3, n2, n1, rnd_c, n, d, c);

    // ensure abs(d) > 2.0 to avoid div-by-zero / overflow
    d = fp_and(d, 0xFF7FFFFFu);
    d = fp_or(d, 0x40000000u);
    r = r + crdiv(n, d);
}

// Returns the ivec4 result; also writes the fp32 "va" contribution into `sum`.
ivec4 single_compute(vec4 n0, vec4 n1, vec4 n2, vec4 n3, float cnt, vec4 rnd_c, out vec4 sum) {
    precise vec4 c = vec4(cnt);
    precise vec4 r = vec4(0.0);

    for (int i = 0; i < 4; ++i)
        round_compute(n0, n1, n2, n3, rnd_c, c, r);

    // quick fmod by forcing exponent to 2
    r = fp_and(r, 0x807FFFFFu);
    r = fp_or(r, 0x40000000u);
    sum = r;
    r = r * vec4(536870880.0);
    return ivec4(roundEven(r));
}

// int4 byte rotate (OpenCL _mm_alignr_epi8), rot in 1..3.
ivec4 alignr_epi8(ivec4 a, uint rot) {
    uint right = 8u * rot;
    uint left  = 32u - 8u * rot;
    uvec4 u = uvec4(a);
    return ivec4(
        int((u.x >> right) | (u.y << left)),
        int((u.y >> right) | (u.z << left)),
        int((u.z >> right) | (u.w << left)),
        int((u.w >> right) | (u.x << left)));
}

void single_compute_wrap(uint rot, ivec4 v0, ivec4 v1, ivec4 v2, ivec4 v3,
                         float cnt, vec4 rnd_c, out vec4 va_out, out ivec4 out_val) {
    vec4 n0 = vec4(v0);
    vec4 n1 = vec4(v1);
    vec4 n2 = vec4(v2);
    vec4 n3 = vec4(v3);
    ivec4 r = single_compute(n0, n1, n2, n3, cnt, rnd_c, va_out);
    out_val = (rot == 0u) ? r : alignr_epi8(r, rot);
}

// Look-up permutation and per-lane constants for cn1 (flattened 16x4).
const uint cn_look[64] = uint[64](
    0u,1u,2u,3u,  0u,2u,3u,1u,  0u,3u,1u,2u,  0u,3u,2u,1u,
    1u,0u,2u,3u,  1u,2u,3u,0u,  1u,3u,0u,2u,  1u,3u,2u,0u,
    2u,1u,0u,3u,  2u,0u,3u,1u,  2u,3u,1u,0u,  2u,3u,0u,1u,
    3u,1u,2u,0u,  3u,2u,0u,1u,  3u,0u,1u,2u,  3u,0u,2u,1u);

const float cn_ccnt[16] = float[16](
    1.34375,   1.28125,   1.359375,  1.3671875,
    1.4296875, 1.3984375, 1.3828125, 1.3046875,
    1.4140625, 1.2734375, 1.2578125, 1.2890625,
    1.3203125, 1.3515625, 1.3359375, 1.4609375);

#endif // FUSIONHASH_COMMON_GLSL
