//! Bit-exact CPU reference for FusionHash (CryptoNight-GPU / cn/gpu).
//!
//! Used to (a) verify candidate shares before submitting them to the pool and
//! (b) drive the GPU self-test. It mirrors the GLSL kernels operation for
//! operation: every fp32 multiply/add is rounded independently and division is
//! performed in fp64 (== IEEE correctly-rounded fp32 divide).

pub const MEMORY: usize = 2 * 1024 * 1024;
pub const ITERATIONS: usize = 49152;
pub const MASK: usize = ((MEMORY - 1) >> 6) << 6; // 0x1FFFC0

// ---------------------------------------------------------------------------
// Keccak-f[1600]
// ---------------------------------------------------------------------------

const KECCAK_RNDC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];
const KECCAK_ROTC: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];
const KECCAK_PILN: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

#[inline]
fn bitselect(a: u64, b: u64, c: u64) -> u64 {
    (a & !c) | (b & c)
}

pub fn keccakf(st: &mut [u64; 25]) {
    let mut bc = [0u64; 5];
    for round in 0..24 {
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20]
            ^ (st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22]).rotate_left(1);
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21]
            ^ (st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23]).rotate_left(1);
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22]
            ^ (st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24]).rotate_left(1);
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23]
            ^ (st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20]).rotate_left(1);
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24]
            ^ (st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21]).rotate_left(1);

        st[0] ^= bc[4]; st[5] ^= bc[4]; st[10] ^= bc[4]; st[15] ^= bc[4]; st[20] ^= bc[4];
        st[1] ^= bc[0]; st[6] ^= bc[0]; st[11] ^= bc[0]; st[16] ^= bc[0]; st[21] ^= bc[0];
        st[2] ^= bc[1]; st[7] ^= bc[1]; st[12] ^= bc[1]; st[17] ^= bc[1]; st[22] ^= bc[1];
        st[3] ^= bc[2]; st[8] ^= bc[2]; st[13] ^= bc[2]; st[18] ^= bc[2]; st[23] ^= bc[2];
        st[4] ^= bc[3]; st[9] ^= bc[3]; st[14] ^= bc[3]; st[19] ^= bc[3]; st[24] ^= bc[3];

        let mut t = st[1];
        for i in 0..24 {
            let j = KECCAK_PILN[i];
            let tmp = st[j];
            st[j] = t.rotate_left(KECCAK_ROTC[i]);
            t = tmp;
        }

        let mut i = 0;
        while i < 25 {
            let t1 = st[i];
            let t2 = st[i + 1];
            st[i] = bitselect(st[i] ^ st[i + 2], st[i], st[i + 1]);
            st[i + 1] = bitselect(st[i + 1] ^ st[i + 3], st[i + 1], st[i + 2]);
            st[i + 2] = bitselect(st[i + 2] ^ st[i + 4], st[i + 2], st[i + 3]);
            st[i + 3] = bitselect(st[i + 3] ^ t1, st[i + 3], st[i + 4]);
            st[i + 4] = bitselect(st[i + 4] ^ t2, st[i + 4], t1);
            i += 5;
        }

        st[0] ^= KECCAK_RNDC[round];
    }
}

#[inline]
fn bswap64(x: u64) -> u64 {
    x.swap_bytes()
}

// ---------------------------------------------------------------------------
// fp32 single_compute core
// ---------------------------------------------------------------------------

type F4 = [f32; 4];
type I4 = [i32; 4];

#[inline]
fn fp_and(a: F4, m: u32) -> F4 {
    [
        f32::from_bits(a[0].to_bits() & m),
        f32::from_bits(a[1].to_bits() & m),
        f32::from_bits(a[2].to_bits() & m),
        f32::from_bits(a[3].to_bits() & m),
    ]
}
#[inline]
fn fp_or(a: F4, m: u32) -> F4 {
    [
        f32::from_bits(a[0].to_bits() | m),
        f32::from_bits(a[1].to_bits() | m),
        f32::from_bits(a[2].to_bits() | m),
        f32::from_bits(a[3].to_bits() | m),
    ]
}
#[inline]
fn vadd(a: F4, b: F4) -> F4 { [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]] }
#[inline]
fn vsub(a: F4, b: F4) -> F4 { [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]] }
#[inline]
fn vmul(a: F4, b: F4) -> F4 { [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]] }
#[inline]
fn splat(x: f32) -> F4 { [x, x, x, x] }
#[inline]
fn crdiv(a: F4, b: F4) -> F4 {
    [
        ((a[0] as f64) / (b[0] as f64)) as f32,
        ((a[1] as f64) / (b[1] as f64)) as f32,
        ((a[2] as f64) / (b[2] as f64)) as f32,
        ((a[3] as f64) / (b[3] as f64)) as f32,
    ]
}
#[inline]
fn fma_break(x: F4) -> F4 {
    fp_or(fp_and(x, 0xFEFFFFFF), 0x00800000)
}

fn sub_round(n0: F4, n1: F4, n2: F4, n3: F4, rnd_c: F4, n: &mut F4, d: &mut F4, c: &mut F4) {
    let n1 = vadd(n1, *c);
    let mut nn = vmul(n0, *c);
    nn = vmul(n1, vmul(nn, nn));
    nn = fma_break(nn);
    *n = vadd(*n, nn);

    let n3 = vsub(n3, *c);
    let mut dd = vmul(n2, *c);
    dd = vmul(n3, vmul(dd, dd));
    dd = fma_break(dd);
    *d = vadd(*d, dd);

    *c = vadd(*c, rnd_c);
    *c = vadd(*c, splat(0.734375));
    let mut r = vadd(nn, dd);
    r = fp_and(r, 0x807FFFFF);
    r = fp_or(r, 0x40000000);
    *c = vadd(*c, r);
}

fn round_compute(n0: F4, n1: F4, n2: F4, n3: F4, rnd_c: F4, c: &mut F4, r: &mut F4) {
    let mut n = [0f32; 4];
    let mut d = [0f32; 4];
    sub_round(n0, n1, n2, n3, rnd_c, &mut n, &mut d, c);
    sub_round(n1, n2, n3, n0, rnd_c, &mut n, &mut d, c);
    sub_round(n2, n3, n0, n1, rnd_c, &mut n, &mut d, c);
    sub_round(n3, n0, n1, n2, rnd_c, &mut n, &mut d, c);
    sub_round(n3, n2, n1, n0, rnd_c, &mut n, &mut d, c);
    sub_round(n2, n1, n0, n3, rnd_c, &mut n, &mut d, c);
    sub_round(n1, n0, n3, n2, rnd_c, &mut n, &mut d, c);
    sub_round(n0, n3, n2, n1, rnd_c, &mut n, &mut d, c);

    d = fp_and(d, 0xFF7FFFFF);
    d = fp_or(d, 0x40000000);
    *r = vadd(*r, crdiv(n, d));
}

#[inline]
fn rte_i32(x: f32) -> i32 {
    x.round_ties_even() as i32
}

fn single_compute(n0: F4, n1: F4, n2: F4, n3: F4, cnt: f32, rnd_c: F4) -> (I4, F4) {
    let mut c = splat(cnt);
    let mut r = [0f32; 4];
    for _ in 0..4 {
        round_compute(n0, n1, n2, n3, rnd_c, &mut c, &mut r);
    }
    r = fp_and(r, 0x807FFFFF);
    r = fp_or(r, 0x40000000);
    let sum = r;
    let r = vmul(r, splat(536870880.0));
    ([rte_i32(r[0]), rte_i32(r[1]), rte_i32(r[2]), rte_i32(r[3])], sum)
}

/// Micro-test hook: run `single_compute` on raw inputs (matches sctest.comp).
pub fn sc_test(v: &[i32; 16], rnd_c: [f32; 4], cnt: f32) -> ([i32; 4], [f32; 4]) {
    let n0 = [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32];
    let n1 = [v[4] as f32, v[5] as f32, v[6] as f32, v[7] as f32];
    let n2 = [v[8] as f32, v[9] as f32, v[10] as f32, v[11] as f32];
    let n3 = [v[12] as f32, v[13] as f32, v[14] as f32, v[15] as f32];
    single_compute(n0, n1, n2, n3, cnt, rnd_c)
}

fn alignr_epi8(a: I4, rot: u32) -> I4 {
    let right = 8 * rot;
    let left = 32 - 8 * rot;
    let u = [a[0] as u32, a[1] as u32, a[2] as u32, a[3] as u32];
    [
        ((u[0] >> right) | (u[1] << left)) as i32,
        ((u[1] >> right) | (u[2] << left)) as i32,
        ((u[2] >> right) | (u[3] << left)) as i32,
        ((u[3] >> right) | (u[0] << left)) as i32,
    ]
}

fn single_compute_wrap(rot: u32, v0: I4, v1: I4, v2: I4, v3: I4, cnt: f32, rnd_c: F4) -> (F4, I4) {
    let n0 = [v0[0] as f32, v0[1] as f32, v0[2] as f32, v0[3] as f32];
    let n1 = [v1[0] as f32, v1[1] as f32, v1[2] as f32, v1[3] as f32];
    let n2 = [v2[0] as f32, v2[1] as f32, v2[2] as f32, v2[3] as f32];
    let n3 = [v3[0] as f32, v3[1] as f32, v3[2] as f32, v3[3] as f32];
    let (r, sum) = single_compute(n0, n1, n2, n3, cnt, rnd_c);
    let out = if rot == 0 { r } else { alignr_epi8(r, rot) };
    (sum, out)
}

const CN_LOOK: [usize; 64] = [
    0, 1, 2, 3, 0, 2, 3, 1, 0, 3, 1, 2, 0, 3, 2, 1,
    1, 0, 2, 3, 1, 2, 3, 0, 1, 3, 0, 2, 1, 3, 2, 0,
    2, 1, 0, 3, 2, 0, 3, 1, 2, 3, 1, 0, 2, 3, 0, 1,
    3, 1, 2, 0, 3, 2, 0, 1, 3, 0, 1, 2, 3, 0, 2, 1,
];
const CN_CCNT: [f32; 16] = [
    1.34375, 1.28125, 1.359375, 1.3671875,
    1.4296875, 1.3984375, 1.3828125, 1.3046875,
    1.4140625, 1.2734375, 1.2578125, 1.2890625,
    1.3203125, 1.3515625, 1.3359375, 1.4609375,
];

// ---------------------------------------------------------------------------
// AES tables (cn2)
// ---------------------------------------------------------------------------

include!("aes_tables.rs");

fn aes_round(x: [u32; 4], mut key: [u32; 4]) -> [u32; 4] {
    let byte = |v: u32, n: u32| ((v >> (n * 8)) & 0xFF) as usize;
    let a0 = |b: usize| AES0_C[b];
    let a1 = |b: usize| AES0_C[b].rotate_left(8);
    let a2 = |b: usize| AES0_C[b].rotate_left(16);
    let a3 = |b: usize| AES0_C[b].rotate_left(24);

    key[0] ^= a0(byte(x[0], 0)); key[1] ^= a0(byte(x[1], 0)); key[2] ^= a0(byte(x[2], 0)); key[3] ^= a0(byte(x[3], 0));
    key[0] ^= a2(byte(x[2], 2)); key[1] ^= a2(byte(x[3], 2)); key[2] ^= a2(byte(x[0], 2)); key[3] ^= a2(byte(x[1], 2));
    key[0] ^= a1(byte(x[1], 1)); key[1] ^= a1(byte(x[2], 1)); key[2] ^= a1(byte(x[3], 1)); key[3] ^= a1(byte(x[0], 1));
    key[0] ^= a3(byte(x[3], 3)); key[1] ^= a3(byte(x[0], 3)); key[2] ^= a3(byte(x[1], 3)); key[3] ^= a3(byte(x[2], 3));
    key
}

fn aes10(mut x: [u32; 4], key: &[u32; 40]) -> [u32; 4] {
    for j in 0..10 {
        x = aes_round(x, [key[4 * j], key[4 * j + 1], key[4 * j + 2], key[4 * j + 3]]);
    }
    x
}

fn subword(w: u32) -> u32 {
    let s = |n: u32| AES_SBOX[((w >> (n * 8)) & 0xFF) as usize] as u32;
    (s(3) << 24) | (s(2) << 16) | (s(1) << 8) | s(0)
}

fn aes_expand_key256(k: &mut [u32; 40]) {
    let mut i = 1usize;
    for c in 8..40 {
        let mut t = if c % 8 == 0 || c % 8 == 4 { subword(k[c - 1]) } else { k[c - 1] };
        if c % 8 == 0 {
            t = t.rotate_left(24) ^ AES_RCON[i];
            i += 1;
        }
        k[c] = k[c - 8] ^ t;
    }
}

#[inline]
fn xor4(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
}
#[inline]
fn rd_uvec4(sp: &[u32], i: usize) -> [u32; 4] {
    [sp[4 * i], sp[4 * i + 1], sp[4 * i + 2], sp[4 * i + 3]]
}

// ---------------------------------------------------------------------------
// Full hash
// ---------------------------------------------------------------------------

pub fn state_from_input(input: &[u8; 128], nonce: u64) -> [u64; 25] {
    let mut st = [0u64; 25];
    for i in 0..8 {
        st[i] = u64::from_le_bytes(input[i * 8..i * 8 + 8].try_into().unwrap());
    }
    st[8] = bswap64(nonce);
    st[9] = u64::from_le_bytes(input[72..80].try_into().unwrap());
    st[10] = u64::from_le_bytes(input[80..88].try_into().unwrap());
    st[16] = 0x8000000000000000;
    keccakf(&mut st);
    st
}

fn explode(state: &[u64; 25], scratch: &mut [u32]) {
    let blocks = MEMORY / 512;
    let write = |scratch: &mut [u32], u64idx: usize, val: u64| {
        scratch[2 * u64idx] = val as u32;
        scratch[2 * u64idx + 1] = (val >> 32) as u32;
    };
    for idx in 0..blocks {
        let mut hash = [0u64; 25];
        hash[0] = state[0] ^ (idx as u64);
        for i in 1..25 {
            hash[i] = state[i];
        }
        let mut o = idx * 64;
        keccakf(&mut hash);
        for i in 0..20 { write(scratch, o + i, hash[i]); }
        o += 20;
        keccakf(&mut hash);
        for i in 0..22 { write(scratch, o + i, hash[i]); }
        o += 22;
        keccakf(&mut hash);
        for i in 0..22 { write(scratch, o + i, hash[i]); }
    }
}

fn cn1_loop(state0: u64, scratch: &mut [u32]) {
    let mut s: u32 = (state0 as u32) >> 8;
    let mut vs = [0f32; 4];
    let mut sout = [0i32; 64];
    let mut sva = [0f32; 64];

    for _ in 0..ITERATIONS {
        // substep 0: read tmp, fill linear 0..15
        let mut tmp = [0i32; 16];
        for tid in 0..16 {
            let tidd = tid / 4;
            let tidm = tid % 4;
            let sp = ((s as usize & MASK) >> 2) + tidd * 4 + tidm;
            tmp[tid] = scratch[sp] as i32;
            sout[tid] = tmp[tid];
        }
        // substep 1: single_compute per lane (read snapshot, write slots)
        let snap = sout;
        let mut nout = sout;
        let mut nva = sva;
        for tid in 0..16 {
            let tidm = (tid % 4) as u32;
            let rd = |j: usize| -> I4 {
                let b = 4 * j;
                [snap[b], snap[b + 1], snap[b + 2], snap[b + 3]]
            };
            let (va, out) = single_compute_wrap(
                tidm,
                rd(CN_LOOK[tid * 4]),
                rd(CN_LOOK[tid * 4 + 1]),
                rd(CN_LOOK[tid * 4 + 2]),
                rd(CN_LOOK[tid * 4 + 3]),
                CN_CCNT[tid],
                vs,
            );
            let b = 4 * tid;
            nout[b] = out[0]; nout[b + 1] = out[1]; nout[b + 2] = out[2]; nout[b + 3] = out[3];
            nva[b] = va[0]; nva[b + 1] = va[1]; nva[b + 2] = va[2]; nva[b + 3] = va[3];
        }
        sout = nout;
        sva = nva;
        // substep 2: outXor, scratch write, linear sout/sva writes
        let snap_out = sout;
        let snap_va = sva;
        let mut nout = sout;
        let mut nva = sva;
        for tid in 0..16 {
            let tidd = tid / 4;
            let tidm = tid % 4;
            let block = tidd * 16 + tidm;
            let mut out_xor = snap_out[block];
            let mut dd = block + 4;
            while dd < (tidd + 1) * 16 {
                out_xor ^= snap_out[dd];
                dd += 4;
            }
            let sp = ((s as usize & MASK) >> 2) + tidd * 4 + tidm;
            scratch[sp] = (out_xor ^ tmp[tid]) as u32;
            nout[tid] = out_xor;
            let va_tmp1 = snap_va[block] + snap_va[block + 4];
            let va_tmp2 = snap_va[block + 8] + snap_va[block + 12];
            nva[tid] = va_tmp1 + va_tmp2;
        }
        sout = nout;
        sva = nva;
        // substep 3
        let snap_out = sout;
        let snap_va = sva;
        let mut nout = sout;
        let mut nva = sva;
        for tid in 0..16 {
            let tidd = tid / 4;
            let tidm = tid % 4;
            let block = tidd * 16 + tidm;
            let out2 = snap_out[tid] ^ snap_out[tid + 4] ^ snap_out[tid + 8] ^ snap_out[tid + 12];
            let mut va_tmp1 = snap_va[block] + snap_va[block + 4];
            let va_tmp2 = snap_va[block + 8] + snap_va[block + 12];
            va_tmp1 = va_tmp1 + va_tmp2;
            va_tmp1 = va_tmp1.abs();
            let xx = va_tmp1 * 16777216.0f32;
            let xx_int = xx as i32;
            nout[tid] = out2 ^ xx_int;
            nva[tid] = va_tmp1 / 64.0f32;
        }
        sout = nout;
        sva = nva;

        vs = [sva[0], sva[1], sva[2], sva[3]];
        s = (sout[0] ^ sout[1] ^ sout[2] ^ sout[3]) as u32;
    }
}

fn cn2_finalize(state: &mut [u64; 25], scratch: &[u32]) -> u64 {
    let mut text = [[0u32; 4]; 8];
    for l in 0..8 {
        let u = (l + 4) * 2;
        text[l] = [
            state[u] as u32,
            (state[u] >> 32) as u32,
            state[u + 1] as u32,
            (state[u + 1] >> 32) as u32,
        ];
    }
    let mut key = [0u32; 40];
    for j in 0..8 {
        let v = state[4 + j / 2];
        key[j] = if j % 2 == 0 { v as u32 } else { (v >> 32) as u32 };
    }
    aes_expand_key256(&mut key);

    let mut xin1 = [[0u32; 4]; 8];
    let mut xin2 = [[0u32; 4]; 8];
    let sp_stride = MEMORY / 16;
    let mut i1 = [0usize; 8];
    for l in 0..8 {
        i1[l] = l;
    }

    let loops = MEMORY >> 7;
    for _ in 0..loops {
        for l in 0..8 { text[l] = xor4(text[l], rd_uvec4(scratch, i1[l])); }
        for l in 0..8 { text[l] = xor4(text[l], xin2[(l + 1) % 8]); }
        for l in 0..8 { text[l] = aes10(text[l], &key); }
        for l in 0..8 { xin1[l] = text[l]; }
        for l in 0..8 { text[l] = xor4(text[l], rd_uvec4(scratch, i1[l] + 8)); }
        for l in 0..8 { text[l] = xor4(text[l], xin1[(l + 1) % 8]); }
        for l in 0..8 { text[l] = aes10(text[l], &key); }
        for l in 0..8 { xin2[l] = text[l]; }
        for l in 0..8 { i1[l] = (i1[l] + 16) % sp_stride; }
    }
    for l in 0..8 { text[l] = xor4(text[l], xin2[(l + 1) % 8]); }

    for _ in 0..16 {
        for l in 0..8 { text[l] = aes10(text[l], &key); }
        for l in 0..8 { xin1[l] = text[l]; }
        for l in 0..8 { text[l] = xor4(text[l], xin1[(l + 1) % 8]); }
    }

    for l in 0..8 {
        let u = (l + 4) * 2;
        state[u] = (text[l][0] as u64) | ((text[l][1] as u64) << 32);
        state[u + 1] = (text[l][2] as u64) | ((text[l][3] as u64) << 32);
    }
    keccakf(state);
    bswap64(state[0])
}

/// Compute the FusionHash PoW value (byte-reversed first Keccak word) for the
/// given 128-byte input blob and 64-bit nonce. Lower is better; a share is
/// valid when this value is `<= target`.
pub fn fusion_hash(input: &[u8; 128], nonce: u64) -> u64 {
    let mut state = state_from_input(input, nonce);
    let mut scratch = vec![0u32; MEMORY / 4];
    explode(&state, &mut scratch);
    cn1_loop(state[0], &mut scratch);
    cn2_finalize(&mut state, &scratch)
}

/// Per-stage intermediates for the self-test.
pub struct Stages {
    pub state: [u64; 25],
    pub scratch_after_explode: Vec<u32>,
    pub scratch_after_cn1: Vec<u32>,
    #[allow(dead_code)]
    pub pv: u64,
}

pub fn debug_pipeline(input: &[u8; 128], nonce: u64) -> Stages {
    let state = state_from_input(input, nonce);
    let mut scratch = vec![0u32; MEMORY / 4];
    explode(&state, &mut scratch);
    let scratch_after_explode = scratch.clone();
    cn1_loop(state[0], &mut scratch);
    let scratch_after_cn1 = scratch.clone();
    let mut st2 = state;
    let pv = cn2_finalize(&mut st2, &scratch);
    Stages {
        state,
        scratch_after_explode,
        scratch_after_cn1,
        pv,
    }
}
