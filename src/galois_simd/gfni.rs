// `apply_scalar` / `affine_apply` are exercised by the round-trip test in
// `galois_simd::tests` (always built) and by the SIMD kernel's scalar tail
// loop (x86_64 only). On non-x86 targets the SIMD function isn't compiled,
// so these helpers look dead in a non-test lib build; allow it cleanly.
#![cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]

//! GFNI (GF2P8AFFINEQB) Galois-field kernel — affine-matrix derivation
//! and 128-bit SIMD multiply-XOR.
//!
//! For a fixed GF(2^16) coefficient `c`, this module derives the four
//! 8x8 GF(2) sub-matrices that together express multiplication by `c`
//! on a 16-bit symbol split into `(low_byte, high_byte)`. The SIMD
//! kernel (Task 2) evaluates these matrices with `VGF2P8AFFINEQB`.
//!
//! ## Intel GF2P8AFFINEQB convention (Intel SDM Vol 2A)
//!
//! Given an 8x8 GF(2) matrix `M` packed into a u64 and an input byte
//! `x`, GF2P8AFFINEQB computes per output bit `i`:
//!
//!     y[i] = parity(qword.byte[7 - i] AND x) XOR imm[i]
//!
//! with `imm = 0` for our use. The packing is:
//!   - row `i` of the matrix lives in qword byte `7 - i` (byte 0 holds
//!     row 7; byte 7 holds row 0).
//!   - within that byte, matrix entry `M[i][k]` sits at bit `k`
//!     (LSB-first within the byte; no reversal).
//!   - input bit `k` of `x` is at bit `k` of `x` (LSB-first).
//!
//! Both operands enter `parity` symmetrically, so there is no MSB/LSB
//! asymmetry inside the byte — only the row-byte index is reversed.
//!
//! ## Decomposition into four 8x8 blocks
//!
//! Multiplication by `c` is a 16x16 GF(2) linear map on the input
//! vector `x = (x_lo, x_hi)`. Column `j` of that map is `c * (1 << j)`
//! computed in GF(2^16). We then carve into:
//!
//!   | Block    | Output bits | Input bits |
//!   |----------|-------------|------------|
//!   | mat_ll   | 0..8 (lo)   | 0..8 (lo)  |
//!   | mat_lh   | 0..8 (lo)   | 8..16 (hi) |
//!   | mat_hl   | 8..16 (hi)  | 0..8 (lo)  |
//!   | mat_hh   | 8..16 (hi)  | 8..16 (hi) |
//!
//! and at apply time:
//!   out_lo = M_ll · x_lo  XOR  M_lh · x_hi
//!   out_hi = M_hl · x_lo  XOR  M_hh · x_hi

use crate::galois::gf_mul;

/// Per-coefficient affine matrices. Held by the `CoeffSimdTables::Gfni`
/// variant; the public enum forces this type to be `pub`. Fields stay
/// `pub(in crate::galois_simd)` so only the SIMD kernel and the round-trip
/// test reach into the raw u64 packing.
#[derive(Clone, Copy, Debug)]
pub struct GfniTables {
    /// Low output byte from low input byte (rows 0..8, cols 0..8).
    pub(in crate::galois_simd) mat_ll: u64,
    /// Low output byte from high input byte (rows 0..8, cols 8..16).
    pub(in crate::galois_simd) mat_lh: u64,
    /// High output byte from low input byte (rows 8..16, cols 0..8).
    pub(in crate::galois_simd) mat_hl: u64,
    /// High output byte from high input byte (rows 8..16, cols 8..16).
    pub(in crate::galois_simd) mat_hh: u64,
}

impl GfniTables {
    /// Derive the four affine matrices for multiplication by `coeff` in
    /// GF(2^16). Cost: 16 scalar GF multiplies + bit-shuffling — well
    /// under 1 µs on any modern CPU. Callers must filter `coeff in {0, 1}`
    /// before invoking; those cases are handled inline by the dispatch
    /// (Task 2) and would produce a trivial matrix here.
    pub(super) fn from_coeff(coeff: u16) -> Self {
        // Column j of the 16x16 GF(2) matrix is the GF(2^16) product of
        // `coeff` and the basis vector `e_j = 1 << j`. The low 8 bits of
        // that 16-bit result populate output rows 0..8 (the low byte of
        // the result) and the high 8 bits populate rows 8..16.
        let mut cols = [0u16; 16];
        for j in 0..16u32 {
            cols[j as usize] = gf_mul(coeff, 1u16 << j);
        }

        // Pack one 8x8 sub-block.
        //
        // For each row `i` (output bit `out_bit_base + i`), build a byte
        // whose bit at position `ip` (LSB-first per Intel) is the matrix
        // entry M[i][ip] = (cols[in_bit_base + ip] >> (out_bit_base + i)) & 1.
        //
        // Row `i` is placed at byte position `7 - i` of the returned u64
        // (Intel: qword.byte[7 - i] holds row i).
        let pack_block = |out_bit_base: u32, in_bit_base: u32| -> u64 {
            let mut acc = 0u64;
            for i in 0..8u32 {
                let mut row: u8 = 0;
                for ip in 0..8u32 {
                    let col = in_bit_base + ip;
                    let bit = (cols[col as usize] >> (out_bit_base + i)) & 1;
                    // LSB-first within the row byte: M[i][ip] at bit `ip`.
                    row |= (bit as u8) << ip;
                }
                // Row i goes to qword byte (7 - i).
                acc |= (row as u64) << ((7 - i) * 8);
            }
            acc
        };

        Self {
            mat_ll: pack_block(0, 0),
            mat_lh: pack_block(0, 8),
            mat_hl: pack_block(8, 0),
            mat_hh: pack_block(8, 8),
        }
    }

    /// Software reference for one symbol through the four matrices. Used
    /// (a) by the round-trip test and (b) by the SIMD scalar tail in
    /// Task 2. Hot path is `gf_mul_xor_gfni` (Task 2); this is the
    /// correctness anchor.
    pub(super) fn apply_scalar(&self, lo: u8, hi: u8) -> (u8, u8) {
        let out_lo = affine_apply(self.mat_ll, lo) ^ affine_apply(self.mat_lh, hi);
        let out_hi = affine_apply(self.mat_hl, lo) ^ affine_apply(self.mat_hh, hi);
        (out_lo, out_hi)
    }
}

/// Software emulation of one GF2P8AFFINEQB byte lane:
///   y[i] = parity(qword.byte[7 - i] AND x)
fn affine_apply(mat: u64, x: u8) -> u8 {
    let mut y: u8 = 0;
    for i in 0..8u32 {
        let row = ((mat >> ((7 - i) * 8)) & 0xFF) as u8;
        let p = ((row & x).count_ones() & 1) as u8;
        y |= p << i;
    }
    y
}

// --------------------------------------------------------------------------
// 128-bit GFNI multiply-XOR (x86_64 only)
// --------------------------------------------------------------------------
//
// MVP width is 128-bit / 16 symbols per iter, matching the existing SSSE3
// path's data layout exactly. The win over SSSE3 comes from replacing the
// 8× `PSHUFB` nibble lookups + 6× `XOR` per iter with 4× `GF2P8AFFINEQB`
// + 2× `XOR` — typically ~1.5–2× per-iter speedup on Tremont, Gracemont,
// and any GFNI part. Wider AVX2/AVX-512 variants are a follow-up; the
// 128-bit body is the lowest-risk introduction because every shuffle
// stays within a single 16-byte lane.

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::GfniTables;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// `output ^= coeff · input` using GFNI + SSSE3.
    ///
    /// Processes 32 bytes (16 GF(2^16) symbols) per iteration. Tail
    /// bytes (< 32) go through the scalar `apply_scalar` fall-through
    /// so behaviour is byte-identical to the scalar reference at every
    /// length.
    ///
    /// # Safety
    /// Caller must ensure `gfni` and `ssse3` are runtime-available. The
    /// dispatch enum in `galois_simd` enforces this via `detect_dispatch`.
    /// `input.len() == output.len()` and `input.len() % 2 == 0` are
    /// asserted in debug builds.
    #[target_feature(enable = "gfni,ssse3")]
    pub(in crate::galois_simd) unsafe fn gf_mul_xor_gfni(
        t: &GfniTables,
        input: &[u8],
        output: &mut [u8],
    ) {
        debug_assert_eq!(input.len(), output.len());
        debug_assert_eq!(input.len() % 2, 0);

        // Broadcast each 8-byte affine matrix to both qword lanes of an
        // XMM. `_mm_gf2p8affine_epi64_epi8` applies the matrix in the
        // corresponding qword of `b` to each byte of the corresponding
        // qword of `a`, so broadcasting lets the whole 16-byte vector
        // share the same matrix.
        let m_ll = _mm_set1_epi64x(t.mat_ll as i64);
        let m_lh = _mm_set1_epi64x(t.mat_lh as i64);
        let m_hl = _mm_set1_epi64x(t.mat_hl as i64);
        let m_hh = _mm_set1_epi64x(t.mat_hh as i64);

        // Deinterleave: pack the even-indexed (low) bytes of one 16-byte
        // input vector into the low 8 lanes, then `_mm_unpacklo_epi64`
        // with the same trick on the second 16-byte vector to assemble
        // all 16 low bytes into one XMM. Same for odd-indexed (high)
        // bytes. Mirrors the SSSE3 path so the layout cost is identical
        // — the win is purely in the affine vs nibble-shuffle math.
        let lo_idx = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0);
        let hi_idx = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 15, 13, 11, 9, 7, 5, 3, 1);

        let chunks = input.len() / 32;
        for c in 0..chunks {
            let off = c * 32;
            let v0 = _mm_loadu_si128(input.as_ptr().add(off) as *const __m128i);
            let v1 = _mm_loadu_si128(input.as_ptr().add(off + 16) as *const __m128i);

            let lo_v0 = _mm_shuffle_epi8(v0, lo_idx);
            let lo_v1 = _mm_shuffle_epi8(v1, lo_idx);
            let low_bytes = _mm_unpacklo_epi64(lo_v0, lo_v1);

            let hi_v0 = _mm_shuffle_epi8(v0, hi_idx);
            let hi_v1 = _mm_shuffle_epi8(v1, hi_idx);
            let high_bytes = _mm_unpacklo_epi64(hi_v0, hi_v1);

            // out_lo = mat_ll · low_bytes  XOR  mat_lh · high_bytes
            // out_hi = mat_hl · low_bytes  XOR  mat_hh · high_bytes
            let out_lo_byte = _mm_xor_si128(
                _mm_gf2p8affine_epi64_epi8::<0>(low_bytes, m_ll),
                _mm_gf2p8affine_epi64_epi8::<0>(high_bytes, m_lh),
            );
            let out_hi_byte = _mm_xor_si128(
                _mm_gf2p8affine_epi64_epi8::<0>(low_bytes, m_hl),
                _mm_gf2p8affine_epi64_epi8::<0>(high_bytes, m_hh),
            );

            // Re-interleave product low/high bytes back into 32 output
            // bytes (one [lo,hi,lo,hi,...] sequence per 16-byte lane).
            let out_v0 = _mm_unpacklo_epi8(out_lo_byte, out_hi_byte);
            let out_v1 = _mm_unpackhi_epi8(out_lo_byte, out_hi_byte);
            let dst_v0 = _mm_loadu_si128(output.as_ptr().add(off) as *const __m128i);
            let dst_v1 = _mm_loadu_si128(output.as_ptr().add(off + 16) as *const __m128i);
            _mm_storeu_si128(
                output.as_mut_ptr().add(off) as *mut __m128i,
                _mm_xor_si128(dst_v0, out_v0),
            );
            _mm_storeu_si128(
                output.as_mut_ptr().add(off + 16) as *mut __m128i,
                _mm_xor_si128(dst_v1, out_v1),
            );
        }

        // Scalar tail (< 32 bytes): identical math via the affine matrix
        // emulation, keeping the result byte-for-byte equal to the
        // scalar reference even at sub-chunk lengths.
        let consumed = chunks * 32;
        let mut off = consumed;
        while off + 2 <= input.len() {
            let (o_lo, o_hi) = t.apply_scalar(input[off], input[off + 1]);
            output[off] ^= o_lo;
            output[off + 1] ^= o_hi;
            off += 2;
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) use simd::gf_mul_xor_gfni;
