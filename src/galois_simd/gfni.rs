// `from_coeff` / `apply_scalar` / `affine_apply` are exercised by the
// round-trip test in `galois_simd::tests` (always built) and consumed
// by the SIMD kernel + scalar tail loop landed in Task 2 of the GFNI
// plan. Until Task 2 lands, a release `--lib` build sees them as dead.
#![allow(dead_code)]

//! GFNI (GF2P8AFFINEQB) Galois-field kernel — affine-matrix derivation.
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

#[derive(Clone, Copy, Debug)]
pub(super) struct GfniTables {
    /// Low output byte from low input byte (rows 0..8, cols 0..8).
    pub(super) mat_ll: u64,
    /// Low output byte from high input byte (rows 0..8, cols 8..16).
    pub(super) mat_lh: u64,
    /// High output byte from low input byte (rows 8..16, cols 0..8).
    pub(super) mat_hl: u64,
    /// High output byte from high input byte (rows 8..16, cols 8..16).
    pub(super) mat_hh: u64,
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
