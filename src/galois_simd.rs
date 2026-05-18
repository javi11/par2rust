//! SIMD GF(2^16) multiply-and-XOR paths.
//!
//! The hot loop in PAR2 encode is, conceptually:
//!
//! ```text
//! for each (input_block, recovery_block) pair:
//!     recovery_buf ^= coeff * input_buf      (element-wise over u16 symbols)
//! ```
//!
//! Three layers, all producing bit-identical results:
//!
//!   1. `gf_mul_xor_scalar` (in `reedsolomon.rs`): per-symbol log/antilog. The
//!      correctness reference and the build-everywhere fallback.
//!   2. `gf_mul_xor_table_scalar`: precomputes two 256-entry u16 lookup tables
//!      per coefficient (`L[s_lo]` and `H[s_hi]`) so the inner loop is two
//!      loads + one XOR per symbol. ~5–10× faster than (1) with no intrinsics.
//!   3. `gf_mul_xor_neon` / `gf_mul_xor_ssse3`: nibble-table variants that do
//!      16 lookups per `vqtbl1q_u8` / `pshufb`. Each call processes 16 bytes
//!      (8 GF symbols) per loop iteration.
//!
//! `gf_mul_xor_dispatch` picks the fastest available path at runtime.

use crate::galois::gf_mul;
use crate::reedsolomon::gf_mul_xor_scalar;

/// Precomputed per-coefficient lookup tables for the byte-split fast path.
///
/// For a fixed `coeff`:
///   - `lo[s]` = `coeff · (s as low-byte-symbol)` — i.e. the symbol whose low
///     byte is `s` and whose high byte is 0, multiplied by `coeff`.
///   - `hi[s]` = `coeff · ((s as u16) << 8)` — symbol with low byte 0, high
///     byte `s`, multiplied by `coeff`.
///
/// Then for any input symbol `sym = (sym_hi << 8) | sym_lo`:
///     coeff · sym = lo[sym_lo] XOR hi[sym_hi]
///
/// This works because GF(2^16) multiplication is bilinear and `sym` decomposes
/// into a XOR of low- and high-byte contributions over GF(2).
pub struct CoeffTables {
    pub lo: [u16; 256],
    pub hi: [u16; 256],
}

impl CoeffTables {
    pub fn new(coeff: u16) -> Self {
        let mut lo = [0u16; 256];
        let mut hi = [0u16; 256];
        for s in 0..256u16 {
            lo[s as usize] = gf_mul(coeff, s);
            hi[s as usize] = gf_mul(coeff, s << 8);
        }
        CoeffTables { lo, hi }
    }
}

/// Byte-split-table scalar path: `output ^= coeff · input`, two lookups per
/// symbol. Same observable result as `gf_mul_xor_scalar` but avoids the
/// per-symbol log addition and modular reduction.
pub fn gf_mul_xor_table_scalar(tables: &CoeffTables, input: &[u8], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % 2, 0);
    for k in 0..(input.len() / 2) {
        let lo = tables.lo[input[2 * k] as usize];
        let hi = tables.hi[input[2 * k + 1] as usize];
        let prod = lo ^ hi;
        output[2 * k] ^= prod as u8;
        output[2 * k + 1] ^= (prod >> 8) as u8;
    }
}

// --------------------------------------------------------------------------
// aarch64 NEON path
// --------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::CoeffTables;

    /// Nibble-split lookup tables used by the NEON path.
    ///
    /// For a coefficient `c`, we need to compute, for any byte `b`, the 16-bit
    /// product `c · b_as_low_byte` and `c · b_as_high_byte`. We split each byte
    /// into nibbles (b = (b_hi << 4) | b_lo) and precompute eight 16-byte
    /// tables — one for each combination of {input byte position, nibble
    /// position, output byte}.
    ///
    /// Memory layout (all `[u8; 16]`, the natural width of a NEON byte vector):
    ///   - `lo_lo`: low byte of `c · (n)` for n=0..15 (input byte is low byte)
    ///   - `lo_hi`: high byte of `c · (n)`            (input byte is low byte)
    ///   - `lo_lo_n16`: low byte of `c · (n << 4)`    (input byte is low byte, high nibble)
    ///   - `lo_hi_n16`: high byte of `c · (n << 4)`   ditto
    ///   - `hi_lo`: low byte of `c · (n << 8)`        (input byte is high byte)
    ///   - `hi_hi`: high byte of `c · (n << 8)`       ditto
    ///   - `hi_lo_n16`: low byte of `c · (n << 12)`   (input byte is high byte, high nibble)
    ///   - `hi_hi_n16`: high byte of `c · (n << 12)`  ditto
    pub struct NeonTables {
        pub lo_lo: [u8; 16],
        pub lo_hi: [u8; 16],
        pub lo_lo_n16: [u8; 16],
        pub lo_hi_n16: [u8; 16],
        pub hi_lo: [u8; 16],
        pub hi_hi: [u8; 16],
        pub hi_lo_n16: [u8; 16],
        pub hi_hi_n16: [u8; 16],
    }

    impl NeonTables {
        pub fn from_coeff_tables(byte_tables: &CoeffTables) -> Self {
            let mut t = NeonTables {
                lo_lo: [0; 16],
                lo_hi: [0; 16],
                lo_lo_n16: [0; 16],
                lo_hi_n16: [0; 16],
                hi_lo: [0; 16],
                hi_hi: [0; 16],
                hi_lo_n16: [0; 16],
                hi_hi_n16: [0; 16],
            };
            for n in 0..16usize {
                let low_only = byte_tables.lo[n]; // c · n        (low-byte input)
                let low_n16 = byte_tables.lo[n << 4]; // c · (n<<4)
                let high_only = byte_tables.hi[n]; // c · (n << 8)
                let high_n16 = byte_tables.hi[n << 4]; // c · (n<<12)

                t.lo_lo[n] = low_only as u8;
                t.lo_hi[n] = (low_only >> 8) as u8;
                t.lo_lo_n16[n] = low_n16 as u8;
                t.lo_hi_n16[n] = (low_n16 >> 8) as u8;
                t.hi_lo[n] = high_only as u8;
                t.hi_hi[n] = (high_only >> 8) as u8;
                t.hi_lo_n16[n] = high_n16 as u8;
                t.hi_hi_n16[n] = (high_n16 >> 8) as u8;
            }
            t
        }
    }

    /// NEON multiply-XOR. Processes the buffer 16 bytes at a time (8 GF symbols
    /// per iteration); the trailing < 16 bytes are handled by the byte-table
    /// scalar fallback so any (even) length is supported.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available. On aarch64 this is part of the
    /// base ISA so it is always available; the `#[target_feature]` annotation
    /// is for documentation and to allow the compiler to assume NEON
    /// instructions are legal here.
    #[target_feature(enable = "neon")]
    pub unsafe fn gf_mul_xor_neon(tables: &NeonTables, input: &[u8], output: &mut [u8]) {
        debug_assert_eq!(input.len(), output.len());
        debug_assert_eq!(input.len() % 2, 0);

        // SAFETY: each `vld1q_u8` reads exactly 16 bytes from a `[u8; 16]`
        // borrow — alignment is irrelevant for NEON byte loads. The tables are
        // borrowed for the duration of this function so the pointers remain
        // valid.
        let t_lo_lo = vld1q_u8(tables.lo_lo.as_ptr());
        let t_lo_hi = vld1q_u8(tables.lo_hi.as_ptr());
        let t_lo_lo_n16 = vld1q_u8(tables.lo_lo_n16.as_ptr());
        let t_lo_hi_n16 = vld1q_u8(tables.lo_hi_n16.as_ptr());
        let t_hi_lo = vld1q_u8(tables.hi_lo.as_ptr());
        let t_hi_hi = vld1q_u8(tables.hi_hi.as_ptr());
        let t_hi_lo_n16 = vld1q_u8(tables.hi_lo_n16.as_ptr());
        let t_hi_hi_n16 = vld1q_u8(tables.hi_hi_n16.as_ptr());

        let mask_low_nibble = vdupq_n_u8(0x0F);

        // Wide loop: 32 bytes (16 symbols) per iteration, with native
        // deinterleave via `vld2q_u8`. The trailing < 32 bytes are handled by
        // the byte-table scalar fallback below.
        let wide_chunks = input.len() / 32;
        let mut consumed = 0usize;
        for chunk in 0..wide_chunks {
            let off = chunk * 32;
            let pair = vld2q_u8(input.as_ptr().add(off));
            let low_bytes = pair.0; // 16 lanes, one byte from each of 16 symbols (low halves)
            let high_bytes = pair.1; // high halves

            // Split each byte into high and low nibbles
            let low_lo_nib = vandq_u8(low_bytes, mask_low_nibble);
            let low_hi_nib = vshrq_n_u8(low_bytes, 4);
            let high_lo_nib = vandq_u8(high_bytes, mask_low_nibble);
            let high_hi_nib = vshrq_n_u8(high_bytes, 4);

            // Look up each nibble in the appropriate table.
            let prod_lo_byte = veorq_u8(
                veorq_u8(
                    vqtbl1q_u8(t_lo_lo, low_lo_nib),
                    vqtbl1q_u8(t_lo_lo_n16, low_hi_nib),
                ),
                veorq_u8(
                    vqtbl1q_u8(t_hi_lo, high_lo_nib),
                    vqtbl1q_u8(t_hi_lo_n16, high_hi_nib),
                ),
            );
            let prod_hi_byte = veorq_u8(
                veorq_u8(
                    vqtbl1q_u8(t_lo_hi, low_lo_nib),
                    vqtbl1q_u8(t_lo_hi_n16, low_hi_nib),
                ),
                veorq_u8(
                    vqtbl1q_u8(t_hi_hi, high_lo_nib),
                    vqtbl1q_u8(t_hi_hi_n16, high_hi_nib),
                ),
            );

            // Interleave back: store 32 bytes as alternating low/high halves
            let existing = vld2q_u8(output.as_ptr().add(off));
            let new_lo = veorq_u8(existing.0, prod_lo_byte);
            let new_hi = veorq_u8(existing.1, prod_hi_byte);
            vst2q_u8(output.as_mut_ptr().add(off), uint8x16x2_t(new_lo, new_hi));
            consumed = off + 32;
        }

        // Trailing bytes (< 32) handled by the byte-table scalar path.
        if consumed < input.len() {
            super::gf_mul_xor_table_scalar_from_neon_tables(
                tables,
                &input[consumed..],
                &mut output[consumed..],
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use neon::{gf_mul_xor_neon, NeonTables};

/// Tail-handler bridge: when the NEON path can't fill its 32-byte window, fall
/// back to the byte-table scalar path. We reconstruct the byte tables from the
/// nibble tables to avoid carrying both representations through the encoder.
#[cfg(target_arch = "aarch64")]
fn gf_mul_xor_table_scalar_from_neon_tables(nt: &NeonTables, input: &[u8], output: &mut [u8]) {
    debug_assert_eq!(input.len() % 2, 0);
    for k in 0..(input.len() / 2) {
        let lb = input[2 * k];
        let hb = input[2 * k + 1];
        let lb_lo = (lb & 0xF) as usize;
        let lb_hi = (lb >> 4) as usize;
        let hb_lo = (hb & 0xF) as usize;
        let hb_hi = (hb >> 4) as usize;
        let prod_lo = nt.lo_lo[lb_lo] ^ nt.lo_lo_n16[lb_hi] ^ nt.hi_lo[hb_lo] ^ nt.hi_lo_n16[hb_hi];
        let prod_hi = nt.lo_hi[lb_lo] ^ nt.lo_hi_n16[lb_hi] ^ nt.hi_hi[hb_lo] ^ nt.hi_hi_n16[hb_hi];
        output[2 * k] ^= prod_lo;
        output[2 * k + 1] ^= prod_hi;
    }
}

// --------------------------------------------------------------------------
// x86_64 SSSE3 path
// --------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::CoeffTables;

    pub struct X86Tables {
        pub lo_lo: [u8; 16],
        pub lo_hi: [u8; 16],
        pub lo_lo_n16: [u8; 16],
        pub lo_hi_n16: [u8; 16],
        pub hi_lo: [u8; 16],
        pub hi_hi: [u8; 16],
        pub hi_lo_n16: [u8; 16],
        pub hi_hi_n16: [u8; 16],
    }

    impl X86Tables {
        pub fn from_coeff_tables(byte_tables: &CoeffTables) -> Self {
            let mut t = X86Tables {
                lo_lo: [0; 16],
                lo_hi: [0; 16],
                lo_lo_n16: [0; 16],
                lo_hi_n16: [0; 16],
                hi_lo: [0; 16],
                hi_hi: [0; 16],
                hi_lo_n16: [0; 16],
                hi_hi_n16: [0; 16],
            };
            for n in 0..16usize {
                let low_only = byte_tables.lo[n];
                let low_n16 = byte_tables.lo[n << 4];
                let high_only = byte_tables.hi[n];
                let high_n16 = byte_tables.hi[n << 4];

                t.lo_lo[n] = low_only as u8;
                t.lo_hi[n] = (low_only >> 8) as u8;
                t.lo_lo_n16[n] = low_n16 as u8;
                t.lo_hi_n16[n] = (low_n16 >> 8) as u8;
                t.hi_lo[n] = high_only as u8;
                t.hi_hi[n] = (high_only >> 8) as u8;
                t.hi_lo_n16[n] = high_n16 as u8;
                t.hi_hi_n16[n] = (high_n16 >> 8) as u8;
            }
            t
        }
    }

    /// SSSE3 multiply-XOR. Processes 32 bytes (16 symbols) per iteration using
    /// `pshufb` for nibble lookups. Caller must verify SSSE3 is available.
    ///
    /// # Safety
    /// SSSE3 must be enabled; buffers must be valid for `input.len()` bytes.
    #[target_feature(enable = "ssse3")]
    pub unsafe fn gf_mul_xor_ssse3(tables: &X86Tables, input: &[u8], output: &mut [u8]) {
        debug_assert_eq!(input.len(), output.len());
        debug_assert_eq!(input.len() % 2, 0);

        let t_lo_lo = _mm_loadu_si128(tables.lo_lo.as_ptr() as *const __m128i);
        let t_lo_hi = _mm_loadu_si128(tables.lo_hi.as_ptr() as *const __m128i);
        let t_lo_lo_n16 = _mm_loadu_si128(tables.lo_lo_n16.as_ptr() as *const __m128i);
        let t_lo_hi_n16 = _mm_loadu_si128(tables.lo_hi_n16.as_ptr() as *const __m128i);
        let t_hi_lo = _mm_loadu_si128(tables.hi_lo.as_ptr() as *const __m128i);
        let t_hi_hi = _mm_loadu_si128(tables.hi_hi.as_ptr() as *const __m128i);
        let t_hi_lo_n16 = _mm_loadu_si128(tables.hi_lo_n16.as_ptr() as *const __m128i);
        let t_hi_hi_n16 = _mm_loadu_si128(tables.hi_hi_n16.as_ptr() as *const __m128i);
        let mask_lo = _mm_set1_epi8(0x0F);

        // Permutation mask to deinterleave 32 bytes (interleaved low/high of
        // 16 symbols) into two 16-byte vectors of low bytes and high bytes.
        // SSSE3 has no `pshufb` across two registers, so we do two loads and
        // shuffle each, then combine. `_mm_set_epi8` writes its first argument
        // to the highest-indexed lane, so the meaningful indices must sit in
        // the low half here for `_mm_unpacklo_epi64` (below) to pick them up.
        let lo_idx = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0);
        let hi_idx = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 15, 13, 11, 9, 7, 5, 3, 1);

        let chunks = input.len() / 32;
        let mut consumed = 0usize;
        for c in 0..chunks {
            let off = c * 32;
            let v0 = _mm_loadu_si128(input.as_ptr().add(off) as *const __m128i);
            let v1 = _mm_loadu_si128(input.as_ptr().add(off + 16) as *const __m128i);

            // De-interleave: pack even-indexed bytes of each half into the low
            // half of `low_bytes`, then OR with the same trick on `v1` shifted
            // into the high lanes via `_mm_unpacklo_epi64`.
            let lo_v0 = _mm_shuffle_epi8(v0, lo_idx);
            let lo_v1 = _mm_shuffle_epi8(v1, lo_idx);
            let low_bytes = _mm_unpacklo_epi64(lo_v0, lo_v1);

            let hi_v0 = _mm_shuffle_epi8(v0, hi_idx);
            let hi_v1 = _mm_shuffle_epi8(v1, hi_idx);
            let high_bytes = _mm_unpacklo_epi64(hi_v0, hi_v1);

            let low_lo_nib = _mm_and_si128(low_bytes, mask_lo);
            let low_hi_nib = _mm_and_si128(_mm_srli_epi16(low_bytes, 4), mask_lo);
            let high_lo_nib = _mm_and_si128(high_bytes, mask_lo);
            let high_hi_nib = _mm_and_si128(_mm_srli_epi16(high_bytes, 4), mask_lo);

            let prod_lo_byte = _mm_xor_si128(
                _mm_xor_si128(
                    _mm_shuffle_epi8(t_lo_lo, low_lo_nib),
                    _mm_shuffle_epi8(t_lo_lo_n16, low_hi_nib),
                ),
                _mm_xor_si128(
                    _mm_shuffle_epi8(t_hi_lo, high_lo_nib),
                    _mm_shuffle_epi8(t_hi_lo_n16, high_hi_nib),
                ),
            );
            let prod_hi_byte = _mm_xor_si128(
                _mm_xor_si128(
                    _mm_shuffle_epi8(t_lo_hi, low_lo_nib),
                    _mm_shuffle_epi8(t_lo_hi_n16, low_hi_nib),
                ),
                _mm_xor_si128(
                    _mm_shuffle_epi8(t_hi_hi, high_lo_nib),
                    _mm_shuffle_epi8(t_hi_hi_n16, high_hi_nib),
                ),
            );

            // Re-interleave low/high product bytes back into 32 output bytes
            // and XOR into output.
            let out_v0 = _mm_unpacklo_epi8(prod_lo_byte, prod_hi_byte);
            let out_v1 = _mm_unpackhi_epi8(prod_lo_byte, prod_hi_byte);

            let existing0 = _mm_loadu_si128(output.as_ptr().add(off) as *const __m128i);
            let existing1 = _mm_loadu_si128(output.as_ptr().add(off + 16) as *const __m128i);
            _mm_storeu_si128(
                output.as_mut_ptr().add(off) as *mut __m128i,
                _mm_xor_si128(existing0, out_v0),
            );
            _mm_storeu_si128(
                output.as_mut_ptr().add(off + 16) as *mut __m128i,
                _mm_xor_si128(existing1, out_v1),
            );
            consumed = off + 32;
        }
        if consumed < input.len() {
            super::gf_mul_xor_table_scalar_from_x86_tables(
                tables,
                &input[consumed..],
                &mut output[consumed..],
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86::{gf_mul_xor_ssse3, X86Tables};

#[cfg(target_arch = "x86_64")]
fn gf_mul_xor_table_scalar_from_x86_tables(nt: &X86Tables, input: &[u8], output: &mut [u8]) {
    for k in 0..(input.len() / 2) {
        let lb = input[2 * k];
        let hb = input[2 * k + 1];
        let prod_lo = nt.lo_lo[(lb & 0xF) as usize]
            ^ nt.lo_lo_n16[(lb >> 4) as usize]
            ^ nt.hi_lo[(hb & 0xF) as usize]
            ^ nt.hi_lo_n16[(hb >> 4) as usize];
        let prod_hi = nt.lo_hi[(lb & 0xF) as usize]
            ^ nt.lo_hi_n16[(lb >> 4) as usize]
            ^ nt.hi_hi[(hb & 0xF) as usize]
            ^ nt.hi_hi_n16[(hb >> 4) as usize];
        output[2 * k] ^= prod_lo;
        output[2 * k + 1] ^= prod_hi;
    }
}

// --------------------------------------------------------------------------
// Runtime dispatch
// --------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Dispatch {
    Scalar,
    TableScalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Ssse3,
}

/// Select the fastest available SIMD path on this machine. Called once at
/// startup; the result feeds into a per-coefficient table builder.
pub fn detect_dispatch() -> Dispatch {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is part of the aarch64 base ISA; no feature check required.
        return Dispatch::Neon;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("ssse3") {
            return Dispatch::Ssse3;
        }
    }
    #[allow(unreachable_code)]
    Dispatch::TableScalar
}

/// Apply `output ^= coeff · input` using the best available path. This is the
/// safe public entrypoint that all SIMD `unsafe` work is funneled through.
pub fn gf_mul_xor_dispatch(dispatch: Dispatch, coeff: u16, input: &[u8], output: &mut [u8]) {
    let tables = CoeffSimdTables::new(dispatch, coeff);
    gf_mul_xor_with_tables(&tables, input, output);
}

/// Pre-built per-coefficient lookup tables for whichever dispatch path is in
/// use. Building one of these is the expensive part of an encode step — the
/// matrix-shaped hot loop in PAR2 reuses the same recovery-row coefficient
/// against many input blocks (and vice versa), so the encoder builds these
/// once per slice and reuses them across all recovery buffers in the
/// parallel inner loop. See `creator.rs::build_volume_file`.
pub enum CoeffSimdTables {
    /// `coeff ∈ {0, 1}` — handled inline, no table needed.
    Trivial(u16),
    /// Scalar log/antilog path. Stores only the coefficient.
    Scalar(u16),
    /// Byte-split (256-entry) scalar fallback used on x86_64 without SSSE3.
    /// Boxed because `CoeffTables` is ~1 KiB while the SIMD variants are
    /// 128 B — keeping it inline would bloat the whole enum and the
    /// per-slice `Vec<CoeffSimdTables>` the encoder allocates.
    TableScalar(Box<CoeffTables>),
    #[cfg(target_arch = "aarch64")]
    Neon(neon::NeonTables),
    #[cfg(target_arch = "x86_64")]
    Ssse3(x86::X86Tables),
}

impl CoeffSimdTables {
    /// Build the per-coefficient lookup tables that the selected dispatch path
    /// needs. Trivial coefficients (0 and 1) skip table construction entirely.
    pub fn new(dispatch: Dispatch, coeff: u16) -> Self {
        if coeff == 0 || coeff == 1 {
            return CoeffSimdTables::Trivial(coeff);
        }
        match dispatch {
            Dispatch::Scalar => CoeffSimdTables::Scalar(coeff),
            Dispatch::TableScalar => {
                CoeffSimdTables::TableScalar(Box::new(CoeffTables::new(coeff)))
            }
            #[cfg(target_arch = "aarch64")]
            Dispatch::Neon => {
                let byte_tables = CoeffTables::new(coeff);
                CoeffSimdTables::Neon(neon::NeonTables::from_coeff_tables(&byte_tables))
            }
            #[cfg(target_arch = "x86_64")]
            Dispatch::Ssse3 => {
                let byte_tables = CoeffTables::new(coeff);
                CoeffSimdTables::Ssse3(x86::X86Tables::from_coeff_tables(&byte_tables))
            }
        }
    }
}

/// Apply `output ^= coeff · input` using a pre-built dispatch table. This is
/// the SIMD-only hot-path entrypoint — it does no table construction, so the
/// encoder loop is pure SIMD work.
pub fn gf_mul_xor_with_tables(tables: &CoeffSimdTables, input: &[u8], output: &mut [u8]) {
    match tables {
        CoeffSimdTables::Trivial(0) => {}
        CoeffSimdTables::Trivial(1) => {
            for (o, i) in output.iter_mut().zip(input.iter()) {
                *o ^= *i;
            }
        }
        CoeffSimdTables::Trivial(_) => unreachable!("Trivial holds only 0 or 1"),
        CoeffSimdTables::Scalar(coeff) => gf_mul_xor_scalar(*coeff, input, output),
        CoeffSimdTables::TableScalar(t) => gf_mul_xor_table_scalar(t, input, output),
        #[cfg(target_arch = "aarch64")]
        CoeffSimdTables::Neon(t) => {
            // SAFETY: NEON is part of the aarch64 base ISA. Buffer lengths are
            // validated by debug assertions in the callee.
            unsafe { neon::gf_mul_xor_neon(t, input, output) };
        }
        #[cfg(target_arch = "x86_64")]
        CoeffSimdTables::Ssse3(t) => {
            // SAFETY: a Ssse3 variant is only constructed via `CoeffSimdTables::new`
            // when the caller passed `Dispatch::Ssse3`, which itself only comes
            // from `detect_dispatch()` after a runtime SSSE3 check.
            unsafe { x86::gf_mul_xor_ssse3(t, input, output) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.push((state >> 33) as u8);
        }
        out
    }

    fn check_against_scalar(coeff: u16, input: &[u8], dispatch: Dispatch) {
        let mut out_scalar = vec![0u8; input.len()];
        gf_mul_xor_scalar(coeff, input, &mut out_scalar);

        let mut out_other = vec![0u8; input.len()];
        gf_mul_xor_dispatch(dispatch, coeff, input, &mut out_other);

        assert_eq!(
            out_scalar,
            out_other,
            "dispatch {:?} diverged from scalar for coeff=0x{:04X}, len={}",
            dispatch,
            coeff,
            input.len(),
        );
    }

    #[test]
    fn table_scalar_matches_scalar_for_random_inputs() {
        let coeffs = [0x0001u16, 0x0002, 0x00FF, 0x1234, 0xABCD, 0xFFFF];
        let lengths = [2usize, 4, 16, 30, 32, 34, 64, 256, 4096];
        for &coeff in &coeffs {
            for &len in &lengths {
                let input = deterministic(coeff as u64 ^ len as u64, len);
                check_against_scalar(coeff, &input, Dispatch::TableScalar);
            }
        }
    }

    #[test]
    fn dispatch_zero_is_noop() {
        let input = deterministic(1, 32);
        let mut out = vec![0xAB; 32];
        gf_mul_xor_dispatch(detect_dispatch(), 0, &input, &mut out);
        assert!(out.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn dispatch_one_is_xor() {
        let input = deterministic(7, 64);
        let mut out_dispatch = vec![0x33; 64];
        let mut out_xor = vec![0x33; 64];
        gf_mul_xor_dispatch(detect_dispatch(), 1, &input, &mut out_dispatch);
        for (a, b) in out_xor.iter_mut().zip(input.iter()) {
            *a ^= *b;
        }
        assert_eq!(out_dispatch, out_xor);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_for_random_coeffs_and_lengths() {
        let coeffs = [
            0x0002u16, 0x00FF, 0x0100, 0x1234, 0x8000, 0xABCD, 0xFFFE, 0xFFFF,
        ];
        // Mix of clean multiples of 32 and trailing fragments.
        let lengths = [2usize, 4, 16, 30, 32, 34, 62, 64, 96, 100, 1024, 4096, 4098];
        for &coeff in &coeffs {
            for &len in &lengths {
                let input = deterministic((coeff as u64).rotate_left(13) ^ len as u64, len);
                check_against_scalar(coeff, &input, Dispatch::Neon);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn ssse3_matches_scalar_when_available() {
        if !std::is_x86_feature_detected!("ssse3") {
            return;
        }
        let coeffs = [0x0002u16, 0x1234, 0xABCD, 0xFFFF];
        let lengths = [32usize, 34, 64, 100, 4096];
        for &coeff in &coeffs {
            for &len in &lengths {
                let input = deterministic(coeff as u64 ^ len as u64, len);
                check_against_scalar(coeff, &input, Dispatch::Ssse3);
            }
        }
    }
}
