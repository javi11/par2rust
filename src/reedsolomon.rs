//! Reed-Solomon encoder for PAR2 over GF(2^16).
//!
//! PAR2 stores its RS matrix in Vandermonde form. For an input block i and a
//! recovery block r:
//!
//! ```text
//! coefficient[r][i] = base[i] ^ exponent[r]
//! ```
//!
//! - `base[i]` is `α^logbase_i`, where `logbase_i` is the i-th non-negative
//!   integer coprime to `FIELD_LIMIT = 65535`. Coprimality guarantees that
//!   `base[i]` has multiplicative order 65535, so its powers stay distinct over
//!   the full exponent range — a necessary condition for the RS matrix to be
//!   invertible from any 32 768-row subset.
//! - `exponent[r]` is simply the consecutive integer assigned to recovery
//!   block r (typically `r` itself when `first_exponent = 0`).
//!
//! Each recovery slice is `slice_size` bytes; we treat consecutive pairs of
//! bytes as little-endian GF(2^16) symbols. PAR2 requires `slice_size % 4 == 0`,
//! so the byte count is always even.

use crate::galois::{gf_pow, gf16_tables, FIELD_LIMIT};

/// Builder/holder for a PAR2 RS encoding context.
pub struct RsEncoder {
    /// `base[i]` for each input block (a GF(2^16) element with order 65535).
    pub input_bases: Vec<u16>,
    /// Exponent for each recovery block (the leu32 stored in `RecoveryBlockPacket`).
    pub recovery_exponents: Vec<u16>,
}

impl RsEncoder {
    /// Build the input bases + recovery exponents for a fresh create job.
    ///
    /// `input_count` is the total number of source blocks (sum of slice counts
    /// across all input files). `first_exponent` is usually 0 (matches upstream
    /// default for `par2 c`); `recovery_count` is how many recovery blocks the
    /// user asked for.
    pub fn new(input_count: u32, first_exponent: u16, recovery_count: u32) -> Self {
        let input_bases = generate_input_bases(input_count as usize);
        let recovery_exponents = (0..recovery_count)
            .map(|r| {
                let e = first_exponent as u32 + r;
                debug_assert!(e <= u16::MAX as u32, "recovery exponent overflows u16");
                e as u16
            })
            .collect();
        RsEncoder { input_bases, recovery_exponents }
    }

    /// Look up the matrix coefficient for (recovery row, input column).
    #[inline]
    pub fn coefficient(&self, recovery_idx: usize, input_idx: usize) -> u16 {
        let base = self.input_bases[input_idx];
        let exp = self.recovery_exponents[recovery_idx] as u32;
        gf_pow(base, exp)
    }

    /// XOR `coeff * input` into `output`, treating the buffers as arrays of
    /// little-endian u16 GF symbols. Both buffers must have the same length and
    /// that length must be even.
    pub fn accumulate(
        &self,
        recovery_idx: usize,
        input_idx: usize,
        input: &[u8],
        output: &mut [u8],
    ) {
        assert_eq!(input.len(), output.len(), "buffer length mismatch");
        assert_eq!(input.len() % 2, 0, "buffer length must be even (16-bit symbols)");

        let coeff = self.coefficient(recovery_idx, input_idx);
        gf_mul_xor_scalar(coeff, input, output);
    }
}

/// Scalar GF(2^16) multiply-and-XOR: `output ^= coeff * input` over consecutive
/// little-endian u16 symbols. This is the correctness reference; SIMD paths in
/// `galois_simd.rs` must produce bit-identical output for any (coeff, input).
pub fn gf_mul_xor_scalar(coeff: u16, input: &[u8], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % 2, 0);

    // Fast path for the trivial coefficient values; both occur often (coeff=0
    // is meaningful when a recovery row touches an unused position, and coeff=1
    // happens at exponent=0).
    if coeff == 0 {
        return;
    }
    if coeff == 1 {
        for (o, i) in output.iter_mut().zip(input.iter()) {
            *o ^= *i;
        }
        return;
    }

    let t = gf16_tables();
    let log_coeff = t.log[coeff as usize] as u32;

    // Each iteration handles one GF(2^16) symbol == two little-endian bytes.
    for k in 0..(input.len() / 2) {
        let i_lo = input[2 * k] as u32;
        let i_hi = input[2 * k + 1] as u32;
        let sym = i_lo | (i_hi << 8);
        let product = if sym == 0 {
            0
        } else {
            let sum = log_coeff + t.log[sym as usize] as u32;
            let idx = if sum >= FIELD_LIMIT { sum - FIELD_LIMIT } else { sum };
            t.antilog[idx as usize]
        };
        output[2 * k] ^= product as u8;
        output[2 * k + 1] ^= (product >> 8) as u8;
    }
}

/// Generate `count` input bases whose discrete log is coprime to 65535. This is
/// the same walk that upstream's `ReedSolomon<Galois16>::SetInput` performs.
fn generate_input_bases(count: usize) -> Vec<u16> {
    let t = gf16_tables();
    let mut bases = Vec::with_capacity(count);
    let mut logbase: u32 = 0;
    while bases.len() < count {
        if logbase >= FIELD_LIMIT {
            // PAR2 caps input blocks at 32768 — long before this branch could fire
            // we'd already have hit `Par2Error::TooManyFiles` upstream.
            panic!("exhausted GF(2^16) bases coprime to {} (asked for {})", FIELD_LIMIT, count);
        }
        if gcd(FIELD_LIMIT, logbase) == 1 {
            bases.push(t.antilog[logbase as usize]);
        }
        logbase += 1;
    }
    bases
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::galois::gf_mul;

    #[test]
    fn first_input_base_has_logbase_one() {
        // gcd(65535, 0) = 65535 ≠ 1, gcd(65535, 1) = 1 → first base = α^1 = 2.
        let bases = generate_input_bases(3);
        assert_eq!(bases[0], 2);
    }

    #[test]
    fn input_bases_are_all_distinct() {
        let bases = generate_input_bases(1000);
        let mut sorted = bases.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), bases.len());
    }

    #[test]
    fn input_bases_skip_logs_sharing_factors_with_65535() {
        // 65535 = 3 · 5 · 17 · 257. So logbase=3, 5, 6, 9, 10, 15, 17, 25, 30...
        // should all be skipped.
        let t = gf16_tables();
        let bases = generate_input_bases(2);
        // First base = antilog[1] (logbase 1 — coprime).
        assert_eq!(bases[0], t.antilog[1]);
        // Second base = antilog[2] (logbase 2 — gcd(65535,2)=1).
        assert_eq!(bases[1], t.antilog[2]);

        // Generate a longer run and check none of the skipped logbases appear.
        let bases = generate_input_bases(20);
        let bases_as_logs: Vec<u16> = bases.iter().map(|b| t.log[*b as usize]).collect();
        for skip in [0u16, 3, 5, 6, 9, 10, 15, 17] {
            assert!(!bases_as_logs.contains(&skip), "logbase {skip} should be skipped");
        }
    }

    #[test]
    fn coefficient_matches_explicit_power() {
        let rs = RsEncoder::new(4, 0, 5);
        for r in 0..rs.recovery_exponents.len() {
            for i in 0..rs.input_bases.len() {
                let expected = gf_pow(rs.input_bases[i], rs.recovery_exponents[r] as u32);
                assert_eq!(rs.coefficient(r, i), expected);
            }
        }
    }

    #[test]
    fn coefficient_at_exponent_zero_is_one() {
        let rs = RsEncoder::new(3, 0, 1);
        // Recovery row 0 has exponent 0 → base^0 = 1 for every input.
        for i in 0..3 {
            assert_eq!(rs.coefficient(0, i), 1);
        }
    }

    #[test]
    fn accumulate_with_coeff_zero_is_noop() {
        let mut out = vec![0u8; 8];
        out[0] = 0x11;
        out[7] = 0x77;
        gf_mul_xor_scalar(0, &[0xFFu8; 8], &mut out);
        assert_eq!(out[0], 0x11);
        assert_eq!(out[7], 0x77);
        assert_eq!(&out[1..7], &[0u8; 6]);
    }

    #[test]
    fn accumulate_with_coeff_one_is_xor() {
        let input = [0x12, 0x34, 0x56, 0x78];
        let mut out = [0xAA, 0xBB, 0xCC, 0xDD];
        gf_mul_xor_scalar(1, &input, &mut out);
        assert_eq!(out, [0xAA ^ 0x12, 0xBB ^ 0x34, 0xCC ^ 0x56, 0xDD ^ 0x78]);
    }

    #[test]
    fn accumulate_matches_per_symbol_gf_mul() {
        // Build a small input, multiply by a non-trivial coefficient, then verify
        // each symbol is exactly gf_mul(coeff, in_sym) XORed into the output.
        let coeff: u16 = 0x1234;
        let input = vec![0x11, 0x22, 0xFF, 0x00, 0x99, 0xAB];
        let mut out = vec![0x10, 0x20, 0x30, 0x40, 0x50, 0x60];
        let mut expected = out.clone();

        for k in 0..3 {
            let s = input[2 * k] as u16 | ((input[2 * k + 1] as u16) << 8);
            let p = gf_mul(coeff, s);
            expected[2 * k] ^= p as u8;
            expected[2 * k + 1] ^= (p >> 8) as u8;
        }

        gf_mul_xor_scalar(coeff, &input, &mut out);
        assert_eq!(out, expected);
    }

    #[test]
    fn rs_encode_roundtrip_with_one_redundancy_block() {
        // With 2 input blocks and 1 recovery block, the encoder should produce
        // `out = coeff0·in0 ⊕ coeff1·in1` for the single recovery row.
        let rs = RsEncoder::new(2, 0, 1);
        let in0: Vec<u8> = (0..16u8).collect();
        let in1: Vec<u8> = (16..32u8).collect();

        let mut out = vec![0u8; 16];
        rs.accumulate(0, 0, &in0, &mut out);
        rs.accumulate(0, 1, &in1, &mut out);

        // Recovery row 0 has exponent 0 → coefficients are both 1 → out = in0 ⊕ in1.
        let expected: Vec<u8> = in0.iter().zip(in1.iter()).map(|(a, b)| a ^ b).collect();
        assert_eq!(out, expected);
    }
}
