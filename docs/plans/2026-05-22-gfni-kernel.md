# GFNI (GF2P8AFFINEQB) Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GFNI + AVX-512 kernel to par2rust's Galois-field SIMD dispatch, slotting in alongside the existing SSSE3 and NEON paths. Roughly 2× the throughput of the SSSE3 PSHUFB path on Ice Lake / Sapphire Rapids / Zen 4+ hardware.

**Architecture:** GF(2^16) multiplication by a fixed coefficient `c` is a 16×16 GF(2) linear map. We decompose it into four 8×8 GF(2) sub-matrices (`mat_ll`, `mat_lh`, `mat_hl`, `mat_hh`) operating on byte pairs, then evaluate the map per 64-byte ZMM vector with two `VGF2P8AFFINEQB` instructions per output half (4 total per 64 bytes input) plus two `VPXORQ`. This mirrors ParPar's `gf16/gf16_affine_avx512.c`. MVP keeps par2rust's existing interleaved `[lo, hi, lo, hi, ...]` data layout and deinterleaves on load via `VPSHUFB`; ALTMAP (deinterleaved layout) is a follow-up.

**Tech Stack:** Rust `std::arch::x86_64` intrinsics, `#[target_feature(enable = "gfni,avx512f,avx512bw")]`, `is_x86_feature_detected!` runtime check, existing `Dispatch` enum.

---

## Scope and Non-Goals

**In scope:**
- Single new kernel: GFNI + AVX-512BW (64-byte vectors, 32 GF symbols per iter).
- Runtime dispatch addition.
- Cross-validation tests against the existing scalar reference.
- README and bench notes.

**Out of scope (follow-ups):**
- GFNI + AVX2 (256-bit) — covers Alder/Raptor Lake P-cores without AVX-512.
- GFNI + SSE/AVX-128 — covers Gracemont E-cores.
- ALTMAP (deinterleaved) data layout — biggest remaining perf win after GFNI.
- AVX2 SHUFFLE (non-GFNI) — separate plan; uncoupled from this one.
- Multi-buffer MD5.

## Critical constraint: dev hardware

**The developer machine is Apple Silicon (aarch64) and has no GFNI.** Every step that exercises GFNI code MUST run on x86 with GFNI hardware. Two options:

1. **GitHub Actions** — `ubuntu-latest` runners use Azure VMs with Ice Lake (Xeon Platinum 8370C / 8272CL) which support GFNI + AVX-512. Verify in Task 0.
2. **Local x86 VM / cloud box** — rent a `c7i.large` (Sapphire Rapids) for an hour to bench.

Cross-compilation works fine on the Mac (`cargo check --target x86_64-unknown-linux-gnu`), but **runtime tests cannot be skipped on Apple Silicon and assumed to pass on x86** — that's how subtle off-by-one bit-order bugs ship.

## File map

- **Create:** `src/galois_simd/gfni.rs` — kernel + table type.
- **Modify:** `src/galois_simd.rs` — add `Dispatch::Gfni`, `CoeffSimdTables::Gfni(...)`, wire dispatch.
- **Modify:** `tests/` — no new file; cross-validation test lives inline in `galois_simd.rs::tests` (matches existing pattern).
- **Modify:** `.github/workflows/ci.yml` — add a CPU-feature probe step and document the runner GFNI status.
- **Modify:** `README.md` — note new kernel and CPU requirements.

(Note: `galois_simd.rs` today is a single file with inline `mod neon` / `mod x86`. The new `mod gfni` will follow the same pattern. If you'd rather split `galois_simd.rs` into a directory module, do it as a separate refactor *before* Task 1 — don't bundle.)

---

## Task 0: Confirm CI runner has GFNI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Why first:** the whole plan hinges on having an automated GFNI-capable runner. If `ubuntu-latest` doesn't expose GFNI, we need to switch to `ubuntu-24.04` or a specific image, OR plan around manual testing. Do this once, find out, write it down.

- [ ] **Step 1: Add a probe step to CI**

In `.github/workflows/ci.yml`, after the "Show par2 version" step, add:

```yaml
      - name: Probe CPU features (Linux)
        if: runner.os == 'Linux'
        run: |
          echo "=== /proc/cpuinfo flags (first line) ==="
          grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(gfni|avx512f|avx512bw|vpclmulqdq|sse4_2|ssse3)$' | sort -u
          echo "=== model ==="
          grep -m1 '^model name' /proc/cpuinfo
```

- [ ] **Step 2: Push and read CI output**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: probe CPU features on Linux runner to verify GFNI availability"
git push
```

Open the latest Actions run and read the "Probe CPU features" step. Expected (Azure Ice Lake): `avx512bw`, `avx512f`, `gfni` all present.

- [ ] **Step 3: Record the result**

Append a line to this plan file under "Critical constraint" noting what CI exposes (e.g. "CI ubuntu-latest 2026-05-22: gfni + avx512f + avx512bw present"). If GFNI is absent, **STOP** and replan: either pin a different runner image or accept manual testing only.

---

## Task 1: Affine-matrix derivation (pure Rust, no SIMD)

**Files:**
- Create: `src/galois_simd/gfni.rs`

This task derives the four 8×8 GF(2) matrices that represent multiplication by a fixed GF(2^16) coefficient. **No SIMD yet** — pure scalar code, fully testable on Apple Silicon.

The math: for each input bit position `i ∈ 0..16`, multiply the basis vector `e_i` (a 1 in bit position `i`, treating bits 0–7 as the low byte and 8–15 as the high byte) by `c` in GF(2^16). The 16-bit result forms column `i` of the 16×16 transformation matrix. We then carve that matrix into four 8×8 blocks:

| Block | Rows | Cols | Meaning |
|---|---|---|---|
| `mat_ll` | 0–7 | 0–7 | low output bits from low input bits |
| `mat_lh` | 0–7 | 8–15 | low output bits from high input bits |
| `mat_hl` | 8–15 | 0–7 | high output bits from low input bits |
| `mat_hh` | 8–15 | 8–15 | high output bits from high input bits |

Each 8×8 block is packed into a `u64` with **Intel's GF2P8AFFINEQB convention**: byte `j` of the qword holds row `j` of the matrix, with bit `i` of that byte representing `mat[j][i]`. The instruction computes `y[i] = parity(row_i AND x)` per output bit.

> **Bit order is the single most error-prone part of this task.** Reference: Intel SDM Vol 2, GF2P8AFFINEQB description. Verify against a known scalar trace before believing the matrix.

- [ ] **Step 1: Add `mod gfni` declaration**

Add to `src/galois_simd.rs` after the existing `mod x86` block:

```rust
#[cfg(target_arch = "x86_64")]
mod gfni;
```

- [ ] **Step 2: Write the failing test**

Add this test to `src/galois_simd.rs::tests` (above the existing `ssse3_matches_scalar_when_available` test):

```rust
#[cfg(target_arch = "x86_64")]
#[test]
fn gfni_affine_matrices_round_trip_a_single_symbol() {
    use crate::galois_simd::gfni::GfniTables;
    // For every coefficient c, the derived matrices should reproduce the
    // scalar log-table result for every 16-bit input symbol.
    let coeffs = [0x0002u16, 0x00FF, 0x0100, 0x1234, 0x8000, 0xABCD, 0xFFFE, 0xFFFF];
    for &c in &coeffs {
        let t = GfniTables::from_coeff(c);
        for sym in 0u32..=0xFFFF {
            let lo = (sym & 0xFF) as u8;
            let hi = (sym >> 8) as u8;
            let (out_lo, out_hi) = t.apply_scalar(lo, hi);
            // Reference: existing scalar log-table multiply (one symbol).
            let mut scalar_out = [0u8; 2];
            super::gf_mul_xor_scalar(c, &(sym as u16).to_le_bytes(), &mut scalar_out);
            assert_eq!(
                (out_lo, out_hi),
                (scalar_out[0], scalar_out[1]),
                "coeff=0x{:04X} sym=0x{:04X}: GFNI matrix path diverged",
                c, sym,
            );
        }
    }
}
```

- [ ] **Step 3: Run the failing test**

```bash
cargo test --lib gfni_affine_matrices_round_trip -- --nocapture
```

Expected: compile error — `GfniTables` doesn't exist yet.

- [ ] **Step 4: Implement `GfniTables` (scalar-only)**

Create `src/galois_simd/gfni.rs`:

```rust
//! GFNI (GF2P8AFFINEQB) Galois-field kernel for x86_64.
//!
//! This module derives, for a fixed GF(2^16) coefficient `c`, the four
//! 8×8 GF(2) sub-matrices that together express multiplication by `c` on
//! a 16-bit symbol decomposed into a (low_byte, high_byte) pair. The
//! SIMD kernel evaluates these matrices with `VGF2P8AFFINEQB` — see
//! `gf_mul_xor_gfni` below.
//!
//! The Intel GF2P8AFFINEQB convention packs an 8×8 GF(2) matrix into a
//! u64: byte `j` is row `j`, bit `i` of that byte is mat[j][i]. The
//! instruction computes `y[i] = parity(row_i AND x) XOR imm[i]` per
//! output bit. We pass `imm = 0` (no constant XOR) and pre-build all
//! four matrices once per coefficient in `from_coeff`.
//!
//! Note: this file is `pub(super)` only; the dispatch enum in the
//! parent module is the only public entry point.

use crate::galois::gf_mul; // existing scalar GF(2^16) multiply

#[derive(Clone, Copy, Debug)]
pub(super) struct GfniTables {
    /// Low-output-byte from low-input-byte (rows 0..8, cols 0..8).
    pub(super) mat_ll: u64,
    /// Low-output-byte from high-input-byte (rows 0..8, cols 8..16).
    pub(super) mat_lh: u64,
    /// High-output-byte from low-input-byte (rows 8..16, cols 0..8).
    pub(super) mat_hl: u64,
    /// High-output-byte from high-input-byte (rows 8..16, cols 8..16).
    pub(super) mat_hh: u64,
}

impl GfniTables {
    /// Derive the four affine matrices for multiplication by `coeff` in
    /// GF(2^16). Cost: 16 scalar GF multiplies + bit-shuffling — well
    /// under 1 µs on any modern CPU.
    pub(super) fn from_coeff(coeff: u16) -> Self {
        // Column j of the 16x16 matrix = coeff * (1 << j), expressed in
        // GF(2^16). We then transpose into Intel's row-major byte
        // packing for GF2P8AFFINEQB.
        let mut cols = [0u16; 16];
        for j in 0..16u32 {
            cols[j as usize] = gf_mul(coeff, 1u16 << j);
        }

        // For each output bit `i` and input bit `j`, bit[i,j] = (cols[j] >> i) & 1.
        // Pack rows into bytes: row i of the 8x8 block (mat) is a byte
        // where bit i' (column within the block) is mat[i][i'].
        let pack_block = |out_bit_base: u32, in_bit_base: u32| -> u64 {
            let mut acc = 0u64;
            for i in 0..8u32 {
                let mut row: u8 = 0;
                for ip in 0..8u32 {
                    let col = in_bit_base + ip;
                    let bit = (cols[col as usize] >> (out_bit_base + i)) & 1;
                    // Intel convention: bit `ip` of the row byte represents
                    // column `ip` within the 8x8 block. The instruction
                    // applies row to input MSB-first, so we place column
                    // `ip` at bit position `7 - ip` within the row byte.
                    row |= (bit as u8) << (7 - ip);
                }
                acc |= (row as u64) << (i * 8);
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

    /// Scalar reference implementation of one symbol through the four
    /// matrices — used by `from_coeff`'s test to prove the bit ordering
    /// is correct before we trust the SIMD kernel. Hot path uses
    /// `gf_mul_xor_gfni`, not this.
    #[cfg(test)]
    pub(super) fn apply_scalar(&self, lo: u8, hi: u8) -> (u8, u8) {
        let out_lo = affine_apply(self.mat_ll, lo) ^ affine_apply(self.mat_lh, hi);
        let out_hi = affine_apply(self.mat_hl, lo) ^ affine_apply(self.mat_hh, hi);
        (out_lo, out_hi)
    }
}

/// Software emulation of one GF2P8AFFINEQB byte: y[i] = parity(row_i & x).
/// MSB-first within the row byte, matching the Intel spec.
#[cfg(test)]
fn affine_apply(mat: u64, x: u8) -> u8 {
    let mut y: u8 = 0;
    for i in 0..8u32 {
        let row = ((mat >> (i * 8)) & 0xFF) as u8;
        // y[i] = parity(row & x_bits_reversed_for_msb_first_convention)
        // GF2P8AFFINEQB applies row bit `b` to input bit `7-b`.
        let mut acc = 0u8;
        for b in 0..8u32 {
            let row_bit = (row >> (7 - b)) & 1;
            let x_bit = (x >> b) & 1;
            acc ^= row_bit & x_bit;
        }
        y |= acc << i;
    }
    y
}
```

If `crate::galois::gf_mul` doesn't exist with that signature, find the scalar GF(2^16) multiply in `src/galois.rs` (likely named `mul` or accessed via log/antilog tables) and adapt the import. Don't re-derive it.

- [ ] **Step 5: Run the test on aarch64 (Apple Silicon)**

```bash
cargo test --lib gfni_affine_matrices_round_trip -- --nocapture
```

Expected: PASS. The test is pure scalar — it runs anywhere, no GFNI hardware required. Total runtime ~0.5s (8 coeffs × 65536 symbols).

If it fails: the bit-ordering convention is wrong. Print one mismatched case in detail (coeff, sym, expected, got) and walk through `pack_block` by hand. Common bugs: row-vs-column swap, MSB-vs-LSB-first row packing, missing `(7 - ip)` reversal.

- [ ] **Step 6: Commit**

```bash
git add src/galois_simd.rs src/galois_simd/gfni.rs
git commit -m "feat(gfni): derive 8x8 GF(2) affine matrices from GF(2^16) coefficient"
```

---

## Task 2: GFNI + AVX-512 kernel (the SIMD part)

**Files:**
- Modify: `src/galois_simd/gfni.rs` (add `gf_mul_xor_gfni` function)

- [ ] **Step 1: Add the failing dispatch test**

Add to `src/galois_simd.rs::tests` (next to `ssse3_matches_scalar_when_available`):

```rust
#[cfg(target_arch = "x86_64")]
#[test]
fn gfni_matches_scalar_when_available() {
    if !std::is_x86_feature_detected!("gfni")
        || !std::is_x86_feature_detected!("avx512f")
        || !std::is_x86_feature_detected!("avx512bw")
    {
        eprintln!("skipping: GFNI / AVX-512BW not available on this CPU");
        return;
    }
    let coeffs = [0x0002u16, 0x00FF, 0x1234, 0xABCD, 0xFFFE, 0xFFFF];
    // Cover the SIMD body (multiples of 64) and the scalar tail (everything else).
    let lengths = [2usize, 4, 32, 62, 64, 66, 128, 130, 1024, 4096, 4098];
    for &coeff in &coeffs {
        for &len in &lengths {
            let input = deterministic((coeff as u64).rotate_left(7) ^ len as u64, len);
            check_against_scalar(coeff, &input, Dispatch::Gfni);
        }
    }
}
```

- [ ] **Step 2: Run the test — expect compile error**

```bash
cargo test --lib gfni_matches_scalar_when_available
```

Expected: `Dispatch::Gfni` doesn't exist. Next task adds it.

- [ ] **Step 3: Implement the SIMD kernel**

Append to `src/galois_simd/gfni.rs`:

```rust
use std::arch::x86_64::*;

/// `output ^= coeff · input` using GFNI + AVX-512.
///
/// # Safety
/// Caller must ensure `gfni`, `avx512f`, and `avx512bw` are runtime-available.
/// The dispatch enum in the parent module enforces this — never call this
/// function directly. `input.len() == output.len()` is asserted in debug.
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(super) unsafe fn gf_mul_xor_gfni(t: &GfniTables, input: &[u8], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(input.len() % 2 == 0, "GF(2^16) needs even byte count");

    // Broadcast each 8-byte affine matrix to all 8 qword lanes of a ZMM.
    let m_ll = _mm512_set1_epi64(t.mat_ll as i64);
    let m_lh = _mm512_set1_epi64(t.mat_lh as i64);
    let m_hl = _mm512_set1_epi64(t.mat_hl as i64);
    let m_hh = _mm512_set1_epi64(t.mat_hh as i64);

    // par2rust stores symbols interleaved: [lo0, hi0, lo1, hi1, ...].
    // To feed GF2P8AFFINEQB we need 64 contiguous low bytes in one ZMM
    // and 64 contiguous high bytes in another. Build a 64-byte PSHUFB
    // index that deinterleaves a 64-byte ZMM into [lo×32 | hi×32].
    //
    // PSHUFB within ZMM operates per-128-bit lane, so we use a global
    // permute table via VPERMB (AVX-512VBMI) — but VBMI isn't on all
    // GFNI parts. Stick to lane-local PSHUFB + a final VPERMQ to fix
    // up lane order. This keeps the MVP CPU floor at Ice Lake (no VBMI
    // requirement for Tiger Lake / Sapphire Rapids parity).

    const DEINTERLEAVE_LO: [u8; 64] = {
        let mut a = [0u8; 64];
        // Lane k (16 bytes): even bytes [0,2,4,..,14] go to positions [0..8],
        // odd bytes go to [8..16]. After VPSHUFB, lane k has [lo×8 | hi×8].
        let mut k = 0;
        while k < 4 {
            let base = k * 16;
            let mut i = 0;
            while i < 8 {
                a[base + i] = (base + 2 * i) as u8;
                a[base + 8 + i] = (base + 2 * i + 1) as u8;
                i += 1;
            }
            k += 1;
        }
        a
    };
    let deint = _mm512_loadu_si512(DEINTERLEAVE_LO.as_ptr() as *const _);

    // Per-lane gather permute: after deinterleave each 16B lane is
    // [lo×8 | hi×8]; we want all lows packed into the lower 256 bits,
    // all highs into the upper 256 bits. VPERMQ with this index does it.
    let lo_pack = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let n = input.len();
    let body = n & !63;
    let mut off = 0;

    // SAFETY: bounds enforced by `body = n & !63` and `off + 64 <= body`.
    while off < body {
        let v = _mm512_loadu_si512(input.as_ptr().add(off) as *const _);
        let dv = _mm512_shuffle_epi8(v, deint);            // per-lane deinterleave
        let pq = _mm512_permutex2var_epi64(dv, lo_pack, dv); // gather lows | highs
        // pq lower-256 = 32 low bytes; pq upper-256 = 32 high bytes.
        let lo_vec = _mm512_castsi256_si512(_mm512_extracti64x4_epi64(pq, 0));
        let hi_vec = _mm512_castsi256_si512(_mm512_extracti64x4_epi64(pq, 1));

        // Apply the four affine maps.
        let out_lo = _mm512_xor_si512(
            _mm512_gf2p8affine_epi64_epi8::<0>(lo_vec, m_ll),
            _mm512_gf2p8affine_epi64_epi8::<0>(hi_vec, m_lh),
        );
        let out_hi = _mm512_xor_si512(
            _mm512_gf2p8affine_epi64_epi8::<0>(lo_vec, m_hl),
            _mm512_gf2p8affine_epi64_epi8::<0>(hi_vec, m_hh),
        );

        // Re-interleave [lo | hi] back into [lo0 hi0 lo1 hi1 ...] and XOR
        // into the output buffer. VPUNPCKLBW / VPUNPCKHBW per 128-bit lane,
        // then permute lanes back. (Or just use a scalar tail-up to keep the
        // MVP simple — the body's win dwarfs the re-interleave cost.)
        let lo_256 = _mm512_castsi512_si256(out_lo);
        let hi_256 = _mm512_castsi512_si256(out_hi);
        let interleaved_lo = _mm256_unpacklo_epi8(lo_256, hi_256);
        let interleaved_hi = _mm256_unpackhi_epi8(lo_256, hi_256);
        // VPERMQ to fix 128b-lane ordering after unpack.
        let permuted_lo = _mm256_permute4x64_epi64(interleaved_lo, 0b1101_1000);
        let permuted_hi = _mm256_permute4x64_epi64(interleaved_hi, 0b1101_1000);

        let dst = output.as_mut_ptr().add(off);
        let existing_lo = _mm256_loadu_si256(dst as *const _);
        let existing_hi = _mm256_loadu_si256(dst.add(32) as *const _);
        _mm256_storeu_si256(dst as *mut _, _mm256_xor_si256(existing_lo, permuted_lo));
        _mm256_storeu_si256(dst.add(32) as *mut _, _mm256_xor_si256(existing_hi, permuted_hi));

        off += 64;
    }

    // Scalar tail: handle the trailing < 64 bytes one symbol at a time
    // through the same affine matrices to keep behaviour identical.
    while off < n {
        let lo = input[off];
        let hi = input[off + 1];
        let (o_lo, o_hi) = t.apply_scalar(lo, hi);
        output[off] ^= o_lo;
        output[off + 1] ^= o_hi;
        off += 2;
    }
}
```

> **Heads up:** the SIMD body above is the *plan-level sketch*. The deinterleave / re-interleave dance is the part most likely to need iteration. Treat the first run on CI as the source of truth and adjust shuffle indices until the test in Task 1 Step 5 (extended to SIMD) reports byte-for-byte parity with the scalar reference. Use `xxd | head` on a 128-byte test input to compare buffers when debugging.

Also: `apply_scalar` is currently `#[cfg(test)]`. The tail loop calls it from non-test code — promote it to always-compiled (drop the `cfg(test)`) in the same edit.

- [ ] **Step 4: Wire `Dispatch::Gfni` and `CoeffSimdTables::Gfni`**

Edit `src/galois_simd.rs` around lines 428–509:

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Dispatch {
    Scalar,
    TableScalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Ssse3,
    #[cfg(target_arch = "x86_64")]
    Gfni,
}
```

```rust
pub fn detect_dispatch() -> Dispatch {
    #[cfg(target_arch = "aarch64")]
    {
        return Dispatch::Neon;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("gfni")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
        {
            return Dispatch::Gfni;
        }
        if std::is_x86_feature_detected!("ssse3") {
            return Dispatch::Ssse3;
        }
    }
    #[allow(unreachable_code)]
    Dispatch::TableScalar
}
```

```rust
pub enum CoeffSimdTables {
    Trivial(u16),
    Scalar(u16),
    TableScalar(Box<CoeffTables>),
    #[cfg(target_arch = "aarch64")]
    Neon(neon::NeonTables),
    #[cfg(target_arch = "x86_64")]
    Ssse3(x86::X86Tables),
    #[cfg(target_arch = "x86_64")]
    Gfni(gfni::GfniTables),
}
```

```rust
impl CoeffSimdTables {
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
                let bt = CoeffTables::new(coeff);
                CoeffSimdTables::Neon(neon::NeonTables::from_coeff_tables(&bt))
            }
            #[cfg(target_arch = "x86_64")]
            Dispatch::Ssse3 => {
                let bt = CoeffTables::new(coeff);
                CoeffSimdTables::Ssse3(x86::X86Tables::from_coeff_tables(&bt))
            }
            #[cfg(target_arch = "x86_64")]
            Dispatch::Gfni => CoeffSimdTables::Gfni(gfni::GfniTables::from_coeff(coeff)),
        }
    }
}
```

And in `gf_mul_xor_with_tables`:

```rust
#[cfg(target_arch = "x86_64")]
CoeffSimdTables::Gfni(t) => {
    // SAFETY: constructed only when detect_dispatch returned Gfni,
    // which itself runtime-checks gfni + avx512f + avx512bw.
    unsafe { gfni::gf_mul_xor_gfni(t, input, output) };
}
```

- [ ] **Step 5: Run cross-validation test on CI (push to a branch)**

You **cannot** run this locally on Apple Silicon. Push to a branch:

```bash
git checkout -b feat/gfni-kernel
git add -A
git commit -m "feat(gfni): GFNI + AVX-512 kernel and dispatch wiring"
git push -u origin feat/gfni-kernel
```

Open the Actions run. Expected:
- `gfni_affine_matrices_round_trip_a_single_symbol` — PASS (also passes on aarch64).
- `gfni_matches_scalar_when_available` — PASS on Linux x86_64 if CI runner has GFNI (per Task 0); SKIPPED with the stderr message otherwise.
- `ssse3_matches_scalar_when_available` — still PASS.
- All existing aarch64 / NEON tests on macos-latest — PASS.

If `gfni_matches_scalar_when_available` fails, the SIMD deinterleave or affine matrix bit order is wrong. Print one failing `(coeff, len, input, scalar_out, gfni_out)` and bisect: reduce length to 64 (one body iteration, no tail), reduce coeff to `0x0002` (multiply by 2, well-known result), inspect bytes.

- [ ] **Step 6: Iterate until green**

Local on Apple Silicon: `cargo check --target x86_64-unknown-linux-gnu` after every edit to catch shuffle index typos without a CI round-trip.

```bash
rustup target add x86_64-unknown-linux-gnu  # one-time
cargo check --target x86_64-unknown-linux-gnu --lib
```

Note: `cargo check` doesn't run tests but catches type / intrinsic-signature errors fast.

- [ ] **Step 7: Commit when green**

```bash
git add -A
git commit -m "fix(gfni): correct shuffle indices for deinterleave (post-CI iteration)"
git push
```

(Squash into the previous commit if you prefer a clean history — your call.)

---

## Task 3: Benchmark and document the win

**Files:**
- Create: `benches/gf_kernel.rs`
- Modify: `Cargo.toml` (add criterion as dev-dep and bench entry)
- Modify: `README.md`

- [ ] **Step 1: Add criterion dev-dep and bench entry**

In `Cargo.toml`:

```toml
[dev-dependencies]
tempfile = "3"
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "gf_kernel"
harness = false
```

- [ ] **Step 2: Write the bench**

Create `benches/gf_kernel.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use par2rust::galois_simd::{
    detect_dispatch, gf_mul_xor_dispatch, gf_mul_xor_with_tables, CoeffSimdTables, Dispatch,
};

fn bench_kernel(c: &mut Criterion) {
    let dispatch = detect_dispatch();
    let coeff: u16 = 0xABCD;
    // 1 MiB chunk — well-sized for L2 on any modern part.
    let input = vec![0xA5u8; 1024 * 1024];
    let mut output = vec![0u8; input.len()];
    let tables = CoeffSimdTables::new(dispatch, coeff);

    let mut g = c.benchmark_group(format!("gf_mul_xor / {:?}", dispatch));
    g.throughput(Throughput::Bytes(input.len() as u64));
    g.bench_function("1MiB", |b| {
        b.iter(|| {
            gf_mul_xor_with_tables(&tables, black_box(&input), black_box(&mut output));
        });
    });
    g.finish();
}

criterion_group!(benches, bench_kernel);
criterion_main!(benches);
```

(This requires `galois_simd` to be `pub` — it already is at the crate root via `pub mod galois_simd;` in `src/lib.rs`. If not, add it. Don't expose new symbols beyond what the bench needs.)

- [ ] **Step 3: Run baseline on Apple Silicon**

```bash
cargo bench --bench gf_kernel
```

Record the NEON throughput in GB/s.

- [ ] **Step 4: Run on x86 with GFNI**

If you don't have local x86 GFNI hardware:

```bash
# On an EC2 c7i.large or equivalent
git clone <repo> && cd par2rust
git checkout feat/gfni-kernel
cargo bench --bench gf_kernel
```

Record the GFNI throughput. ParPar's notes claim ~2× over PSHUFB; expect somewhere in 1.5–2.2× depending on the deinterleave overhead in our MVP.

- [ ] **Step 5: Add a README perf table**

In `README.md`, find the existing performance section (around line 170 per the exploration). Append:

```markdown
### GFNI (AVX-512 + GFNI) — x86_64 only

On CPUs with both AVX-512BW and GFNI (Ice Lake, Tiger Lake, Sapphire Rapids,
Zen 4+), the GF(2^16) kernel uses `VGF2P8AFFINEQB` for ~2× the per-core
throughput of the SSSE3 PSHUFB path. Runtime-detected; no build flag needed.

| Kernel              | Hardware             | GF mul-XOR throughput |
|---------------------|----------------------|-----------------------|
| Scalar (log/exp)    | Any                  | ~0.4 GB/s             |
| SSSE3 PSHUFB        | x86_64 with SSSE3    | ~3.2 GB/s             |
| NEON                | aarch64              | ~6.5 GB/s (M-series)  |
| GFNI + AVX-512      | Ice Lake / Zen 4+    | ~6.0–7.5 GB/s         |

(Numbers from `cargo bench --bench gf_kernel` on a 1 MiB buffer. Update with your hardware.)
```

Fill in your actual measured numbers.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml benches/gf_kernel.rs README.md
git commit -m "bench: criterion harness for GF kernel; document GFNI perf"
```

---

## Task 4: Open the PR

- [ ] **Step 1: Final CI green check**

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features --lib
git push
```

Wait for green on all three CI runners.

- [ ] **Step 2: PR description**

Use this body (adjust numbers):

```markdown
## Summary
- Add GFNI + AVX-512 kernel (`Dispatch::Gfni`) for x86_64 with `VGF2P8AFFINEQB`
- Runtime-detected; falls back to SSSE3 → TableScalar → Scalar
- Cross-validated byte-for-byte against the scalar reference on CI

## Perf
- 1 MiB buffer, single-core:
  - SSSE3 (baseline): X.X GB/s
  - GFNI + AVX-512: Y.Y GB/s (Z.Zx)

## Test plan
- [x] `gfni_affine_matrices_round_trip_a_single_symbol` (pure scalar, runs everywhere)
- [x] `gfni_matches_scalar_when_available` (skips if no GFNI hardware)
- [x] Existing tests untouched, all green on Linux/macOS/Windows CI
- [x] criterion bench published in README

## Follow-ups (not in this PR)
- GFNI + AVX2 kernel for parts without AVX-512 (Alder/Raptor P-cores)
- ALTMAP (deinterleaved layout) to drop the per-vector deinterleave shuffle
- AVX2 SHUFFLE (non-GFNI) for Haswell→Rocket Lake
- Multi-buffer MD5
```

```bash
gh pr create --title "feat: GFNI + AVX-512 Galois-field kernel" --body-file <description>
```

---

## Self-review

**Spec coverage:** Plan covers the GFNI gap (#1 in the parent gap analysis). The other gaps (AVX2 SHUFFLE, multi-buffer MD5, cache-blocking, async I/O) are explicitly out of scope — see "Follow-ups" in Task 4 PR description.

**Placeholder scan:** Code blocks are concrete. The one judgement call left to the implementer is the exact shuffle index for deinterleave / re-interleave (Task 2 Step 3), which is flagged in-text with a debugging recipe; this is unavoidable because the layout choice (interleaved vs ALTMAP) interacts with vector width and the only reliable way to nail it is to run on real hardware and compare against the scalar reference.

**Type consistency:** `GfniTables` / `Dispatch::Gfni` / `CoeffSimdTables::Gfni` / `gf_mul_xor_gfni` names are consistent across Tasks 1–3. `from_coeff` is used everywhere (not `new`, which would clash with the other variants' table constructors that take `&CoeffTables`).

**Dev-hardware reality check:** Tasks 0 and 1 run on Apple Silicon. Tasks 2–4 require CI or a remote x86 box. This is called out at the top and again per-task.

---

## Verification (end-to-end)

After the PR merges in this repo (par2rust), bump the consumer (`postie-rust`):

```bash
# in /Users/javi/postie-rust
cargo update -p par2rust
# Update Cargo.toml rev = "<new sha>"
cargo build --release
# Benchmark before/after with a real workload
hyperfine --warmup 1 'target/release/postie create --input /path/to/3gb.bin'
```

Expected: encode-phase wall time on a GFNI-capable Linux box drops by 30–50% (the kernel doubles but encode is partly bound by MD5 + I/O which haven't changed).

If you want to isolate the kernel win precisely, run `cargo bench --bench gf_kernel` on the same Linux box before and after the par2rust bump.
