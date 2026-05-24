# Vendored ISA-L Crypto (multi-buffer MD5 subset)

Pinned to upstream tag **v2.26** (commit `07d0b25`), released 2026-01-22.

Source: https://github.com/intel/isa-l_crypto/tree/v2.26
License: BSD-3-Clause (see `LICENSE` in this directory).
Combined with par2rust's GPL-2.0-or-later, the output binary is GPL-2.0-or-later (BSD is compatible).

## What's vendored

Only the files needed for multi-buffer MD5 on aarch64 (ASIMD) and the
portable scalar fallback. NOT vendored:

- SVE / SVE2 paths (Apple Silicon doesn't expose them; Graviton / Neoverse follow-up).
- x86 SSE/AVX/AVX2/AVX-512 implementations (would need nasm at build time).
- Anything outside `md5_mb/`.

```
vendor/isa-l_crypto/
├── LICENSE                                          # upstream BSD-3-Clause
├── README.md                                         # this file
├── include/
│   ├── isa-l_crypto/
│   │   ├── md5_mb.h            # public API
│   │   ├── multi_buffer.h      # ISAL_HASH_CTX_FLAG, ISAL_HASH_CTX_STS, etc.
│   │   ├── types.h
│   │   └── isal_crypto_api.h
│   └── internal/
│       ├── md5_mb_internal.h
│       ├── memcpy_inline.h
│       └── endian_helper.h
└── md5_mb/
    ├── md5_ctx_base.c          # portable scalar fallback
    ├── md5_mb.c                # high-level ctx helpers
    ├── md5_ref.c               # RFC 1321 reference (used by tests)
    ├── md5_ctx_base_aliases.c  # symbol aliases for non-multibinary builds
    └── aarch64/
        ├── md5_ctx_aarch64_asimd.c       # ctx wrappers (entry points we call)
        ├── md5_mb_mgr_aarch64_asimd.c    # lane scheduler
        ├── md5_mb_asimd_x1.S             # 1-lane ASIMD round
        ├── md5_mb_asimd_x4.S             # 4-lane ASIMD round
        ├── md5_mb_multibinary.S          # multibinary dispatch (skipped, kept for reference)
        └── md5_mb_aarch64_dispatcher.c   # multibinary dispatcher (skipped — uses Linux `getauxval`)
```

## How par2rust calls into this

The multibinary dispatcher (`md5_mb_aarch64_dispatcher.c` + `md5_mb_multibinary.S`)
depends on Linux glibc (`<asm/hwcap.h>`, `<sys/auxv.h>`, `getauxval`) and is
**skipped** by `build.rs`. Instead, par2rust calls the ASIMD entry points
directly: `md5_ctx_mgr_init_asimd`, `md5_ctx_mgr_submit_asimd`,
`md5_ctx_mgr_flush_asimd`. ASIMD is part of the aarch64 base ISA so no runtime
detection is needed.

## Platform status

| Target            | Builds | Why |
|-------------------|--------|-----|
| Linux aarch64     | **expected** (untested) | ISA-L's ELF-aware GAS assembly is its native target. |
| Linux x86_64      | requires `nasm` in PATH; this Phase 1 vendor subset does NOT include x86 SIMD files (follow-up). |
| macOS aarch64     | **blocked** | Apple's Mach-O assembler rejects GNU-as ELF relocation syntax (`adrp x0, .label` + `#:lo12:.label`). The pattern is pervasive across `md5_mb_asimd_x[14].S` — rewriting requires per-instruction translation to Mach-O `@PAGE`/`@PAGEOFF` form. Out of scope for Phase 1. |
| macOS x86_64      | blocked (same reason once x86 .asm files vendor in). |

Phase 1 thus ships build scaffolding that is *correct on Linux* and *gated behind `mb-md5` feature so a default macOS build is unaffected*. The aarch64 macOS port requires either a Mach-O–compatible rewrite of the `.S` files or a switch to NEON intrinsics written in Rust (no `.S`). Track in a follow-up.

## Modifications from upstream

- **Symbol prefixing in `.S` files**: macOS Mach-O requires C symbols to be
  emitted with a leading `_` underscore. The upstream `.S` files use plain
  `.global md5_mb_asimd_x4` which would emit `md5_mb_asimd_x4` on Mach-O while
  the C caller (compiled by clang) emits a call to `_md5_mb_asimd_x4`. The
  vendored `.S` files are wrapped with a small `cdecl()` cpp macro to handle
  both conventions. See the diff in the corresponding files for the exact
  patches; each patched file carries a `// PAR2RUST PATCH:` comment.
- **ELF-only `.type`/`.size` directives** in the `.S` files are wrapped in
  `#ifdef __ELF__` guards (the Mach-O assembler errors on them otherwise).

These are local mechanical patches, not algorithmic changes — the SIMD math
and CTX state machine are byte-identical to upstream.

## Re-vendoring procedure

```bash
TAG=v2.27   # or whatever the next upstream release is
curl -sL "https://github.com/intel/isa-l_crypto/archive/refs/tags/${TAG}.tar.gz" -o /tmp/isal.tgz
tar -xzf /tmp/isal.tgz -C /tmp
# Re-copy the subset (see git history for the exact `cp` commands).
# Re-apply the macOS Mach-O patches (search for "PAR2RUST PATCH:" in this dir).
# Update the SHA + tag at the top of this README.
```
