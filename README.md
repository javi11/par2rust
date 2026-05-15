# par2rust

A Rust port of the **create** side of [par2cmdline](https://github.com/Parchive/par2cmdline).
Given a set of input files, `par2rust` produces a PAR2 recovery set that is
byte-compatible with the upstream tool — any standard PAR2 verifier (par2cmdline,
quickpar, multipar) can verify and repair files using the output.

## Status

- ✅ PAR2 create: index file + single-volume recovery file
- ✅ Reed-Solomon GF(2¹⁶) encoder with PAR2 generator `0x1100B`
- ✅ SIMD acceleration:
  - **NEON** on `aarch64` (Apple Silicon, ARM servers)
  - **SSSE3** on `x86_64`
  - Byte-table scalar fallback elsewhere
- ✅ Golden tests against upstream `par2 v` and `par2 r`
- 🚧 **Not implemented**: verify, repair, PAR1 legacy format, multi-volume
  distribution schemes, Windows wide-char path handling

## Install / build

Requires Rust 1.74+ and (for golden tests) the upstream `par2` binary:

```bash
brew install par2          # macOS
apt install par2           # Debian/Ubuntu
cargo build --release      # produces target/release/par2rust
```

## CLI usage

Same flag conventions as `par2cmdline`'s `create` subcommand:

```bash
# Protect data.bin with 50 recovery blocks of 256 KiB each
par2rust create -s 262144 -c 50 backup.par2 data.bin

# Protect several files at once
par2rust c -s 4096 -c 10 backup.par2 a.bin b.bin c.bin
```

Output:
```
Wrote 2 files:
  backup.par2 (552 bytes)
  backup.vol0+50.par2 (13_127_384 bytes)
```

Verify and repair with upstream tools:

```bash
par2 v backup.par2          # verify
par2 r backup.par2          # repair if any data file is damaged
```

## Use as a library

`par2rust` is also a regular Rust crate. Add it with the default `cli`
feature disabled so `clap` isn't pulled into your dependency tree:

```toml
[dependencies]
par2rust = { version = "0.1", default-features = false }
```

Call `run_create` directly:

```rust
use std::path::PathBuf;
use par2rust::{run_create, CreateOptions, SourceFile};

fn main() -> par2rust::Result<()> {
    let path = PathBuf::from("data.bin");
    let name = path.file_name().unwrap().as_encoded_bytes().to_vec();
    let source = SourceFile::scan(&path, name, 4096)?;

    let written = run_create(
        &CreateOptions {
            output: PathBuf::from("backup.par2"),
            slice_size: 4096,
            recovery_block_count: 10,
        },
        &[source],
    )?;
    for p in &written {
        println!("wrote {}", p.display());
    }
    Ok(())
}
```

A runnable version lives at [`examples/create_from_lib.rs`](examples/create_from_lib.rs):

```bash
cargo run --example create_from_lib
```

Public API surface:

- `run_create(&CreateOptions, &[SourceFile]) -> Result<Vec<PathBuf>>` — full create pipeline
- `SourceFile::scan(path, display_name, slice_size)` — hash one input file
- `CreateOptions { output, slice_size, recovery_block_count }`
- Errors via `Par2Error` (`thiserror`-derived)
- Constants: `MAX_FILES`, `MAX_RECOVERY_BLOCKS`

## Performance

On Apple Silicon (M-series), single-threaded, vs. upstream `par2cmdline` with OpenMP:

| Workload                  | par2cmdline (OpenMP) | par2rust (single-thread) |
|---------------------------|----------------------|--------------------------|
| 100 MB → 50 recovery blocks | 0.69 s wall / 1.62 s CPU | **0.54 s wall / 0.49 s CPU** |

`par2rust` is single-threaded but beats the multi-threaded C++ implementation
on wall-clock time thanks to the NEON GF(2¹⁶) multiplier. Per-core CPU time is
roughly 3.3× faster.

## Architecture

- [`src/format.rs`](src/format.rs) — packet header layout, magic constants, MD5 wrapper
- [`src/source.rs`](src/source.rs) — input file scanner (slice hashing, file_id derivation)
- [`src/packet/`](src/packet) — one builder per PAR2 packet type (main, file_desc, file_verify, recovery, creator)
- [`src/galois.rs`](src/galois.rs) — GF(2¹⁶) log/antilog tables and scalar arithmetic
- [`src/reedsolomon.rs`](src/reedsolomon.rs) — RS encoder (Vandermonde matrix construction + scalar reference)
- [`src/galois_simd.rs`](src/galois_simd.rs) — SIMD multiply-XOR paths (NEON, SSSE3, byte-table fallback)
- [`src/creator.rs`](src/creator.rs) — pipeline that orchestrates the create flow
- [`src/main.rs`](src/main.rs) — `clap`-based CLI

All SIMD paths are property-tested against the scalar reference for byte-identical
output across thousands of randomized inputs ([src/galois_simd.rs#tests](src/galois_simd.rs)).

## Testing

```bash
cargo test                 # unit + golden integration tests
cargo test --lib           # unit tests only (no upstream par2 needed)
PAR2_BIN=/usr/local/bin/par2 cargo test   # use a specific upstream binary
```

The golden suite covers:
- Index-only PAR2 (verify-only set) — accepted by upstream
- Multi-file index PAR2 — accepted by upstream
- Single-file corruption-and-repair round trip
- Multi-file corruption-and-repair round trip
- CLI end-to-end (subprocess invocation of our binary)

## License

GPL-2.0-or-later, matching the upstream par2cmdline project.
