# par2rust

A Rust implementation of par2 file creation.
Given a set of input files, `par2rust` produces a PAR2 recovery set that is
byte-compatible with the upstream tool — any standard PAR2 verifier (par2cmdline,
quickpar, multipar) can verify and repair files using the output.

## Status

- ✅ PAR2 create: index file + multi-volume recovery files (exponential
  split by default, single-volume via `--single-volume`)
- ✅ Reed-Solomon GF(2¹⁶) encoder with PAR2 generator `0x1100B`
- ✅ SIMD acceleration:
  - **NEON** on `aarch64` (Apple Silicon, ARM servers)
  - **SSSE3** on `x86_64`
  - Byte-table scalar fallback elsewhere
- ✅ Windows long-path support (`\\?\` prefix for paths >260 chars)
- ✅ Golden tests against upstream `par2 v` and `par2 r`
- 🚧 **Not implemented**: verify, repair, PAR1 legacy format,
  par2cmdline's `-u`/`-l` distribution flags (uniform / limit-count)

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

Output (default — `par2cmdline`-style exponential split):
```
Wrote 8 files:
  backup.par2 (552 bytes)
  backup.vol0+1.par2 ...
  backup.vol1+1.par2 ...
  backup.vol2+2.par2 ...
  backup.vol4+4.par2 ...
  backup.vol8+8.par2 ...
  backup.vol16+16.par2 ...
  backup.vol32+18.par2 ...
```

To collapse into a single recovery file (the previous default), pass
`--single-volume`:
```bash
par2rust create --single-volume -s 262144 -c 50 backup.par2 data.bin
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
            ..Default::default()
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
- `CreateOptions { output, slice_size, recovery_block_count, volume_scheme }`
- `VolumeScheme::{Single, Exponential, Explicit(Vec<u32>)}` — recovery-file split
- Errors via `Par2Error` (`thiserror`-derived)
- Constants: `MAX_FILES`, `MAX_RECOVERY_BLOCKS`

## Performance

`par2rust` parallelises the Reed-Solomon accumulator with `rayon` (chunked per
worker thread to keep scheduling overhead negligible). Pass `-t/--threads N` to
pin the worker count; `0` (default) uses one per logical CPU.

Benchmark — Apple M4, 10 cores, macOS 25.3, 16 GB RAM. Workload: a 513 MB MKV
file with `-s 524288 -c 200` (~10% redundancy, multi-volume exponential split).
Best of 3 runs:

| Tool                             | Wall   | User CPU | Notes                            |
|----------------------------------|-------:|---------:|----------------------------------|
| `par2rust create` (default, 10 threads) | **4.55 s** | 13.50 s  | this crate                       |
| `par2cmdline 1.1.1` (OpenMP)     | 6.18 s | 25.03 s  | upstream reference               |
| `par2rust create -t 1`           | 7.16 s | 6.49 s   | single-threaded baseline         |

Result on this hardware: **~26% faster wall-clock than par2cmdline** while using
**~46% less CPU time** (≈3.9× more cycle-efficient per unit of wall time). The
single-threaded mode still beats par2cmdline's *per-core* throughput by ~3.5×
thanks to the NEON GF(2¹⁶) multiplier.

Scaling on this workload plateaus around 4 threads — at 200 × 524 KB recovery
buffers (≈105 MB live) the inner loop becomes memory-bandwidth-bound rather
than compute-bound. Tune `slice-size` / `recovery-count` to fit your CPU's
shared cache for best results.

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
