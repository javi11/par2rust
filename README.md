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
- ✅ Tests against upstream `par2 v` and `par2 r`
- ✅ Distribution flags: `-u` (uniform),
  `-l` (limit volume size to largest source file), `-n<count>` (volume count)
- ✅ ParPar-style flags: `--out`, `--comment`, `--recurse`, `--input-file`,
  `--quiet`
- 🚧 **Not implemented**: verify, repair, PAR1 legacy format

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

Alternative distributions matching par2cmdline's flags:

```bash
# -u: uniform — split recovery blocks evenly. -n sets the volume count
# (defaults to 15, capped at the recovery-block total).
par2rust c -s 4096 -c 50 -u -n 5 backup.par2 data.bin
# → 5 volumes of 10 blocks each

# -l: cap each volume's size to the largest source file. Composes with -u
# and with the default exponential layout.
par2rust c -s 4096 -c 50 -l backup.par2 data.bin
```

Verify and repair with upstream tools:

```bash
par2 v backup.par2          # verify
par2 r backup.par2          # repair if any data file is damaged
```

### ParPar-style flags

A small subset of [ParPar](https://github.com/animetosho/ParPar)'s CLI is also
accepted, alongside the par2cmdline flags. Where ParPar's short flag conflicts
with par2cmdline (`-c`, `-n`, `-r`), only the long form is offered:

```bash
# -o/--out: alternate to the positional <ARCHIVE>. When --out is given, every
# positional argument is treated as an input file.
par2rust create -o backup.par2 a.bin b.bin

# --comment: embed a comment packet (repeatable). Non-ASCII text additionally
# emits a Unicode comment packet linked to the ASCII variant. (Long-only —
# par2cmdline already uses -c for --recovery-count.)
par2rust create --comment "release v1.2" --comment "by alice" backup.par2 data.bin

# -R/--recurse: walk directory inputs recursively (without it, a directory
# input is an error).
par2rust create -R backup.par2 ./photos

# -i/--input-file: read additional input paths from a newline-separated file
# (use "-" to read from stdin). Composes with positional inputs.
par2rust create -i files.txt backup.par2

# -q/--quiet: suppress progress bars and the "Wrote N files" summary. Errors
# still go to stderr.
par2rust create -q backup.par2 data.bin

# --version: print version and exit.
par2rust --version
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
- `VolumeScheme::{Single, Exponential, Uniform { count }, Limited { max_blocks_per_volume, inner }, Explicit(Vec<u32>)}` — recovery-file split
- Errors via `Par2Error` (`thiserror`-derived)
- Constants: `MAX_FILES`, `MAX_RECOVERY_BLOCKS`

## Performance

`par2rust` parallelises the Reed-Solomon accumulator with `rayon` (chunked per
worker thread to keep scheduling overhead negligible). Pass `-t/--threads N` to
pin the worker count; `0` (default) uses one per logical CPU.

Benchmark — Apple M4, 10 cores, macOS 25.3, 16 GB RAM. Workload: a 513 MiB
(538,218,411-byte) MKV file with `-s 524288 -c 200` (~10% redundancy,
multi-volume exponential split). Max RSS via `/usr/bin/time -l`. Best of 3
runs:

| Tool                             | Wall   | User CPU | Max RSS  | Notes                    |
|----------------------------------|-------:|---------:|---------:|--------------------------|
| `par2rust create` (default, 10 threads) | **5.52 s** | 16.02 s | 122 MB | this crate              |
| `par2cmdline 1.1.1` (OpenMP)     | 7.78 s | 28.39 s  | 104 MB   | upstream reference       |
| `par2rust create -t 1`           | 7.38 s | 6.73 s   | 137 MB   | single-threaded baseline |

Result on this hardware: **~29% faster wall-clock than par2cmdline** while
using **~44% less CPU time** (≈2.6× more cycle-efficient per unit of wall
time) for ~17% more resident memory. The single-threaded mode still beats
par2cmdline's *per-core* throughput by ~4.2× thanks to the NEON GF(2¹⁶)
multiplier.

Scaling on this workload plateaus around 4 threads — at 200 × 524 KB recovery
buffers (≈105 MB live) the inner loop becomes memory-bandwidth-bound rather
than compute-bound. Tune `slice-size` / `recovery-count` to fit your CPU's
shared cache for best results.

### GF(2¹⁶) kernel throughput

The Reed-Solomon inner loop dispatches to the fastest GF kernel available
on the host CPU. Runtime detection picks one of:

| Dispatch       | Hardware required           | Inner loop                                                |
|----------------|-----------------------------|-----------------------------------------------------------|
| `Neon`         | aarch64 (base ISA)          | `vqtbl1q_u8` nibble lookup, 32 bytes/iter                 |
| `Gfni`         | x86_64 with `gfni + ssse3`  | `GF2P8AFFINEQB` affine, 32 bytes/iter (~1.5–2× SSSE3)     |
| `Ssse3`        | x86_64 with `ssse3`         | `PSHUFB` nibble lookup, 32 bytes/iter                     |
| `TableScalar`  | any                         | Two 256-entry u16 lookup tables per coefficient           |
| `Scalar`       | any                         | Per-symbol log/antilog (correctness reference)            |

Bench harness: `cargo bench --bench gf_kernel`. Measures wall-clock
throughput on a 1 MiB L2-resident buffer; the dispatch name in the
group label tells you which kernel ran. Indicative numbers:

| Kernel          | Hardware                  | 1 MiB throughput |
|-----------------|---------------------------|------------------|
| `Neon`          | Apple M4 (10 cores)       | ~20.8 GiB/s      |
| `Gfni` / `Ssse3`| (run the bench on your x86 box to populate)              |

`Dispatch::Gfni` is preferred over `Ssse3` when both are available; the
two share an identical 32-bytes-per-iter data layout and differ only in
the per-iter math (4× `GF2P8AFFINEQB` + 2× `XOR` vs 8× `PSHUFB` + 6×
`XOR`). AVX2 (256-bit) and AVX-512 (512-bit) GFNI variants are planned
follow-ups; the 128-bit MVP is the lowest-risk introduction because
every shuffle stays within a single 16-byte lane.

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
