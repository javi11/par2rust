use std::ffi::OsString;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rayon::prelude::*;

/// Fallback BufReader capacity for the encode read path on platforms where
/// mmap is unavailable. Must match `source::FALLBACK_READ_CAPACITY` in spirit:
/// 4 MiB keeps cold sequential reads close to the disk ceiling.
const FALLBACK_READ_CAPACITY: usize = 4 * 1024 * 1024;

/// Source-file reader used by the encode loop. Prefers mmap so the OS page
/// cache absorbs the inevitable re-read after the scan phase; falls back to a
/// 4 MiB BufReader on platforms / file systems where mmap fails (e.g. some
/// Windows network shares).
enum SourceReader {
    Mmap(Mmap),
    Buffered(BufReader<File>),
}

impl SourceReader {
    fn open(path: &Path) -> std::io::Result<Self> {
        let f = File::open(path)?;
        // Diagnostic escape hatch: when PAR2RUST_DISABLE_MMAP_ENCODE is set
        // (any value), skip the mmap fast path and use the 4 MiB BufReader
        // fallback. Lets callers A/B the encode-side mmap without rebuilding.
        let mmap_disabled = std::env::var_os("PAR2RUST_DISABLE_MMAP_ENCODE").is_some();
        if !mmap_disabled {
            // SAFETY: PAR2 input is read-only by contract. If the user mutates
            // the file while we read it the resulting recovery data is simply
            // wrong — not UB in the Rust sense. Same exposure as BufReader.
            if let Ok(m) = unsafe { Mmap::map(&f) } {
                return Ok(SourceReader::Mmap(m));
            }
        }
        Ok(SourceReader::Buffered(BufReader::with_capacity(
            FALLBACK_READ_CAPACITY,
            f,
        )))
    }

    /// Fill `slice_buf` with the bytes of slice `slice_idx`, zero-padding any
    /// tail that runs past EOF (PAR2 spec: trailing partial slice is zero-padded
    /// before checksumming / encoding).
    fn read_slice(&mut self, slice_idx: usize, slice_buf: &mut [u8]) -> std::io::Result<()> {
        match self {
            SourceReader::Mmap(m) => {
                let start = slice_idx.saturating_mul(slice_buf.len());
                let end = std::cmp::min(start.saturating_add(slice_buf.len()), m.len());
                let src = if start < m.len() {
                    &m[start..end]
                } else {
                    &[][..]
                };
                slice_buf[..src.len()].copy_from_slice(src);
                if src.len() < slice_buf.len() {
                    slice_buf[src.len()..].fill(0);
                }
                Ok(())
            }
            SourceReader::Buffered(r) => {
                let filled = read_full(r, slice_buf)?;
                if filled < slice_buf.len() {
                    slice_buf[filled..].fill(0);
                }
                Ok(())
            }
        }
    }
}

use crate::error::{Par2Error, Result};
use crate::format::Md5Hash;
use crate::galois_simd::{detect_dispatch, gf_mul_xor_with_tables, CoeffSimdTables, Dispatch};
use crate::packet::comment::build_comment_packets;
use crate::packet::creator::build_creator_packet;
use crate::packet::file_desc::build_file_desc_packet;
use crate::packet::file_verify::build_file_verify_packet;
use crate::packet::main_packet::{build_main_packet, MainPacket};
use crate::packet::recovery::build_recovery_packet;
use crate::progress::{tick_stride, ProgressEvent, ProgressReporter};
use crate::reedsolomon::RsEncoder;
use crate::source::SourceFile;

/// Maximum number of input files PAR2 supports.
pub const MAX_FILES: usize = 32_768;
/// Maximum number of input blocks (slices) PAR2 supports. This is φ(65535) —
/// the count of GF(2^16) bases whose discrete log is coprime to 65535, which
/// the Vandermonde RS matrix requires to remain invertible.
pub const MAX_INPUT_BLOCKS: u32 = 32_768;
/// Maximum number of recovery blocks per PAR2 set (16-bit exponent space).
pub const MAX_RECOVERY_BLOCKS: u32 = 65_535;

/// How to distribute recovery blocks across `.vol*+*.par2` files.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum VolumeScheme {
    /// All recovery blocks go into a single `vol0+N.par2` file.
    #[default]
    Single,
    /// par2cmdline's default scheme: powers-of-two block counts per volume
    /// (`1, 1, 2, 4, 8, 16, …`), with the final volume holding any remainder
    /// so the sum equals the total recovery block count.
    Exponential,
    /// par2cmdline `-u` (optionally with `-n<count>`): split the total block
    /// count across `count` volumes as evenly as possible. The first
    /// `count - 1` volumes each get `total / count` blocks; the final volume
    /// absorbs the remainder.
    Uniform { count: u32 },
    /// par2cmdline `-l`: cap each volume so its on-disk size does not exceed
    /// the largest source file. The cap is expressed in recovery blocks
    /// (`floor(largest_source_size / slice_size)`). The inner scheme decides
    /// the growth pattern (`Exponential` for plain `-l`, `Uniform` for
    /// `-u -l`); the resolver splits any over-cap entry into cap-sized
    /// chunks.
    Limited {
        max_blocks_per_volume: u32,
        inner: Box<VolumeScheme>,
    },
    /// Caller-supplied volume sizes. The sum must equal
    /// `recovery_block_count`; otherwise [`run_create`] returns
    /// [`Par2Error::InvalidVolumeScheme`].
    Explicit(Vec<u32>),
}

/// Configuration for one create run.
pub struct CreateOptions {
    /// Output path for the index file. Volume files derive their names from
    /// this by replacing the `.par2` extension with `.volX+N.par2`.
    pub output: PathBuf,
    /// Slice size in bytes. Must be > 0 and a multiple of 4.
    pub slice_size: u64,
    /// Number of recovery blocks to produce. 0 → no volume file is written
    /// (Phase 3 milestone: index-only output).
    pub recovery_block_count: u32,
    /// How to split recovery blocks across volume files. Defaults to
    /// [`VolumeScheme::Single`] for backwards compatibility with earlier
    /// callers; the CLI overrides this with [`VolumeScheme::Exponential`] to
    /// match `par2cmdline`'s default behaviour.
    pub volume_scheme: VolumeScheme,
    /// Optional comment packets to embed alongside the critical packets.
    /// Each entry produces an ASCII comment packet; entries with any
    /// non-ASCII character additionally produce a Unicode comment packet
    /// linked to the ASCII variant (matches ParPar's `-c/--comment`).
    pub comments: Vec<String>,
}

impl Default for CreateOptions {
    fn default() -> Self {
        Self {
            output: PathBuf::new(),
            slice_size: 4096,
            recovery_block_count: 0,
            volume_scheme: VolumeScheme::Single,
            comments: Vec::new(),
        }
    }
}

/// Run the full create pipeline:
///   - write the index `.par2` (critical packets only),
///   - if `recovery_block_count > 0`, compute and write a single
///     `.vol0+N.par2` containing all recovery blocks (plus the critical
///     packets repeated for redundancy).
///
/// Returns the list of all files written, with the index file first.
pub fn run_create(opts: &CreateOptions, sources: &[SourceFile]) -> Result<Vec<PathBuf>> {
    run_create_with_progress(opts, sources, None)
}

/// Like [`run_create`] but emits progress events through `reporter`.
/// Pass `None` for behaviour identical to `run_create`.
pub fn run_create_with_progress(
    opts: &CreateOptions,
    sources: &[SourceFile],
    reporter: Option<&dyn ProgressReporter>,
) -> Result<Vec<PathBuf>> {
    if sources.is_empty() {
        return Err(Par2Error::NoInputFiles);
    }
    if sources.len() > MAX_FILES {
        return Err(Par2Error::TooManyFiles(sources.len()));
    }
    if opts.recovery_block_count > MAX_RECOVERY_BLOCKS {
        return Err(Par2Error::TooManyRecoveryBlocks(opts.recovery_block_count));
    }
    if opts.slice_size == 0 || opts.slice_size % 4 != 0 {
        return Err(Par2Error::InvalidSliceSize(opts.slice_size));
    }

    let total_input_blocks: u64 = sources.iter().map(|s| s.slice_checksums.len() as u64).sum();
    if total_input_blocks > MAX_INPUT_BLOCKS as u64 {
        let total_bytes: u64 = sources.iter().map(|s| s.length).sum();
        // Smallest multiple-of-4 slice_size keeping total slices <= MAX_INPUT_BLOCKS.
        let raw = total_bytes.div_ceil(MAX_INPUT_BLOCKS as u64);
        let suggested = raw.div_ceil(4) * 4;
        return Err(Par2Error::TooManyInputBlocks {
            count: total_input_blocks,
            slice_size: opts.slice_size,
            suggested,
        });
    }

    let file_ids: Vec<Md5Hash> = sources.iter().map(|s| s.file_id).collect();
    let MainPacket {
        bytes: main_bytes,
        set_id_hash,
    } = build_main_packet(opts.slice_size, &file_ids);

    let mut critical_packets: Vec<u8> = Vec::new();
    critical_packets.extend_from_slice(&main_bytes);
    for src in sources {
        critical_packets.extend_from_slice(&build_file_desc_packet(&set_id_hash, src));
        critical_packets.extend_from_slice(&build_file_verify_packet(&set_id_hash, src));
    }
    let creator_pkt = build_creator_packet(&set_id_hash);
    critical_packets.extend_from_slice(&creator_pkt);
    if !opts.comments.is_empty() {
        critical_packets.extend_from_slice(&build_comment_packets(&set_id_hash, &opts.comments));
    }

    write_atomic(&opts.output, &critical_packets)?;
    if let Some(r) = reporter {
        r.on_event(ProgressEvent::IndexWritten { path: &opts.output });
    }
    let mut written = vec![opts.output.clone()];

    if opts.recovery_block_count > 0 {
        let sizes = resolve_volume_sizes(&opts.volume_scheme, opts.recovery_block_count)?;
        let total_volumes = sizes.len() as u32;
        let mut first_exp: u32 = 0;
        for (volume_index, count) in sizes.into_iter().enumerate() {
            let volume_index = volume_index as u32;
            // first_exp + count <= recovery_block_count <= 65535, so the u16
            // cast is always safe — we validated the total against
            // MAX_RECOVERY_BLOCKS above.
            let first_exp_u16: u16 = first_exp
                .try_into()
                .expect("first_exp fits in u16 because total <= MAX_RECOVERY_BLOCKS");
            let vol_path = derive_volume_filename(&opts.output, first_exp_u16, count);
            let vol_bytes = build_volume_file(
                &set_id_hash,
                &critical_packets,
                sources,
                opts.slice_size,
                first_exp_u16,
                count,
                volume_index,
                total_volumes,
                reporter,
            )?;
            write_atomic(&vol_path, &vol_bytes)?;
            if let Some(r) = reporter {
                r.on_event(ProgressEvent::VolumeWritten { path: &vol_path });
            }
            written.push(vol_path);
            first_exp += count;
        }
    }

    Ok(written)
}

/// Materialise a [`VolumeScheme`] into a concrete list of per-volume block
/// counts. The sum is guaranteed to equal `total` on success.
fn resolve_volume_sizes(scheme: &VolumeScheme, total: u32) -> Result<Vec<u32>> {
    match scheme {
        VolumeScheme::Single => Ok(vec![total]),
        VolumeScheme::Exponential => Ok(exponential_split(total)),
        VolumeScheme::Uniform { count } => uniform_split(total, *count),
        VolumeScheme::Limited {
            max_blocks_per_volume,
            inner,
        } => {
            if *max_blocks_per_volume == 0 {
                return Err(Par2Error::InvalidVolumeScheme(
                    "Limited scheme requires max_blocks_per_volume > 0".into(),
                ));
            }
            let inner_sizes = resolve_volume_sizes(inner, total)?;
            Ok(cap_volume_sizes(&inner_sizes, *max_blocks_per_volume))
        }
        VolumeScheme::Explicit(sizes) => {
            if sizes.is_empty() {
                return Err(Par2Error::InvalidVolumeScheme(
                    "explicit scheme requires at least one volume".into(),
                ));
            }
            if sizes.contains(&0) {
                return Err(Par2Error::InvalidVolumeScheme(
                    "explicit scheme volumes must each have >0 recovery blocks".into(),
                ));
            }
            let sum: u64 = sizes.iter().map(|&n| n as u64).sum();
            if sum != total as u64 {
                return Err(Par2Error::InvalidVolumeScheme(format!(
                    "explicit volume sizes sum to {sum}, expected {total}"
                )));
            }
            Ok(sizes.clone())
        }
    }
}

/// par2cmdline-style exponential distribution: 1, 1, 2, 4, 8, 16, …, with the
/// final volume holding any remainder so the sum equals `total`. Returns an
/// empty vector when `total == 0` (caller should already have early-returned).
fn exponential_split(total: u32) -> Vec<u32> {
    if total == 0 {
        return Vec::new();
    }
    let mut sizes = Vec::new();
    let mut remaining = total;
    let mut next: u32 = 1;
    while remaining > 0 {
        let take = next.min(remaining);
        sizes.push(take);
        remaining -= take;
        // After the first two volumes, double the capacity each step:
        // produces the canonical 1, 1, 2, 4, 8, 16, … sequence.
        if sizes.len() >= 2 {
            next = next.saturating_mul(2);
        }
    }
    sizes
}

/// par2cmdline `-u` distribution: split `total` blocks across `count` volumes
/// as evenly as possible. The first `count - 1` volumes each get
/// `total / count` blocks; the last volume absorbs the remainder.
fn uniform_split(total: u32, count: u32) -> Result<Vec<u32>> {
    if count == 0 {
        return Err(Par2Error::InvalidVolumeScheme(
            "Uniform scheme requires count > 0".into(),
        ));
    }
    if count > total {
        return Err(Par2Error::InvalidVolumeScheme(format!(
            "Uniform scheme count ({count}) exceeds recovery block total ({total})"
        )));
    }
    let base = total / count;
    let remainder = total - base * count;
    let mut sizes = Vec::with_capacity(count as usize);
    for _ in 0..(count - 1) {
        sizes.push(base);
    }
    sizes.push(base + remainder);
    Ok(sizes)
}

/// Cap each entry of `sizes` at `cap`. Any over-cap entry is split into
/// consecutive `cap`-sized volumes plus a trailing remainder. Preserves the
/// total sum.
fn cap_volume_sizes(sizes: &[u32], cap: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(sizes.len());
    for &n in sizes {
        if n <= cap {
            out.push(n);
        } else {
            let mut remaining = n;
            while remaining > cap {
                out.push(cap);
                remaining -= cap;
            }
            if remaining > 0 {
                out.push(remaining);
            }
        }
    }
    out
}

/// Legacy entrypoint kept so Phase-3-era tests still build. Writes the index
/// file only (no recovery slices). Prefer `run_create` for new code.
pub fn write_index_file(opts: &CreateOptions, sources: &[SourceFile]) -> Result<PathBuf> {
    let no_recovery = CreateOptions {
        output: opts.output.clone(),
        slice_size: opts.slice_size,
        recovery_block_count: 0,
        volume_scheme: VolumeScheme::Single,
        comments: opts.comments.clone(),
    };
    let files = run_create(&no_recovery, sources)?;
    Ok(files
        .into_iter()
        .next()
        .expect("run_create returns at least the index file"))
}

/// Produce the bytes for a `.vol*.par2` file: critical packets, then the
/// recovery packets in exponent order, then critical packets again (the
/// upstream convention — improves the chance that the index can be
/// reconstructed from any single volume file alone).
#[allow(clippy::too_many_arguments)]
fn build_volume_file(
    set_id_hash: &Md5Hash,
    critical_packets: &[u8],
    sources: &[SourceFile],
    slice_size: u64,
    first_exponent: u16,
    recovery_block_count: u32,
    volume_index: u32,
    total_volumes: u32,
    reporter: Option<&dyn ProgressReporter>,
) -> Result<Vec<u8>> {
    let slice_size_usize: usize = slice_size
        .try_into()
        .map_err(|_| Par2Error::InvalidSliceSize(slice_size))?;
    let recovery_count = recovery_block_count as usize;

    // 1. Compute the flat input-block list across all sources.
    let mut input_blocks: Vec<(usize, u64)> = Vec::new();
    for (file_idx, src) in sources.iter().enumerate() {
        for slice_idx in 0..src.slice_checksums.len() {
            input_blocks.push((file_idx, slice_idx as u64));
        }
    }

    let input_blocks_total = input_blocks.len() as u64;
    if let Some(r) = reporter {
        r.on_event(ProgressEvent::EncodeStarted {
            volume_index,
            total_volumes,
            input_blocks: input_blocks_total,
            recovery_blocks: recovery_block_count,
        });
    }
    let progress_stride = tick_stride(input_blocks_total);

    // 2. Initialise RS encoder + dispatch path.
    let rs = RsEncoder::new(
        input_blocks.len() as u32,
        first_exponent,
        recovery_block_count,
    );
    let dispatch = detect_dispatch();

    // 3. Allocate one zeroed buffer per recovery block. Total memory:
    //    recovery_count · slice_size. PAR2's defaults keep this in the low MB.
    let mut recovery_buffers: Vec<Vec<u8>> = (0..recovery_count)
        .map(|_| vec![0u8; slice_size_usize])
        .collect();

    // 4. Stream input blocks one slice at a time, accumulating into all
    //    recovery buffers. Each file is opened (and mmap'd) once. With mmap
    //    the kernel page cache typically still has the bytes warm from the
    //    scan pass, so this loop pays no second physical read.
    let mut slice_buf = vec![0u8; slice_size_usize];
    let mut current_file: Option<(usize, SourceReader)> = None;
    let workers = rayon::current_num_threads().max(1);
    let chunk_len = recovery_count.div_ceil(workers).max(1);

    for (input_idx, (file_idx, slice_idx)) in input_blocks.iter().enumerate() {
        // (Re)open the reader on file boundaries.
        let switched_file = !matches!(&current_file, Some((idx, _)) if *idx == *file_idx);
        if switched_file {
            let reader = SourceReader::open(&to_long_path(&sources[*file_idx].path))?;
            current_file = Some((*file_idx, reader));
        }
        let reader = &mut current_file.as_mut().unwrap().1;

        reader.read_slice(*slice_idx as usize, &mut slice_buf)?;

        // Precompute one SIMD lookup table per recovery row for THIS input
        // slice's column of the RS matrix. Previously the table was rebuilt
        // inside `gf_mul_xor_dispatch` on every (input, recovery) iteration,
        // i.e. `input_count * recovery_count` times — the rebuild dominates
        // the SIMD multiply-XOR for the workloads we care about.
        //
        // The build itself is parallelised because there's no data dependency
        // across recovery rows. For typical create jobs (R = 50..500) this
        // amortises perfectly across rayon workers.
        let coeff_tables: Vec<CoeffSimdTables> = (0..recovery_count)
            .into_par_iter()
            .map(|r_idx| {
                let coeff = rs.coefficient(r_idx, input_idx);
                CoeffSimdTables::new(dispatch, coeff)
            })
            .collect();

        // Accumulate into every recovery buffer. Each buffer is disjoint, so
        // rayon hands out exclusive `&mut [u8]` slots with zero locking.
        // `coeff_tables` and `slice_buf` are read-only here.
        //
        // Chunked iteration keeps task count == thread count and lets each
        // task run a tight serial inner SIMD loop.
        let slice_ref: &[u8] = &slice_buf;
        let tables_ref: &[CoeffSimdTables] = &coeff_tables;
        recovery_buffers
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base = chunk_idx * chunk_len;
                for (offset, out) in chunk.iter_mut().enumerate() {
                    let r_idx = base + offset;
                    gf_mul_xor_with_tables(&tables_ref[r_idx], slice_ref, out);
                }
            });

        if let Some(r) = reporter {
            let done = (input_idx as u64) + 1;
            if done == input_blocks_total || done % progress_stride == 0 {
                r.on_event(ProgressEvent::EncodeProgress {
                    volume_index,
                    input_block_done: done,
                    input_blocks: input_blocks_total,
                });
            }
        }
    }
    drop(current_file);

    // 5. Build recovery packets in exponent order.
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(critical_packets);
    for (r_idx, buf) in recovery_buffers.iter().enumerate() {
        let exp = rs.recovery_exponents[r_idx];
        let pkt = build_recovery_packet(set_id_hash, exp, buf);
        out.extend_from_slice(&pkt);
    }
    out.extend_from_slice(critical_packets);

    // Sanity: silences `dispatch` going unused on exotic targets.
    let _ = (dispatch, &Dispatch::Scalar);

    if let Some(r) = reporter {
        r.on_event(ProgressEvent::EncodeCompleted { volume_index });
    }

    Ok(out)
}

/// Build a volume filename like `recovery.vol0+4.par2` from a base index path
/// `recovery.par2` and exponent range.
fn derive_volume_filename(index_path: &Path, first_exponent: u16, count: u32) -> PathBuf {
    let stem = index_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("par2rust");
    let extension = index_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("par2");
    let parent = index_path.parent().unwrap_or_else(|| Path::new(""));
    parent.join(format!("{stem}.vol{first_exponent}+{count}.{extension}"))
}

fn read_full<R: Read>(reader: &mut R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let path = to_long_path(path);
    let mut tmp = path.clone();
    let mut name = tmp
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".par2rust.tmp");
    tmp.set_file_name(name);

    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// On Windows, prefix output paths with `\\?\` so the Win32 layer skips its
/// 260-character `MAX_PATH` check. The Rust stdlib already calls the wide-char
/// APIs (`CreateFileW`, etc.) under the hood, so this is the only piece of
/// Windows path handling par2rust needs.
///
/// On non-Windows platforms this is a no-op pass-through.
#[cfg(windows)]
fn to_long_path(path: &Path) -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::new());
    apply_long_path_prefix(path, &cwd)
}

#[cfg(not(windows))]
fn to_long_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}

/// Pure path-prefixing logic, factored out so it can be unit-tested on any
/// platform. Used by [`to_long_path`] on Windows.
///
/// - Verbatim (`\\?\`) and device (`\\.\`) prefixes are returned unchanged.
/// - Relative paths are joined onto `cwd` before being prefixed.
/// - Absolute paths get the prefix applied directly.
#[cfg_attr(not(windows), allow(dead_code))]
fn apply_long_path_prefix(path: &Path, cwd: &Path) -> PathBuf {
    let bytes = path.as_os_str().as_encoded_bytes();
    if bytes.starts_with(br"\\?\") || bytes.starts_with(br"\\.\") {
        return path.to_path_buf();
    }
    let abs = if path.is_relative() {
        cwd.join(path)
    } else {
        path.to_path_buf()
    };
    let mut prefixed = OsString::from(r"\\?\");
    prefixed.push(abs.as_os_str());
    PathBuf::from(prefixed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::{ProgressEvent, ProgressReporter};
    use std::sync::Mutex;
    use tempfile::tempdir;

    #[derive(Debug, PartialEq)]
    enum Phase {
        ScanStart,
        ScanProgress(u64, u64),
        ScanEnd,
        EncodeStart(u32, u32, u64, u32),
        EncodeProgress(u32, u64, u64),
        EncodeEnd(u32),
        IndexWritten,
        VolumeWritten,
    }

    #[derive(Default)]
    struct CaptureReporter {
        events: Mutex<Vec<Phase>>,
    }

    impl ProgressReporter for CaptureReporter {
        fn on_event(&self, event: ProgressEvent<'_>) {
            let mut log = self.events.lock().unwrap();
            log.push(match event {
                ProgressEvent::ScanStarted { .. } => Phase::ScanStart,
                ProgressEvent::ScanProgress {
                    slices_done,
                    total_slices,
                    ..
                } => Phase::ScanProgress(slices_done, total_slices),
                ProgressEvent::ScanCompleted { .. } => Phase::ScanEnd,
                ProgressEvent::EncodeStarted {
                    volume_index,
                    total_volumes,
                    input_blocks,
                    recovery_blocks,
                } => Phase::EncodeStart(volume_index, total_volumes, input_blocks, recovery_blocks),
                ProgressEvent::EncodeProgress {
                    volume_index,
                    input_block_done,
                    input_blocks,
                } => Phase::EncodeProgress(volume_index, input_block_done, input_blocks),
                ProgressEvent::EncodeCompleted { volume_index } => Phase::EncodeEnd(volume_index),
                ProgressEvent::IndexWritten { .. } => Phase::IndexWritten,
                ProgressEvent::VolumeWritten { .. } => Phase::VolumeWritten,
            });
        }
    }

    fn make_source(dir: &Path, name: &str, content: &[u8], slice: u64) -> SourceFile {
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(content).unwrap();
        f.flush().unwrap();
        SourceFile::scan(&p, name.as_bytes().to_vec(), slice).unwrap()
    }

    #[test]
    fn index_only_run_produces_single_file() {
        let dir = tempdir().unwrap();
        let src = make_source(dir.path(), "a.bin", b"hello par2 world", 4);
        let out = dir.path().join("recovery.par2");

        let files = run_create(
            &CreateOptions {
                output: out.clone(),
                slice_size: 4,
                recovery_block_count: 0,
                volume_scheme: VolumeScheme::Single,
                comments: Vec::new(),
            },
            &[src],
        )
        .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], out);
    }

    #[test]
    fn run_with_recovery_produces_index_and_volume() {
        let dir = tempdir().unwrap();
        let src = make_source(dir.path(), "a.bin", b"hello par2 world!!!!", 4);
        let out = dir.path().join("recovery.par2");
        let vol_expected = dir.path().join("recovery.vol0+2.par2");

        let files = run_create(
            &CreateOptions {
                output: out.clone(),
                slice_size: 4,
                recovery_block_count: 2,
                volume_scheme: VolumeScheme::Single,
                comments: Vec::new(),
            },
            &[src],
        )
        .unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0], out);
        assert_eq!(files[1], vol_expected);
        assert!(out.exists());
        assert!(vol_expected.exists());
    }

    #[test]
    fn errors_when_no_input_files() {
        let dir = tempdir().unwrap();
        let out = dir.path().join("r.par2");
        let err = run_create(
            &CreateOptions {
                output: out,
                slice_size: 4,
                recovery_block_count: 0,
                volume_scheme: VolumeScheme::Single,
                comments: Vec::new(),
            },
            &[],
        )
        .unwrap_err();
        matches!(err, Par2Error::NoInputFiles);
    }

    #[test]
    fn errors_when_total_slices_exceed_max_input_blocks() {
        use crate::source::SliceChecksum;
        let dir = tempdir().unwrap();
        let out = dir.path().join("r.par2");
        let slice_size: u64 = 4;
        let blocks = (MAX_INPUT_BLOCKS as usize) + 1;
        // Fabricate a SourceFile directly — scanning a multi-GB synthetic
        // file would dominate test time. The encoder never runs because
        // run_create_with_progress rejects up front.
        let fake = SourceFile {
            name: b"big.bin".to_vec(),
            path: dir.path().join("big.bin"),
            length: blocks as u64 * slice_size,
            hash_full: [0u8; 16],
            hash16k: [0u8; 16],
            file_id: [0u8; 16],
            slice_checksums: vec![
                SliceChecksum {
                    md5: [0u8; 16],
                    crc32: 0,
                };
                blocks
            ],
        };
        let err = run_create(
            &CreateOptions {
                output: out,
                slice_size,
                recovery_block_count: 1,
                volume_scheme: VolumeScheme::Single,
                comments: Vec::new(),
            },
            &[fake],
        )
        .unwrap_err();
        match err {
            Par2Error::TooManyInputBlocks {
                count,
                slice_size: ss,
                suggested,
            } => {
                assert_eq!(count, blocks as u64);
                assert_eq!(ss, slice_size);
                assert!(suggested > slice_size);
                assert_eq!(suggested % 4, 0);
            }
            other => panic!("expected TooManyInputBlocks, got {other:?}"),
        }
    }

    #[test]
    fn errors_when_recovery_count_exceeds_limit() {
        let dir = tempdir().unwrap();
        let src = make_source(dir.path(), "a.bin", b"x", 4);
        let out = dir.path().join("r.par2");
        let err = run_create(
            &CreateOptions {
                output: out,
                slice_size: 4,
                recovery_block_count: MAX_RECOVERY_BLOCKS + 1,
                volume_scheme: VolumeScheme::Single,
                comments: Vec::new(),
            },
            &[src],
        )
        .unwrap_err();
        matches!(err, Par2Error::TooManyRecoveryBlocks(_));
    }

    #[test]
    fn legacy_write_index_file_still_works() {
        let dir = tempdir().unwrap();
        let src = make_source(dir.path(), "a.bin", b"hello par2", 4);
        let out = dir.path().join("recovery.par2");
        let returned = write_index_file(
            &CreateOptions {
                output: out.clone(),
                slice_size: 4,
                recovery_block_count: 0,
                volume_scheme: VolumeScheme::Single,
                comments: Vec::new(),
            },
            &[src],
        )
        .unwrap();
        assert_eq!(returned, out);
        assert!(out.exists());
    }

    // ---------------------------------------------------------------
    // Multi-volume distribution
    // ---------------------------------------------------------------

    #[test]
    fn exponential_split_total_one_is_single_volume() {
        assert_eq!(exponential_split(1), vec![1]);
    }

    #[test]
    fn exponential_split_total_two_is_two_volumes_of_one() {
        assert_eq!(exponential_split(2), vec![1, 1]);
    }

    #[test]
    fn exponential_split_total_four_doubles_after_pair() {
        assert_eq!(exponential_split(4), vec![1, 1, 2]);
    }

    #[test]
    fn exponential_split_total_ten_remainder_clamps() {
        // 1 + 1 + 2 + 4 = 8, remaining 2 → final volume = 2.
        assert_eq!(exponential_split(10), vec![1, 1, 2, 4, 2]);
    }

    #[test]
    fn exponential_split_sums_to_total_for_many_sizes() {
        for total in [1u32, 2, 3, 7, 50, 100, 1000, 65_535] {
            let sizes = exponential_split(total);
            let sum: u64 = sizes.iter().map(|&n| n as u64).sum();
            assert_eq!(sum, total as u64, "scheme didn't sum to {total}: {sizes:?}");
            assert!(
                sizes.iter().all(|&n| n > 0),
                "scheme produced zero-sized volume: {sizes:?}"
            );
        }
    }

    #[test]
    fn exponential_scheme_produces_expected_volume_files() {
        let dir = tempdir().unwrap();
        // 4096-byte slices; payload fills enough slices for 10 recovery blocks
        // to be meaningful. The actual block math doesn't matter for the
        // filename-layout assertion below.
        let payload = vec![0xABu8; 4096 * 4];
        let src = make_source(dir.path(), "a.bin", &payload, 4096);
        let out = dir.path().join("recovery.par2");

        let files = run_create(
            &CreateOptions {
                output: out.clone(),
                slice_size: 4096,
                recovery_block_count: 10,
                volume_scheme: VolumeScheme::Exponential,
                comments: Vec::new(),
            },
            &[src],
        )
        .unwrap();

        // index + 5 volume files (1+1+2+4+2).
        let names: Vec<String> = files
            .iter()
            .skip(1) // drop the index
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec![
                "recovery.vol0+1.par2",
                "recovery.vol1+1.par2",
                "recovery.vol2+2.par2",
                "recovery.vol4+4.par2",
                "recovery.vol8+2.par2",
            ]
        );
    }

    #[test]
    fn explicit_scheme_rejects_sum_mismatch() {
        let dir = tempdir().unwrap();
        let src = make_source(dir.path(), "a.bin", b"hello par2 world!!!!", 4);
        let out = dir.path().join("recovery.par2");
        let err = run_create(
            &CreateOptions {
                output: out,
                slice_size: 4,
                recovery_block_count: 5,
                volume_scheme: VolumeScheme::Explicit(vec![2, 2]),
                comments: Vec::new(),
            },
            &[src],
        )
        .unwrap_err();
        assert!(
            matches!(err, Par2Error::InvalidVolumeScheme(_)),
            "expected InvalidVolumeScheme, got: {err:?}"
        );
    }

    #[test]
    fn explicit_scheme_rejects_zero_sized_volume() {
        let dir = tempdir().unwrap();
        let src = make_source(dir.path(), "a.bin", b"hello par2 world!!!!", 4);
        let out = dir.path().join("recovery.par2");
        let err = run_create(
            &CreateOptions {
                output: out,
                slice_size: 4,
                recovery_block_count: 4,
                volume_scheme: VolumeScheme::Explicit(vec![2, 0, 2]),
                comments: Vec::new(),
            },
            &[src],
        )
        .unwrap_err();
        assert!(matches!(err, Par2Error::InvalidVolumeScheme(_)));
    }

    #[test]
    fn explicit_scheme_honours_caller_layout() {
        let dir = tempdir().unwrap();
        let payload = vec![0x5Au8; 4096 * 3];
        let src = make_source(dir.path(), "a.bin", &payload, 4096);
        let out = dir.path().join("recovery.par2");

        let files = run_create(
            &CreateOptions {
                output: out.clone(),
                slice_size: 4096,
                recovery_block_count: 6,
                volume_scheme: VolumeScheme::Explicit(vec![3, 3]),
                comments: Vec::new(),
            },
            &[src],
        )
        .unwrap();
        let names: Vec<String> = files
            .iter()
            .skip(1)
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["recovery.vol0+3.par2", "recovery.vol3+3.par2"]);
    }

    // ---------------------------------------------------------------
    // Uniform (-u) and Limited (-l) distribution
    // ---------------------------------------------------------------

    #[test]
    fn uniform_split_divides_evenly() {
        assert_eq!(uniform_split(10, 5).unwrap(), vec![2, 2, 2, 2, 2]);
    }

    #[test]
    fn uniform_split_dumps_remainder_into_last_volume() {
        // 13 / 4 = 3 with remainder 1 → 3, 3, 3, 4.
        assert_eq!(uniform_split(13, 4).unwrap(), vec![3, 3, 3, 4]);
    }

    #[test]
    fn uniform_split_count_one_matches_single() {
        assert_eq!(uniform_split(42, 1).unwrap(), vec![42]);
    }

    #[test]
    fn uniform_split_rejects_zero_count() {
        let err = uniform_split(10, 0).unwrap_err();
        assert!(matches!(err, Par2Error::InvalidVolumeScheme(_)));
    }

    #[test]
    fn uniform_split_rejects_count_greater_than_total() {
        let err = uniform_split(3, 5).unwrap_err();
        assert!(matches!(err, Par2Error::InvalidVolumeScheme(_)));
    }

    #[test]
    fn uniform_split_sums_to_total_across_sizes() {
        for (total, count) in [(1u32, 1), (7, 3), (50, 7), (100, 10), (65_535, 17)] {
            let sizes = uniform_split(total, count).unwrap();
            assert_eq!(sizes.len() as u32, count);
            let sum: u64 = sizes.iter().map(|&n| n as u64).sum();
            assert_eq!(sum, total as u64, "sum mismatch for ({total},{count})");
        }
    }

    #[test]
    fn cap_volume_sizes_leaves_in_bounds_entries_alone() {
        assert_eq!(cap_volume_sizes(&[1, 2, 3], 4), vec![1, 2, 3]);
    }

    #[test]
    fn cap_volume_sizes_splits_oversize_entries() {
        // cap = 3; the 10 splits to 3, 3, 3, 1.
        assert_eq!(cap_volume_sizes(&[2, 10, 1], 3), vec![2, 3, 3, 3, 1, 1]);
    }

    #[test]
    fn cap_volume_sizes_clean_split_no_remainder() {
        // cap = 4; 8 splits exactly into 4, 4 (no trailing remainder).
        assert_eq!(cap_volume_sizes(&[8], 4), vec![4, 4]);
    }

    #[test]
    fn limited_scheme_caps_exponential_growth() {
        let scheme = VolumeScheme::Limited {
            max_blocks_per_volume: 3,
            inner: Box::new(VolumeScheme::Exponential),
        };
        // Exponential(20) = [1, 1, 2, 4, 8, 4]; cap = 3 → [1, 1, 2, 3, 1, 3, 3, 2, 3, 1].
        let sizes = resolve_volume_sizes(&scheme, 20).unwrap();
        assert!(sizes.iter().all(|&n| n <= 3 && n > 0));
        let sum: u32 = sizes.iter().sum();
        assert_eq!(sum, 20);
    }

    #[test]
    fn limited_scheme_caps_uniform_layout() {
        let scheme = VolumeScheme::Limited {
            max_blocks_per_volume: 4,
            inner: Box::new(VolumeScheme::Uniform { count: 2 }),
        };
        // Uniform(20, 2) = [10, 10]; cap = 4 → [4, 4, 2, 4, 4, 2].
        let sizes = resolve_volume_sizes(&scheme, 20).unwrap();
        assert_eq!(sizes, vec![4, 4, 2, 4, 4, 2]);
    }

    #[test]
    fn limited_scheme_rejects_zero_cap() {
        let scheme = VolumeScheme::Limited {
            max_blocks_per_volume: 0,
            inner: Box::new(VolumeScheme::Exponential),
        };
        let err = resolve_volume_sizes(&scheme, 10).unwrap_err();
        assert!(matches!(err, Par2Error::InvalidVolumeScheme(_)));
    }

    #[test]
    fn uniform_scheme_produces_expected_volume_files() {
        let dir = tempdir().unwrap();
        let payload = vec![0xCDu8; 4096 * 4];
        let src = make_source(dir.path(), "a.bin", &payload, 4096);
        let out = dir.path().join("recovery.par2");

        let files = run_create(
            &CreateOptions {
                output: out.clone(),
                slice_size: 4096,
                recovery_block_count: 10,
                volume_scheme: VolumeScheme::Uniform { count: 4 },
                comments: Vec::new(),
            },
            &[src],
        )
        .unwrap();

        // Uniform(10, 4) = [2, 2, 2, 4].
        let names: Vec<String> = files
            .iter()
            .skip(1)
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec![
                "recovery.vol0+2.par2",
                "recovery.vol2+2.par2",
                "recovery.vol4+2.par2",
                "recovery.vol6+4.par2",
            ]
        );
    }

    #[test]
    fn create_options_default_is_single_scheme() {
        let opts = CreateOptions::default();
        assert_eq!(opts.volume_scheme, VolumeScheme::Single);
    }

    // ---------------------------------------------------------------
    // Long-path helper
    // ---------------------------------------------------------------

    #[cfg(not(windows))]
    #[test]
    fn to_long_path_is_passthrough_on_unix() {
        let p = Path::new("/tmp/a.bin");
        assert_eq!(to_long_path(p), PathBuf::from("/tmp/a.bin"));
    }

    /// Verbatim-prefixed paths must round-trip unchanged so we don't end up
    /// double-prefixing like `\\?\\\?\C:\foo`. Runs on every platform since
    /// the byte-level prefix check is OS-independent.
    #[test]
    fn apply_long_path_prefix_leaves_verbatim_paths_unchanged() {
        let cwd = Path::new("/cwd/should/be/ignored");
        let p = Path::new(r"\\?\C:\foo\bar");
        assert_eq!(
            apply_long_path_prefix(p, cwd),
            PathBuf::from(r"\\?\C:\foo\bar")
        );
    }

    /// Device paths (`\\.\PhysicalDrive0`, `\\.\COM1`) are also already in a
    /// form Win32 accepts and must not be wrapped.
    #[test]
    fn apply_long_path_prefix_leaves_device_paths_unchanged() {
        let cwd = Path::new("/cwd");
        let p = Path::new(r"\\.\PhysicalDrive0");
        assert_eq!(
            apply_long_path_prefix(p, cwd),
            PathBuf::from(r"\\.\PhysicalDrive0")
        );
    }

    /// Relative paths must be joined onto the supplied cwd before being
    /// prefixed — `\\?\rel` would be meaningless to Win32. The exact
    /// separator characters depend on the host (`PathBuf::join` keeps the
    /// cwd's separators and introduces its own between segments), so we
    /// assert on substrings rather than a fixed suffix.
    #[test]
    fn apply_long_path_prefix_absolutises_relative_path_against_cwd() {
        let cwd = Path::new("/work/dir");
        let out = apply_long_path_prefix(Path::new("sub/file.par2"), cwd);
        let s = out.to_string_lossy();
        assert!(s.starts_with(r"\\?\"), "missing \\\\?\\ prefix: {s}");
        assert!(s.contains("work"), "cwd not appended: {s}");
        assert!(s.contains("dir"), "cwd not appended: {s}");
        assert!(s.contains("sub"), "relative path not appended: {s}");
        assert!(s.contains("file.par2"), "relative path not appended: {s}");
    }

    /// Already-absolute paths get the prefix applied directly without
    /// touching the cwd.
    #[test]
    fn apply_long_path_prefix_does_not_touch_cwd_for_absolute_path() {
        let cwd = Path::new("/should/not/appear");
        let p = if cfg!(windows) {
            Path::new(r"C:\some\dir\file.par2")
        } else {
            Path::new("/some/dir/file.par2")
        };
        let out = apply_long_path_prefix(p, cwd);
        let s = out.to_string_lossy();
        assert!(s.starts_with(r"\\?\"));
        assert!(
            !s.contains("should/not/appear"),
            "cwd leaked into absolute path: {s}"
        );
    }

    /// Idempotency: applying the helper to its own output is a no-op because
    /// the result already starts with `\\?\`. Guards against accidentally
    /// double-prefixing across nested calls.
    #[test]
    fn apply_long_path_prefix_is_idempotent() {
        let cwd = Path::new("/cwd");
        let first = apply_long_path_prefix(Path::new("a/b/c.par2"), cwd);
        let second = apply_long_path_prefix(&first, cwd);
        assert_eq!(first, second);
    }

    /// Long path (>260 chars) gets prefixed cleanly — the very case the
    /// `\\?\` wrapper exists to enable on Windows.
    #[test]
    fn apply_long_path_prefix_handles_paths_longer_than_max_path() {
        let cwd = Path::new("/base");
        // A 300-char path component, well past Win32's 260-char MAX_PATH.
        let long_segment: String = "a".repeat(300);
        let p = PathBuf::from(format!("dir/{long_segment}.par2"));
        let out = apply_long_path_prefix(&p, cwd);
        let s = out.to_string_lossy();
        assert!(s.starts_with(r"\\?\"), "missing prefix: {s}");
        assert!(s.len() > 260, "expected >260 chars, got {}", s.len());
    }

    #[test]
    fn progress_events_fire_in_order_for_create_with_recovery() {
        let dir = tempdir().unwrap();
        // 4-byte slices, 20-byte file → 5 input blocks. 2 recovery blocks in
        // a single volume.
        let p = dir.path().join("a.bin");
        std::fs::write(&p, b"hello par2 world!!!!").unwrap();
        let out = dir.path().join("recovery.par2");

        let reporter = CaptureReporter::default();
        let src =
            SourceFile::scan_with_progress(&p, b"a.bin".to_vec(), 4, Some(&reporter)).unwrap();
        let files = run_create_with_progress(
            &CreateOptions {
                output: out.clone(),
                slice_size: 4,
                recovery_block_count: 2,
                volume_scheme: VolumeScheme::Single,
                comments: Vec::new(),
            },
            &[src],
            Some(&reporter),
        )
        .unwrap();
        assert_eq!(files.len(), 2);

        let events = reporter.events.lock().unwrap();
        // Scan precedes encode precedes writes.
        let first_encode = events
            .iter()
            .position(|e| matches!(e, Phase::EncodeStart(..)))
            .expect("EncodeStart present");
        let first_index = events
            .iter()
            .position(|e| matches!(e, Phase::IndexWritten))
            .expect("IndexWritten present");
        let first_volume = events
            .iter()
            .position(|e| matches!(e, Phase::VolumeWritten))
            .expect("VolumeWritten present");
        assert!(first_encode > 0);
        // IndexWritten fires before any volume encoding starts.
        assert!(first_index < first_encode);
        // VolumeWritten fires after encoding completes.
        let last_encode_end = events
            .iter()
            .rposition(|e| matches!(e, Phase::EncodeEnd(_)))
            .unwrap();
        assert!(first_volume > last_encode_end);

        // Final encode progress reaches input_blocks total.
        let final_encode_progress = events
            .iter()
            .filter_map(|e| match e {
                Phase::EncodeProgress(_, done, total) => Some((*done, *total)),
                _ => None,
            })
            .next_back()
            .expect("encode progress fired at least once");
        assert_eq!(
            final_encode_progress.0, final_encode_progress.1,
            "last EncodeProgress must reach total"
        );
        // 5 input blocks expected.
        assert_eq!(final_encode_progress.1, 5);

        // Final scan progress reaches total_slices.
        let final_scan_progress = events
            .iter()
            .filter_map(|e| match e {
                Phase::ScanProgress(done, total) => Some((*done, *total)),
                _ => None,
            })
            .next_back()
            .expect("scan progress fired at least once");
        assert_eq!(final_scan_progress.0, final_scan_progress.1);

        // Exactly one EncodeStart/EncodeEnd pair (single volume).
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Phase::EncodeStart(..)))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Phase::EncodeEnd(_)))
                .count(),
            1
        );
    }

    #[test]
    fn run_create_with_and_without_reporter_produce_identical_output() {
        let dir1 = tempdir().unwrap();
        let dir2 = tempdir().unwrap();
        let src1 = make_source(dir1.path(), "a.bin", b"hello par2 world!!!!", 4);
        let src2 = make_source(dir2.path(), "a.bin", b"hello par2 world!!!!", 4);

        let opts1 = CreateOptions {
            output: dir1.path().join("recovery.par2"),
            slice_size: 4,
            recovery_block_count: 2,
            volume_scheme: VolumeScheme::Single,
            comments: Vec::new(),
        };
        let opts2 = CreateOptions {
            output: dir2.path().join("recovery.par2"),
            slice_size: 4,
            recovery_block_count: 2,
            volume_scheme: VolumeScheme::Single,
            comments: Vec::new(),
        };

        let files1 = run_create(&opts1, &[src1]).unwrap();
        let reporter = CaptureReporter::default();
        let files2 = run_create_with_progress(&opts2, &[src2], Some(&reporter)).unwrap();
        assert_eq!(files1.len(), files2.len());
        for (a, b) in files1.iter().zip(files2.iter()) {
            let bytes_a = std::fs::read(a).unwrap();
            let bytes_b = std::fs::read(b).unwrap();
            assert_eq!(bytes_a, bytes_b, "files differ");
        }
    }
}
