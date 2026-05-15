use std::ffi::OsString;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{Par2Error, Result};
use crate::format::Md5Hash;
use crate::galois_simd::{detect_dispatch, gf_mul_xor_dispatch, Dispatch};
use crate::packet::creator::build_creator_packet;
use crate::packet::file_desc::build_file_desc_packet;
use crate::packet::file_verify::build_file_verify_packet;
use crate::packet::main_packet::{build_main_packet, MainPacket};
use crate::packet::recovery::build_recovery_packet;
use crate::reedsolomon::RsEncoder;
use crate::source::SourceFile;

/// Maximum number of input files PAR2 supports.
pub const MAX_FILES: usize = 32_768;
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
}

impl Default for CreateOptions {
    fn default() -> Self {
        Self {
            output: PathBuf::new(),
            slice_size: 4096,
            recovery_block_count: 0,
            volume_scheme: VolumeScheme::Single,
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

    write_atomic(&opts.output, &critical_packets)?;
    let mut written = vec![opts.output.clone()];

    if opts.recovery_block_count > 0 {
        let sizes = resolve_volume_sizes(&opts.volume_scheme, opts.recovery_block_count)?;
        let mut first_exp: u32 = 0;
        for count in sizes {
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
            )?;
            write_atomic(&vol_path, &vol_bytes)?;
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

/// Legacy entrypoint kept so Phase-3-era tests still build. Writes the index
/// file only (no recovery slices). Prefer `run_create` for new code.
pub fn write_index_file(opts: &CreateOptions, sources: &[SourceFile]) -> Result<PathBuf> {
    let no_recovery = CreateOptions {
        output: opts.output.clone(),
        slice_size: opts.slice_size,
        recovery_block_count: 0,
        volume_scheme: VolumeScheme::Single,
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
fn build_volume_file(
    set_id_hash: &Md5Hash,
    critical_packets: &[u8],
    sources: &[SourceFile],
    slice_size: u64,
    first_exponent: u16,
    recovery_block_count: u32,
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
    //    recovery buffers. Each file is opened once; slices are read in order.
    let mut slice_buf = vec![0u8; slice_size_usize];
    let mut current_file: Option<(usize, BufReader<File>)> = None;

    for (input_idx, (file_idx, slice_idx)) in input_blocks.iter().enumerate() {
        // Open or reuse the reader for this file.
        let reader = match &mut current_file {
            Some((idx, r)) if *idx == *file_idx => r,
            _ => {
                let f = File::open(to_long_path(&sources[*file_idx].path))?;
                current_file = Some((*file_idx, BufReader::with_capacity(1 << 16, f)));
                let (_, r) = current_file.as_mut().unwrap();
                r
            }
        };

        // Seek to the slice (BufReader's internal seek discards its buffer for
        // any non-trivial movement; explicit seek is the most correct way).
        let offset = slice_size * slice_idx;
        reader.seek(SeekFrom::Start(offset))?;
        let filled = read_full(reader, &mut slice_buf)?;
        if filled < slice_buf.len() {
            slice_buf[filled..].fill(0);
        }

        // Accumulate into every recovery buffer.
        for (r_idx, out) in recovery_buffers.iter_mut().enumerate() {
            let coeff = rs.coefficient(r_idx, input_idx);
            gf_mul_xor_dispatch(dispatch, coeff, &slice_buf, out);
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
    use tempfile::tempdir;

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
            },
            &[],
        )
        .unwrap_err();
        matches!(err, Par2Error::NoInputFiles);
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
}
