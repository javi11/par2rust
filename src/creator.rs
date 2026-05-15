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
    let MainPacket { bytes: main_bytes, set_id_hash } =
        build_main_packet(opts.slice_size, &file_ids);

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
        let vol_path =
            derive_volume_filename(&opts.output, 0, opts.recovery_block_count);
        let vol_bytes = build_volume_file(
            &set_id_hash,
            &critical_packets,
            sources,
            opts.slice_size,
            0,
            opts.recovery_block_count,
        )?;
        write_atomic(&vol_path, &vol_bytes)?;
        written.push(vol_path);
    }

    Ok(written)
}

/// Legacy entrypoint kept so Phase-3-era tests still build. Writes the index
/// file only (no recovery slices). Prefer `run_create` for new code.
pub fn write_index_file(opts: &CreateOptions, sources: &[SourceFile]) -> Result<PathBuf> {
    let no_recovery = CreateOptions {
        output: opts.output.clone(),
        slice_size: opts.slice_size,
        recovery_block_count: 0,
    };
    let files = run_create(&no_recovery, sources)?;
    Ok(files.into_iter().next().expect("run_create returns at least the index file"))
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
    let rs = RsEncoder::new(input_blocks.len() as u32, first_exponent, recovery_block_count);
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
                let f = File::open(&sources[*file_idx].path)?;
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
    for r_idx in 0..recovery_count {
        let exp = rs.recovery_exponents[r_idx];
        let pkt = build_recovery_packet(set_id_hash, exp, &recovery_buffers[r_idx]);
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
    let mut tmp = path.to_path_buf();
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
    std::fs::rename(&tmp, path)?;
    Ok(())
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
            &CreateOptions { output: out.clone(), slice_size: 4, recovery_block_count: 0 },
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
            &CreateOptions { output: out.clone(), slice_size: 4, recovery_block_count: 2 },
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
            &CreateOptions { output: out, slice_size: 4, recovery_block_count: 0 },
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
            &CreateOptions { output: out.clone(), slice_size: 4, recovery_block_count: 0 },
            &[src],
        )
        .unwrap();
        assert_eq!(returned, out);
        assert!(out.exists());
    }
}
