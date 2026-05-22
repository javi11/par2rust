use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::Mmap;
use rayon::prelude::*;

use crate::error::{Par2Error, Result};
use crate::format::{md5_of, Md5Hash};
use crate::md5_impl::{self, Md5Ctx};
use crate::progress::{tick_stride, ProgressEvent, ProgressReporter};

/// Length of the "first chunk" hashed into `hash16k`. The PAR2 spec hard-codes
/// this to 16 384 bytes — for files smaller than that, only the available bytes
/// are hashed.
pub const HASH16K_LEN: usize = 16 * 1024;

/// Fallback BufReader capacity when the file cannot be memory-mapped (e.g.
/// some Windows network shares or unusual file systems). 4 MiB keeps the
/// sequential read close to disk throughput without spending much RAM.
const FALLBACK_READ_CAPACITY: usize = 4 * 1024 * 1024;

/// One slice of a source file. Slices are fixed-size (== `slice_size`) except
/// possibly the trailing slice of a file, which is zero-padded up to `slice_size`
/// before its checksum is computed (PAR2 spec, §"File Slice Checksum Packet").
#[derive(Debug, Clone)]
pub struct SliceChecksum {
    pub md5: Md5Hash,
    pub crc32: u32,
}

/// Aggregate metadata for one input file, the way it lands in the packets:
/// `FileDescriptionPacket` consumes `file_id`/`hash_full`/`hash16k`/`length`/`name`,
/// and `FileVerificationPacket` consumes `slice_checksums`.
#[derive(Debug, Clone)]
pub struct SourceFile {
    /// Display name as it appears in the packet (no path component, no NUL terminator).
    pub name: Vec<u8>,
    /// On-disk path used to read the data.
    pub path: PathBuf,
    pub length: u64,
    pub hash_full: Md5Hash,
    pub hash16k: Md5Hash,
    pub file_id: Md5Hash,
    pub slice_checksums: Vec<SliceChecksum>,
}

impl SourceFile {
    /// Read a file from disk and produce its full PAR2 metadata. `display_name`
    /// is what will be embedded in the `FileDescriptionPacket` — typically the
    /// filename component relative to a chosen basepath.
    pub fn scan(path: &Path, display_name: Vec<u8>, slice_size: u64) -> Result<Self> {
        Self::scan_with_progress(path, display_name, slice_size, None)
    }

    /// Like [`SourceFile::scan`] but emits progress events through `reporter`.
    /// Pass `None` for behaviour identical to `scan`.
    pub fn scan_with_progress(
        path: &Path,
        display_name: Vec<u8>,
        slice_size: u64,
        reporter: Option<&dyn ProgressReporter>,
    ) -> Result<Self> {
        if display_name.is_empty() || display_name.contains(&0) {
            return Err(Par2Error::InvalidFileName(path.to_path_buf()));
        }
        if slice_size == 0 || !slice_size.is_multiple_of(4) {
            return Err(Par2Error::InvalidSliceSize(slice_size));
        }

        let length = std::fs::metadata(path)?.len();
        if length == 0 {
            return Err(Par2Error::EmptyFile(path.to_path_buf()));
        }

        let slice_size_usize: usize = slice_size
            .try_into()
            .map_err(|_| Par2Error::InvalidSliceSize(slice_size))?;
        let total_slices = length.div_ceil(slice_size);
        if let Some(r) = reporter {
            r.on_event(ProgressEvent::ScanStarted { path, total_slices });
        }
        let stride = tick_stride(total_slices);

        let file = File::open(path)?;

        // Try mmap first. The per-slice MD5+CRC32 work parallelises perfectly
        // over `mmap.par_chunks(slice_size)`, and the kernel page cache lets
        // the subsequent encode pass re-read the same bytes without a second
        // physical I/O. Fall back to a streamed BufReader path on platforms
        // / file systems where mmap is unavailable.
        //
        // SAFETY: callers must not concurrently write the file. PAR2 input is
        // read-only by definition; if a writer mutates the file under us the
        // resulting hashes are simply wrong (same outcome as a BufReader that
        // races with a writer), not memory-unsafe in the Rust sense.
        let mmap_result = unsafe { Mmap::map(&file) };

        let (slice_checksums, hash_full, hash16k) = match mmap_result {
            Ok(mmap) => scan_via_mmap(
                path,
                length,
                slice_size_usize,
                total_slices,
                stride,
                reporter,
                &mmap,
            ),
            Err(_) => scan_via_reader(
                path,
                length,
                slice_size_usize,
                total_slices,
                stride,
                reporter,
                file,
            )?,
        };

        let file_id = compute_file_id(&hash16k, length, &display_name);

        if let Some(r) = reporter {
            r.on_event(ProgressEvent::ScanCompleted { path });
        }

        Ok(SourceFile {
            name: display_name,
            path: path.to_path_buf(),
            length,
            hash_full,
            hash16k,
            file_id,
            slice_checksums,
        })
    }
}

/// Parallel mmap-backed scan. Per-slice MD5/CRC32 fan out across rayon
/// workers (`par_chunks` hands each thread a contiguous, disjoint slot of the
/// mmap), while the file-wide `hash_full` and `hash16k` run on a separate
/// rayon task via `rayon::join`. MD5 is inherently sequential per-stream, so
/// we can't parallelise within those two hashes — but we can overlap them with
/// the per-slice work.
pub(crate) fn scan_via_mmap(
    path: &Path,
    length: u64,
    slice_size_usize: usize,
    total_slices: u64,
    stride: u64,
    reporter: Option<&dyn ProgressReporter>,
    mmap: &Mmap,
) -> (Vec<SliceChecksum>, Md5Hash, Md5Hash) {
    let bytes: &[u8] = &mmap[..];

    let per_slice = || {
        // `done` counts completed slices across all rayon workers.
        // `last_emitted` gates reporter callbacks so they remain monotonic
        // even though `done` is incremented out-of-order across threads
        // (without this, a faster worker can publish a higher count, then
        // a slower one publishes a lower one — confusing any consumer that
        // assumes monotonic progress).
        let done = AtomicU64::new(0);
        let last_emitted = AtomicU64::new(0);
        bytes
            .par_chunks(slice_size_usize)
            .map(|slice| {
                let cksum = if slice.len() == slice_size_usize {
                    // Common case: full slice → hash the mmap window directly.
                    SliceChecksum {
                        md5: md5_impl::digest(slice),
                        crc32: crc32fast::hash(slice),
                    }
                } else {
                    // Trailing partial slice: zero-pad in a thread-local buffer
                    // (cannot mutate the read-only mmap).
                    let mut buf = vec![0u8; slice_size_usize];
                    buf[..slice.len()].copy_from_slice(slice);
                    SliceChecksum {
                        md5: md5_impl::digest(&buf),
                        crc32: crc32fast::hash(&buf),
                    }
                };

                if let Some(r) = reporter {
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    let is_last = d == total_slices;
                    if is_last || d.is_multiple_of(stride) {
                        // Only emit if `d` is strictly newer than whatever a
                        // sibling worker already published. CAS-loop on
                        // `last_emitted` so concurrent ticks fold into one
                        // monotonic event stream.
                        let mut prev = last_emitted.load(Ordering::Relaxed);
                        loop {
                            if d <= prev {
                                break;
                            }
                            match last_emitted.compare_exchange_weak(
                                prev,
                                d,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => {
                                    r.on_event(ProgressEvent::ScanProgress {
                                        path,
                                        slices_done: d,
                                        total_slices,
                                    });
                                    break;
                                }
                                Err(actual) => prev = actual,
                            }
                        }
                    }
                }

                cksum
            })
            .collect::<Vec<_>>()
    };

    let file_hashes = || {
        let hash_full: Md5Hash = md5_impl::digest(bytes);
        let h16k_end = std::cmp::min(length as usize, HASH16K_LEN);
        let hash16k: Md5Hash = md5_impl::digest(&bytes[..h16k_end]);
        (hash_full, hash16k)
    };

    let (slice_checksums, (hash_full, hash16k)) = rayon::join(per_slice, file_hashes);
    (slice_checksums, hash_full, hash16k)
}

/// Serial BufReader fallback for platforms where mmap fails. Streams the file
/// once with a 4 MiB buffer (vs the previous 64 KiB) so cold reads stay close
/// to the disk's sequential ceiling.
fn scan_via_reader(
    path: &Path,
    length: u64,
    slice_size_usize: usize,
    total_slices: u64,
    stride: u64,
    reporter: Option<&dyn ProgressReporter>,
    file: File,
) -> Result<(Vec<SliceChecksum>, Md5Hash, Md5Hash)> {
    let _ = length; // kept for symmetry with mmap path; not needed here
    let mut reader = BufReader::with_capacity(FALLBACK_READ_CAPACITY, file);

    let mut hash_full_ctx = Md5Ctx::new();
    let mut hash16k_ctx = Md5Ctx::new();
    let mut bytes_into_16k: usize = 0;

    let mut slice_buf = vec![0u8; slice_size_usize];
    let mut slice_checksums: Vec<SliceChecksum> = Vec::new();

    loop {
        let filled = read_full(&mut reader, &mut slice_buf)?;
        if filled == 0 {
            break;
        }

        hash_full_ctx.update(&slice_buf[..filled]);
        if bytes_into_16k < HASH16K_LEN {
            let take = std::cmp::min(HASH16K_LEN - bytes_into_16k, filled);
            hash16k_ctx.update(&slice_buf[..take]);
            bytes_into_16k += take;
        }

        let was_partial = filled < slice_buf.len();
        if was_partial {
            slice_buf[filled..].fill(0);
        }

        let md5: Md5Hash = md5_impl::digest(&slice_buf[..]);
        let crc32 = crc32fast::hash(&slice_buf[..]);
        slice_checksums.push(SliceChecksum { md5, crc32 });

        if let Some(r) = reporter {
            let done = slice_checksums.len() as u64;
            let is_last = was_partial || done == total_slices;
            if is_last || done.is_multiple_of(stride) {
                r.on_event(ProgressEvent::ScanProgress {
                    path,
                    slices_done: done,
                    total_slices,
                });
            }
        }

        if was_partial {
            break;
        }
    }

    let hash_full: Md5Hash = hash_full_ctx.finalize();
    let hash16k: Md5Hash = hash16k_ctx.finalize();
    Ok((slice_checksums, hash_full, hash16k))
}

/// `file_id = MD5( hash16k ‖ length_le8 ‖ name_bytes )`. The name is fed in
/// without any padding and without a NUL terminator — only the bytes that the
/// user typed (or that we resolved from the path).
pub fn compute_file_id(hash16k: &Md5Hash, length: u64, name: &[u8]) -> Md5Hash {
    let mut buf = Vec::with_capacity(16 + 8 + name.len());
    buf.extend_from_slice(hash16k);
    buf.extend_from_slice(&length.to_le_bytes());
    buf.extend_from_slice(name);
    md5_of(&buf)
}

/// Read up to `buf.len()` bytes, returning the number actually filled. Unlike
/// the default `Read::read`, this loops over short reads so the caller can
/// distinguish "end of file" (returns 0) from "kernel handed us a short read".
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_temp(bytes: &[u8]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn rejects_empty_file() {
        let f = write_temp(&[]);
        let err = SourceFile::scan(f.path(), b"x".to_vec(), 4).unwrap_err();
        matches!(err, Par2Error::EmptyFile(_));
    }

    #[test]
    fn rejects_zero_slice_size() {
        let f = write_temp(b"hi");
        let err = SourceFile::scan(f.path(), b"x".to_vec(), 0).unwrap_err();
        matches!(err, Par2Error::InvalidSliceSize(_));
    }

    #[test]
    fn rejects_slice_size_not_multiple_of_4() {
        let f = write_temp(b"hi");
        let err = SourceFile::scan(f.path(), b"x".to_vec(), 5).unwrap_err();
        matches!(err, Par2Error::InvalidSliceSize(_));
    }

    #[test]
    fn rejects_empty_name() {
        let f = write_temp(b"x");
        let err = SourceFile::scan(f.path(), Vec::new(), 4).unwrap_err();
        matches!(err, Par2Error::InvalidFileName(_));
    }

    #[test]
    fn rejects_name_with_nul() {
        let f = write_temp(b"x");
        let err = SourceFile::scan(f.path(), b"bad\0name".to_vec(), 4).unwrap_err();
        matches!(err, Par2Error::InvalidFileName(_));
    }

    #[test]
    fn hashes_short_file_with_zero_padded_last_slice() {
        // 5 bytes, slice size 4 → two slices: "ABCD" then "E" padded with three zeros.
        let f = write_temp(b"ABCDE");
        let sf = SourceFile::scan(f.path(), b"file.bin".to_vec(), 4).unwrap();

        assert_eq!(sf.length, 5);
        assert_eq!(sf.slice_checksums.len(), 2);

        // First slice = MD5("ABCD"), second = MD5("E\0\0\0").
        assert_eq!(sf.slice_checksums[0].md5, md5_of(b"ABCD"));
        assert_eq!(sf.slice_checksums[1].md5, md5_of(b"E\0\0\0"));

        // hash_full covers the real 5 bytes (no padding).
        assert_eq!(sf.hash_full, md5_of(b"ABCDE"));

        // hash16k for a tiny file == MD5 of the whole file.
        assert_eq!(sf.hash16k, md5_of(b"ABCDE"));
    }

    #[test]
    fn hash16k_truncates_at_16384_bytes() {
        let mut data = vec![0u8; HASH16K_LEN + 100];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        let f = write_temp(&data);
        let sf = SourceFile::scan(f.path(), b"big.bin".to_vec(), 4096).unwrap();

        assert_eq!(sf.hash16k, md5_of(&data[..HASH16K_LEN]));
        assert_eq!(sf.hash_full, md5_of(&data));
    }

    #[test]
    fn slice_crc32_matches_independent_implementation() {
        let f = write_temp(b"ABCD");
        let sf = SourceFile::scan(f.path(), b"f".to_vec(), 4).unwrap();
        let expected = crc32fast::hash(b"ABCD");
        assert_eq!(sf.slice_checksums[0].crc32, expected);
    }

    #[test]
    fn file_id_derivation_matches_spec() {
        // Reproduce the derivation by hand and check against compute_file_id.
        let hash16k = [0x11u8; 16];
        let length: u64 = 0x1234_5678;
        let name = b"example.dat";

        let mut buf = Vec::new();
        buf.extend_from_slice(&hash16k);
        buf.extend_from_slice(&length.to_le_bytes());
        buf.extend_from_slice(name);
        let expected = md5_of(&buf);

        assert_eq!(compute_file_id(&hash16k, length, name), expected);
    }

    #[test]
    fn slice_count_matches_ceiling_division() {
        // 17 bytes with slice_size 8 → 3 slices (8, 8, 1+padding).
        let data: Vec<u8> = (0..17u8).collect();
        let f = write_temp(&data);
        let sf = SourceFile::scan(f.path(), b"x".to_vec(), 8).unwrap();
        assert_eq!(sf.slice_checksums.len(), 3);

        // Last slice content = data[16..17] then 7 zero bytes.
        let mut last = vec![16u8];
        last.extend(std::iter::repeat(0).take(7));
        assert_eq!(sf.slice_checksums[2].md5, md5_of(&last));
    }

    #[test]
    fn parallel_scan_matches_serial_for_many_slices() {
        // Regression for the par_chunks fan-out: ensure ordering is preserved
        // even when slice count is large enough to fan across multiple rayon
        // workers. 4096 slices of 16 bytes each, deterministic content.
        let mut data = vec![0u8; 4096 * 16];
        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(0x9E37) >> 5) & 0xff) as u8;
        }
        let f = write_temp(&data);
        let sf = SourceFile::scan(f.path(), b"x".to_vec(), 16).unwrap();
        assert_eq!(sf.slice_checksums.len(), 4096);
        // Spot-check that slice i lines up with bytes [16*i .. 16*i+16].
        for &i in &[0usize, 1, 7, 1023, 4095] {
            let start = i * 16;
            assert_eq!(
                sf.slice_checksums[i].md5,
                md5_of(&data[start..start + 16]),
                "mismatch at slice {i}"
            );
            assert_eq!(
                sf.slice_checksums[i].crc32,
                crc32fast::hash(&data[start..start + 16]),
                "crc mismatch at slice {i}"
            );
        }
    }
}
