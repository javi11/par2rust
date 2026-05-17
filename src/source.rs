use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use md5::{Digest, Md5};

use crate::error::{Par2Error, Result};
use crate::format::{md5_of, Md5Hash};
use crate::progress::{tick_stride, ProgressEvent, ProgressReporter};

/// Length of the "first chunk" hashed into `hash16k`. The PAR2 spec hard-codes
/// this to 16 384 bytes — for files smaller than that, only the available bytes
/// are hashed.
pub const HASH16K_LEN: usize = 16 * 1024;

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
        if slice_size == 0 || slice_size % 4 != 0 {
            return Err(Par2Error::InvalidSliceSize(slice_size));
        }

        let length = std::fs::metadata(path)?.len();
        if length == 0 {
            return Err(Par2Error::EmptyFile(path.to_path_buf()));
        }

        let total_slices = length.div_ceil(slice_size);
        if let Some(r) = reporter {
            r.on_event(ProgressEvent::ScanStarted { path, total_slices });
        }
        let stride = tick_stride(total_slices);

        let file = File::open(path)?;
        let mut reader = BufReader::with_capacity(1 << 16, file);

        let mut hash_full_ctx = Md5::new();
        let mut hash16k_ctx = Md5::new();
        let mut bytes_into_16k: usize = 0;

        let slice_size_usize: usize = slice_size
            .try_into()
            .map_err(|_| Par2Error::InvalidSliceSize(slice_size))?;
        let mut slice_buf = vec![0u8; slice_size_usize];
        let mut slice_checksums: Vec<SliceChecksum> = Vec::new();

        loop {
            let filled = read_full(&mut reader, &mut slice_buf)?;
            if filled == 0 {
                break;
            }

            // Feed the partial-or-full slice into the file-wide hashes first.
            hash_full_ctx.update(&slice_buf[..filled]);
            if bytes_into_16k < HASH16K_LEN {
                let take = std::cmp::min(HASH16K_LEN - bytes_into_16k, filled);
                hash16k_ctx.update(&slice_buf[..take]);
                bytes_into_16k += take;
            }

            // For per-slice checksums, the trailing partial slice is zero-padded.
            let was_partial = filled < slice_buf.len();
            if was_partial {
                slice_buf[filled..].fill(0);
            }

            let mut md5_ctx = Md5::new();
            md5_ctx.update(&slice_buf[..]);
            let md5: Md5Hash = md5_ctx.finalize().into();

            let mut crc_ctx = crc32fast::Hasher::new();
            crc_ctx.update(&slice_buf[..]);
            let crc32 = crc_ctx.finalize();

            slice_checksums.push(SliceChecksum { md5, crc32 });

            if let Some(r) = reporter {
                let done = slice_checksums.len() as u64;
                let is_last = was_partial || done == total_slices;
                if is_last || done % stride == 0 {
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

        let hash_full: Md5Hash = hash_full_ctx.finalize().into();
        let hash16k: Md5Hash = hash16k_ctx.finalize().into();
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
}
