use crate::format::{build_packet, Md5Hash, TYPE_FILE_VERIFY};
use crate::source::SourceFile;

/// Build a `FileVerificationPacket` (a.k.a. "Input File Slice Checksum" / IFSC).
///
/// Body layout:
///   - fileid:   16 bytes
///   - entries:  N × (md5: 16, crc32: leu32) = 20 bytes each
///
/// The PAR2 spec requires the body length to be a multiple of 4 — 16 + 20·N is
/// always a multiple of 4 so no padding is needed.
pub fn build_file_verify_packet(set_id_hash: &Md5Hash, src: &SourceFile) -> Vec<u8> {
    let entries = src.slice_checksums.len();
    let mut body = Vec::with_capacity(16 + entries * 20);
    body.extend_from_slice(&src.file_id);
    for sc in &src.slice_checksums {
        body.extend_from_slice(&sc.md5);
        body.extend_from_slice(&sc.crc32.to_le_bytes());
    }
    debug_assert_eq!(body.len() % 4, 0);

    build_packet(set_id_hash, &TYPE_FILE_VERIFY, &body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::HEADER_SIZE;
    use crate::source::{SliceChecksum, SourceFile};

    fn fake_with_slices(n: usize) -> SourceFile {
        let mut slices = Vec::new();
        for i in 0..n {
            slices.push(SliceChecksum {
                md5: [i as u8; 16],
                crc32: 0xDEADBEEF + i as u32,
            });
        }
        SourceFile {
            name: b"f".to_vec(),
            path: std::path::PathBuf::from("/tmp/fake"),
            length: 4,
            hash_full: [0; 16],
            hash16k: [0; 16],
            file_id: [0x42; 16],
            slice_checksums: slices,
        }
    }

    #[test]
    fn body_layout_with_two_slices() {
        let src = fake_with_slices(2);
        let pkt = build_file_verify_packet(&[0u8; 16], &src);

        // 64 header + 16 fileid + 2·20 entries = 64 + 16 + 40 = 120
        assert_eq!(pkt.len(), HEADER_SIZE + 16 + 40);
        let body = &pkt[HEADER_SIZE..];
        assert_eq!(&body[0..16], &src.file_id);

        for (i, sc) in src.slice_checksums.iter().enumerate() {
            let off = 16 + i * 20;
            assert_eq!(&body[off..off + 16], &sc.md5);
            assert_eq!(
                u32::from_le_bytes(body[off + 16..off + 20].try_into().unwrap()),
                sc.crc32
            );
        }
    }

    #[test]
    fn empty_slice_list_still_produces_well_formed_packet() {
        let src = fake_with_slices(0);
        let pkt = build_file_verify_packet(&[0u8; 16], &src);
        assert_eq!(pkt.len(), HEADER_SIZE + 16);
        assert!(pkt.len().is_multiple_of(4));
    }
}
