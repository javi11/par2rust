use crate::format::{build_packet, round_up_4, Md5Hash, TYPE_FILE_DESC};
use crate::source::SourceFile;

/// Build a `FileDescriptionPacket` for one source file.
///
/// Body layout (all little-endian):
///   - fileid:    16 bytes
///   - hashfull:  16 bytes
///   - hash16k:   16 bytes
///   - length:     8 bytes
///   - name:       padded with 1..=3 zero bytes so total body length is a multiple of 4
///
/// PAR2 says: "If the name of the file is an exact multiple of 4 characters in
/// length then it may not have a NULL termination." We follow upstream's
/// convention of always padding to a multiple of 4 (so a name length divisible
/// by 4 receives zero padding bytes).
pub fn build_file_desc_packet(set_id_hash: &Md5Hash, src: &SourceFile) -> Vec<u8> {
    let name_padded = round_up_4(src.name.len());
    let mut body = Vec::with_capacity(56 + name_padded);
    body.extend_from_slice(&src.file_id);
    body.extend_from_slice(&src.hash_full);
    body.extend_from_slice(&src.hash16k);
    body.extend_from_slice(&src.length.to_le_bytes());
    body.extend_from_slice(&src.name);
    body.resize(56 + name_padded, 0);

    build_packet(set_id_hash, &TYPE_FILE_DESC, &body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{HEADER_SIZE, PACKET_MAGIC};
    use crate::source::SourceFile;

    fn fake_source(name: &[u8]) -> SourceFile {
        SourceFile {
            name: name.to_vec(),
            path: std::path::PathBuf::from("/tmp/fake"),
            length: 0x10,
            hash_full: [0xAA; 16],
            hash16k: [0xBB; 16],
            file_id: [0xCC; 16],
            slice_checksums: vec![],
        }
    }

    #[test]
    fn body_layout_matches_spec_for_aligned_name() {
        // 4-char name → no padding bytes needed.
        let src = fake_source(b"abcd");
        let set_id = [0u8; 16];
        let pkt = build_file_desc_packet(&set_id, &src);

        // header + 16 + 16 + 16 + 8 + 4 = 64 + 60 = 124
        assert_eq!(pkt.len(), HEADER_SIZE + 60);
        assert_eq!(&pkt[0..8], &PACKET_MAGIC);

        let body = &pkt[HEADER_SIZE..];
        assert_eq!(&body[0..16], &src.file_id);
        assert_eq!(&body[16..32], &src.hash_full);
        assert_eq!(&body[32..48], &src.hash16k);
        assert_eq!(
            u64::from_le_bytes(body[48..56].try_into().unwrap()),
            src.length
        );
        assert_eq!(&body[56..60], b"abcd");
    }

    #[test]
    fn body_pads_unaligned_name_with_zeros() {
        // 5-char name → 3 zero bytes of padding for an 8-byte tail.
        let src = fake_source(b"hello");
        let pkt = build_file_desc_packet(&[0u8; 16], &src);

        // 56 fixed + round_up_4(5) = 56 + 8 = 64
        assert_eq!(pkt.len(), HEADER_SIZE + 64);
        let body = &pkt[HEADER_SIZE..];
        assert_eq!(&body[56..61], b"hello");
        assert_eq!(&body[61..64], &[0u8, 0, 0]);
    }

    #[test]
    fn packet_length_field_matches_total_size() {
        let src = fake_source(b"file.bin");
        let pkt = build_file_desc_packet(&[0u8; 16], &src);
        let declared_len = u64::from_le_bytes(pkt[8..16].try_into().unwrap()) as usize;
        assert_eq!(declared_len, pkt.len());
    }
}
