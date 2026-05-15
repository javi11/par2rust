use md5::{Digest, Md5};

pub const PACKET_MAGIC: [u8; 8] = *b"PAR2\0PKT";
pub const HEADER_SIZE: usize = 64;
pub const MD5_SIZE: usize = 16;

pub const TYPE_FILE_DESC: [u8; 16] = *b"PAR 2.0\0FileDesc";
pub const TYPE_FILE_VERIFY: [u8; 16] = *b"PAR 2.0\0IFSC\0\0\0\0";
pub const TYPE_MAIN: [u8; 16] = *b"PAR 2.0\0Main\0\0\0\0";
pub const TYPE_RECOVERY: [u8; 16] = *b"PAR 2.0\0RecvSlic";
pub const TYPE_CREATOR: [u8; 16] = *b"PAR 2.0\0Creator\0";

pub type Md5Hash = [u8; 16];

pub fn md5_of(bytes: &[u8]) -> Md5Hash {
    let mut h = Md5::new();
    h.update(bytes);
    h.finalize().into()
}

/// Round `n` up to the next multiple of 4. PAR2 requires every packet body length
/// (and therefore the trailing filename padding) to be 4-byte aligned.
pub fn round_up_4(n: usize) -> usize {
    (n + 3) & !3
}

/// Serialise a complete PAR2 packet given its `set_id_hash`, type tag, and body
/// bytes (which must already be 4-byte aligned). Returns header + body, with the
/// per-packet MD5 hash field correctly set to `MD5(set_id ‖ type ‖ body)`.
///
/// This is the single place packets are framed; every packet writer composes its
/// body first, then hands the bytes here.
pub fn build_packet(set_id_hash: &Md5Hash, type_tag: &[u8; 16], body: &[u8]) -> Vec<u8> {
    debug_assert!(body.len() % 4 == 0, "packet body must be 4-byte aligned");
    let packet_len = HEADER_SIZE + body.len();

    let mut hash_input = Vec::with_capacity(32 + body.len());
    hash_input.extend_from_slice(set_id_hash);
    hash_input.extend_from_slice(type_tag);
    hash_input.extend_from_slice(body);
    let packet_hash = md5_of(&hash_input);

    let mut out = Vec::with_capacity(packet_len);
    out.extend_from_slice(&PACKET_MAGIC);
    out.extend_from_slice(&(packet_len as u64).to_le_bytes());
    out.extend_from_slice(&packet_hash);
    out.extend_from_slice(set_id_hash);
    out.extend_from_slice(type_tag);
    out.extend_from_slice(body);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_up_4_handles_all_residues() {
        assert_eq!(round_up_4(0), 0);
        assert_eq!(round_up_4(1), 4);
        assert_eq!(round_up_4(2), 4);
        assert_eq!(round_up_4(3), 4);
        assert_eq!(round_up_4(4), 4);
        assert_eq!(round_up_4(5), 8);
        assert_eq!(round_up_4(17), 20);
    }

    #[test]
    fn type_tags_match_upstream_par2cmdline() {
        // Spot-check the literal bytes against par2fileformat.cpp definitions.
        assert_eq!(&TYPE_MAIN, b"PAR 2.0\0Main\0\0\0\0");
        assert_eq!(&TYPE_FILE_DESC, b"PAR 2.0\0FileDesc");
        assert_eq!(&TYPE_FILE_VERIFY, b"PAR 2.0\0IFSC\0\0\0\0");
        assert_eq!(&TYPE_RECOVERY, b"PAR 2.0\0RecvSlic");
        assert_eq!(&TYPE_CREATOR, b"PAR 2.0\0Creator\0");
    }

    #[test]
    fn packet_magic_is_par2_pkt() {
        assert_eq!(&PACKET_MAGIC, b"PAR2\0PKT");
    }

    #[test]
    fn build_packet_has_correct_header_layout() {
        let set_id = [0xAAu8; 16];
        let body = vec![0u8; 8];
        let pkt = build_packet(&set_id, &TYPE_CREATOR, &body);

        assert_eq!(pkt.len(), HEADER_SIZE + body.len());
        assert_eq!(&pkt[0..8], &PACKET_MAGIC);
        assert_eq!(
            u64::from_le_bytes(pkt[8..16].try_into().unwrap()),
            (HEADER_SIZE + body.len()) as u64,
        );
        // bytes 16..32 are the packet MD5 (checked separately)
        assert_eq!(&pkt[32..48], &set_id);
        assert_eq!(&pkt[48..64], &TYPE_CREATOR);
        assert_eq!(&pkt[64..], &body[..]);
    }

    #[test]
    fn build_packet_md5_covers_setid_through_body() {
        // The PAR2 spec defines packet_hash = MD5( setid || type || body ).
        // We verify by recomputing independently here.
        let set_id = [0x11u8; 16];
        let body = b"hello, par2!\0\0\0\0".to_vec(); // 16 bytes, 4-aligned
        let pkt = build_packet(&set_id, &TYPE_CREATOR, &body);

        let mut expected = Vec::new();
        expected.extend_from_slice(&set_id);
        expected.extend_from_slice(&TYPE_CREATOR);
        expected.extend_from_slice(&body);
        let expected_hash = md5_of(&expected);

        assert_eq!(&pkt[16..32], &expected_hash);
    }

    #[test]
    fn md5_of_empty_matches_known_value() {
        // d41d8cd98f00b204e9800998ecf8427e — MD5("")
        let h = md5_of(&[]);
        let expected = [
            0xd4, 0x1d, 0x8c, 0xd9, 0x8f, 0x00, 0xb2, 0x04, 0xe9, 0x80, 0x09, 0x98, 0xec, 0xf8,
            0x42, 0x7e,
        ];
        assert_eq!(h, expected);
    }

    #[test]
    #[should_panic(expected = "4-byte aligned")]
    fn build_packet_rejects_unaligned_body_in_debug() {
        let set_id = [0u8; 16];
        let _ = build_packet(&set_id, &TYPE_CREATOR, &[0u8; 7]);
    }
}
