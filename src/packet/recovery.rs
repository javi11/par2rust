use crate::format::{build_packet, Md5Hash, TYPE_RECOVERY};

/// Build a `RecoveryBlockPacket`.
///
/// Body layout:
///   - exponent: leu32
///   - data:     `slice_size` bytes (the recovery slice itself)
///
/// `slice_size` must be a multiple of 4 (enforced earlier in the pipeline), so
/// the body length 4 + slice_size is also a multiple of 4 — no padding.
pub fn build_recovery_packet(set_id_hash: &Md5Hash, exponent: u16, slice: &[u8]) -> Vec<u8> {
    debug_assert_eq!(slice.len() % 4, 0, "slice_size must be a multiple of 4");

    let mut body = Vec::with_capacity(4 + slice.len());
    body.extend_from_slice(&(exponent as u32).to_le_bytes());
    body.extend_from_slice(slice);

    build_packet(set_id_hash, &TYPE_RECOVERY, &body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::HEADER_SIZE;

    #[test]
    fn body_layout_for_one_byte_slice_zero_padded_to_4() {
        // PAR2 forbids slice_size < 4 anyway, but verify the simplest non-empty case.
        let slice = vec![0x11u8, 0x22, 0x33, 0x44];
        let pkt = build_recovery_packet(&[0u8; 16], 0x1234, &slice);

        assert_eq!(pkt.len(), HEADER_SIZE + 4 + 4);
        let body = &pkt[HEADER_SIZE..];
        assert_eq!(
            u32::from_le_bytes(body[0..4].try_into().unwrap()),
            0x1234u32,
        );
        assert_eq!(&body[4..8], &slice[..]);
    }

    #[test]
    fn exponent_is_widened_to_u32_in_packet() {
        let pkt = build_recovery_packet(&[0u8; 16], 0xFFFF, &[0u8; 4]);
        let body = &pkt[HEADER_SIZE..];
        assert_eq!(
            u32::from_le_bytes(body[0..4].try_into().unwrap()),
            0x0000_FFFFu32,
        );
    }

    #[test]
    fn longer_slice_payload_is_preserved_verbatim() {
        let slice: Vec<u8> = (0..256u16).map(|x| (x ^ 0xAA) as u8).collect();
        let pkt = build_recovery_packet(&[0xCDu8; 16], 7, &slice);
        let body = &pkt[HEADER_SIZE..];
        assert_eq!(&body[4..], &slice[..]);
        assert_eq!(pkt.len(), HEADER_SIZE + 4 + 256);
    }
}
