use crate::format::{build_packet, round_up_4, Md5Hash, TYPE_CREATOR};

pub const CREATOR_STRING: &str = "Created by par2rust v0.1.0";

/// Build the `CreatorPacket`. The body is the literal creator string padded with
/// zeros to a multiple of 4 bytes. PAR2 places no constraint on the content
/// beyond that.
pub fn build_creator_packet(set_id_hash: &Md5Hash) -> Vec<u8> {
    let bytes = CREATOR_STRING.as_bytes();
    let padded = round_up_4(bytes.len());
    let mut body = vec![0u8; padded];
    body[..bytes.len()].copy_from_slice(bytes);
    build_packet(set_id_hash, &TYPE_CREATOR, &body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::HEADER_SIZE;

    #[test]
    fn creator_body_contains_signature_string() {
        let pkt = build_creator_packet(&[0u8; 16]);
        let body = &pkt[HEADER_SIZE..];
        assert!(body.starts_with(CREATOR_STRING.as_bytes()));
        assert_eq!(pkt.len() % 4, 0);
    }
}
