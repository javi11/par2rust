use crate::format::{build_packet, md5_of, Md5Hash, TYPE_MAIN};

/// Result of building a `MainPacket`: the serialised packet bytes, plus the
/// `set_id_hash` derived from the body (the hash *every* other packet in the
/// recovery set carries in its header).
pub struct MainPacket {
    pub bytes: Vec<u8>,
    pub set_id_hash: Md5Hash,
}

/// Build a `MainPacket` from the file IDs of every input file.
///
/// Body layout (all little-endian):
///   - slice_size:                leu64
///   - recoverable_file_count:    leu32
///   - file_ids:                  N · 16 bytes, sorted ascending
///
/// The set_id_hash is `MD5(body)`. Per the PAR2 spec, file IDs MUST be sorted
/// in ascending order so that any creator producing the same logical recovery
/// set arrives at the same set_id_hash.
///
/// For now we treat every input file as "recoverable" (par2rust does not
/// distinguish recoverable from non-recoverable files). The body length is
/// 8 + 4 + 16·N which is always a multiple of 4.
pub fn build_main_packet(slice_size: u64, file_ids: &[Md5Hash]) -> MainPacket {
    let mut sorted: Vec<Md5Hash> = file_ids.to_vec();
    sorted.sort_unstable();

    let recoverable_count: u32 = sorted.len() as u32;

    let mut body = Vec::with_capacity(8 + 4 + sorted.len() * 16);
    body.extend_from_slice(&slice_size.to_le_bytes());
    body.extend_from_slice(&recoverable_count.to_le_bytes());
    for id in &sorted {
        body.extend_from_slice(id);
    }
    debug_assert_eq!(body.len() % 4, 0);

    let set_id_hash = md5_of(&body);
    let bytes = build_packet(&set_id_hash, &TYPE_MAIN, &body);
    MainPacket { bytes, set_id_hash }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::HEADER_SIZE;

    #[test]
    fn file_ids_are_sorted_ascending_in_packet() {
        let ids = vec![[0x20u8; 16], [0x10u8; 16], [0x30u8; 16]];
        let main = build_main_packet(4096, &ids);

        let body = &main.bytes[HEADER_SIZE..];
        // skip slice_size(8) + count(4) = 12
        assert_eq!(&body[12..28], &[0x10u8; 16]);
        assert_eq!(&body[28..44], &[0x20u8; 16]);
        assert_eq!(&body[44..60], &[0x30u8; 16]);
    }

    #[test]
    fn set_id_hash_is_md5_of_body() {
        let ids = vec![[0xAAu8; 16]];
        let main = build_main_packet(4096, &ids);

        let mut expected_body = Vec::new();
        expected_body.extend_from_slice(&4096u64.to_le_bytes());
        expected_body.extend_from_slice(&1u32.to_le_bytes());
        expected_body.extend_from_slice(&[0xAA; 16]);
        let expected = md5_of(&expected_body);

        assert_eq!(main.set_id_hash, expected);
        // And the header carries that same set_id_hash.
        assert_eq!(&main.bytes[32..48], &main.set_id_hash);
    }

    #[test]
    fn slice_size_round_trips_via_le_encoding() {
        let main = build_main_packet(0x1234_5678_9ABC_DEF0, &[]);
        let body = &main.bytes[HEADER_SIZE..];
        assert_eq!(
            u64::from_le_bytes(body[0..8].try_into().unwrap()),
            0x1234_5678_9ABC_DEF0
        );
    }

    #[test]
    fn recoverable_count_matches_file_id_count() {
        let ids = vec![[1u8; 16], [2u8; 16], [3u8; 16], [4u8; 16]];
        let main = build_main_packet(4096, &ids);
        let body = &main.bytes[HEADER_SIZE..];
        assert_eq!(u32::from_le_bytes(body[8..12].try_into().unwrap()), 4);
    }

    #[test]
    fn body_length_is_4_byte_aligned() {
        for n in 0..8 {
            let ids: Vec<Md5Hash> = (0..n).map(|i| [i as u8; 16]).collect();
            let main = build_main_packet(4096, &ids);
            assert_eq!(main.bytes.len() % 4, 0, "n={}", n);
        }
    }
}
