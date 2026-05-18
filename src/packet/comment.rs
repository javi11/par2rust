use crate::format::{build_packet, round_up_4, Md5Hash, TYPE_COMMENT_ASCII, TYPE_COMMENT_UNICODE};

/// Build the bytes for a sequence of PAR2 comment packets, matching ParPar's
/// behaviour:
///
///   - Every comment produces an ASCII packet (`PAR 2.0\0CommASCI`). Non-ASCII
///     code points are replaced with `?` in this variant.
///   - Comments containing any non-ASCII character additionally produce a
///     Unicode packet (`PAR 2.0\0CommUni\0`). The Unicode body has a 16-byte
///     prefix containing the packet MD5 of the matching ASCII variant, then the
///     comment encoded as UTF-16LE, zero-padded to a multiple of 4 bytes.
///
/// Returns the concatenated packet bytes — ready to splice into a critical
/// packet block.
pub fn build_comment_packets(set_id_hash: &Md5Hash, comments: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    for c in comments {
        out.extend_from_slice(&build_comment_packet_pair(set_id_hash, c));
    }
    out
}

fn build_comment_packet_pair(set_id_hash: &Md5Hash, comment: &str) -> Vec<u8> {
    let has_non_ascii = !comment.is_ascii();
    let ascii_pkt = build_ascii_comment_packet(set_id_hash, comment);

    let mut out = Vec::with_capacity(ascii_pkt.len());
    out.extend_from_slice(&ascii_pkt);

    if has_non_ascii {
        // The Unicode body's 16-byte prefix is the MD5 of the corresponding
        // ASCII variant packet — bytes 16..32 of that packet. We just built it
        // above, so the hash is already available.
        let mut ascii_link = [0u8; 16];
        ascii_link.copy_from_slice(&ascii_pkt[16..32]);
        out.extend_from_slice(&build_unicode_comment_packet(
            set_id_hash,
            comment,
            &ascii_link,
        ));
    }
    out
}

fn build_ascii_comment_packet(set_id_hash: &Md5Hash, comment: &str) -> Vec<u8> {
    // ParPar uses Buffer.from(str, 'ascii'), which drops the high byte of each
    // 16-bit code unit. For BMP characters this collapses to chars().map(c =>
    // (c as u32 & 0xff) as u8). We replace non-ASCII with `?` so the user gets
    // a stable, printable fallback rather than mojibake.
    let mut body_text: Vec<u8> = Vec::with_capacity(comment.len());
    for ch in comment.chars() {
        if (ch as u32) < 0x80 {
            body_text.push(ch as u8);
        } else {
            body_text.push(b'?');
        }
    }
    let padded = round_up_4(body_text.len());
    let mut body = vec![0u8; padded];
    body[..body_text.len()].copy_from_slice(&body_text);
    build_packet(set_id_hash, &TYPE_COMMENT_ASCII, &body)
}

fn build_unicode_comment_packet(
    set_id_hash: &Md5Hash,
    comment: &str,
    ascii_link: &[u8; 16],
) -> Vec<u8> {
    // UTF-16LE body, with a 16-byte prefix linking to the ASCII variant's MD5
    // (or zeros when no ASCII variant exists — currently always present).
    let utf16: Vec<u16> = comment.encode_utf16().collect();
    let mut body = Vec::with_capacity(16 + round_up_4(utf16.len() * 2));
    body.extend_from_slice(ascii_link);
    for unit in utf16 {
        body.extend_from_slice(&unit.to_le_bytes());
    }
    while body.len() % 4 != 0 {
        body.push(0);
    }
    build_packet(set_id_hash, &TYPE_COMMENT_UNICODE, &body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::HEADER_SIZE;

    fn count_packets(bytes: &[u8], type_tag: &[u8; 16]) -> usize {
        let mut count = 0;
        let mut i = 0;
        while i + HEADER_SIZE <= bytes.len() {
            if &bytes[i..i + 8] == b"PAR2\0PKT" {
                let len = u64::from_le_bytes(bytes[i + 8..i + 16].try_into().unwrap()) as usize;
                if len >= HEADER_SIZE && i + len <= bytes.len() {
                    if &bytes[i + 48..i + 64] == type_tag {
                        count += 1;
                    }
                    i += len;
                    continue;
                }
            }
            i += 1;
        }
        count
    }

    #[test]
    fn empty_comments_yields_no_bytes() {
        assert!(build_comment_packets(&[0u8; 16], &[]).is_empty());
    }

    #[test]
    fn ascii_only_comment_emits_single_ascii_packet() {
        let bytes = build_comment_packets(&[0u8; 16], &["hello".to_string()]);
        assert_eq!(count_packets(&bytes, &TYPE_COMMENT_ASCII), 1);
        assert_eq!(count_packets(&bytes, &TYPE_COMMENT_UNICODE), 0);
        // Body padded to mult of 4; "hello" is 5 bytes -> body = 8 bytes.
        assert_eq!(bytes.len(), HEADER_SIZE + 8);
        assert_eq!(&bytes[HEADER_SIZE..HEADER_SIZE + 5], b"hello");
        assert_eq!(&bytes[HEADER_SIZE + 5..HEADER_SIZE + 8], &[0u8; 3]);
    }

    #[test]
    fn non_ascii_comment_emits_both_variants() {
        let bytes = build_comment_packets(&[0u8; 16], &["héllo".to_string()]);
        assert_eq!(count_packets(&bytes, &TYPE_COMMENT_ASCII), 1);
        assert_eq!(count_packets(&bytes, &TYPE_COMMENT_UNICODE), 1);
        // ASCII variant must transliterate non-ASCII to '?'.
        assert_eq!(&bytes[HEADER_SIZE..HEADER_SIZE + 5], b"h?llo");
    }

    #[test]
    fn unicode_packet_body_starts_with_ascii_link() {
        let bytes = build_comment_packets(&[0u8; 16], &["héllo".to_string()]);
        // First packet is ASCII variant.
        let ascii_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let ascii_md5 = &bytes[16..32];
        // Unicode packet starts immediately after.
        let uni_start = ascii_len;
        // bytes 16..32 of any packet are the packet MD5; bytes 64..80 are the
        // body, which for the Unicode variant must be the ASCII MD5 link.
        let link = &bytes[uni_start + 64..uni_start + 80];
        assert_eq!(link, ascii_md5);
    }

    #[test]
    fn multiple_comments_concatenate() {
        let bytes = build_comment_packets(
            &[0u8; 16],
            &["one".to_string(), "two".to_string(), "trés".to_string()],
        );
        assert_eq!(count_packets(&bytes, &TYPE_COMMENT_ASCII), 3);
        assert_eq!(count_packets(&bytes, &TYPE_COMMENT_UNICODE), 1);
    }

    #[test]
    fn all_packets_are_4_byte_aligned() {
        let bytes = build_comment_packets(
            &[0u8; 16],
            &["a".to_string(), "abcdé".to_string(), "x".repeat(7)],
        );
        assert_eq!(bytes.len() % 4, 0);
    }
}
