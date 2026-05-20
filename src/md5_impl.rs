//! MD5 backend shim with two interchangeable implementations.
//!
//! - **Default**: the pure-Rust [`md-5`](https://crates.io/crates/md-5)
//!   crate from RustCrypto. Portable, no native deps, ~500 MB/s on
//!   Apple Silicon.
//! - **`fast-md5` feature**: OpenSSL's MD5 (`openssl::hash::Hasher`
//!   with `MessageDigest::md5()`), with hand-tuned ARM assembly on
//!   aarch64 and AVX/AVX-512 paths on x86. Typically ~1 GB/s on
//!   Apple Silicon. Vendored OpenSSL is built from source the first
//!   time, so no system OpenSSL is required at the consumer's machine.
//!
//! MD5 output is spec-defined and byte-identical between the two
//! backends — `tests/interop_*` is the authoritative cross-check.
//!
//! The streaming `Md5Ctx` mirrors the subset of `md-5`'s API par2rust
//! actually uses (`new` / `update` / `finalize`). Each `Md5Ctx` owns
//! exclusive state, so it is `Send` but the type is intentionally not
//! `Sync` — rayon workers construct their own contexts.

/// 16-byte MD5 digest. Re-exported from [`crate::format`] for backward
/// compatibility; new code can use this alias directly.
pub type Md5Digest = [u8; 16];

#[cfg(not(feature = "fast-md5"))]
mod backend {
    use md5::{Digest, Md5};

    /// Streaming MD5 context backed by the pure-Rust `md-5` crate.
    pub struct Md5Ctx(Md5);

    impl Md5Ctx {
        pub fn new() -> Self {
            Self(Md5::new())
        }
        pub fn update(&mut self, data: &[u8]) {
            self.0.update(data);
        }
        pub fn finalize(self) -> super::Md5Digest {
            self.0.finalize().into()
        }
    }

    /// One-shot MD5 over `data`. Equivalent to constructing an
    /// `Md5Ctx`, calling `update`, and `finalize`-ing.
    pub fn digest(data: &[u8]) -> super::Md5Digest {
        Md5::digest(data).into()
    }
}

#[cfg(feature = "fast-md5")]
mod backend {
    use openssl::hash::{Hasher, MessageDigest};

    /// Streaming MD5 context backed by OpenSSL.
    ///
    /// `Hasher` allocates a small EVP context on the heap; per-slice
    /// construction is cheap relative to even a single MB of hashing.
    pub struct Md5Ctx(Hasher);

    impl Md5Ctx {
        pub fn new() -> Self {
            // `MessageDigest::md5()` only fails if OpenSSL is built
            // without MD5 support, which the vendored build never is.
            Self(Hasher::new(MessageDigest::md5()).expect("openssl md5 init"))
        }
        pub fn update(&mut self, data: &[u8]) {
            // `update` only fails on I/O errors against an external
            // sink; here the sink is OpenSSL's internal state.
            self.0.update(data).expect("openssl md5 update");
        }
        pub fn finalize(mut self) -> super::Md5Digest {
            let bytes = self.0.finish().expect("openssl md5 finalize");
            let mut out = [0u8; 16];
            // OpenSSL returns the digest in an OpenSSL-owned slice;
            // copy into the spec-shaped array.
            out.copy_from_slice(&bytes);
            out
        }
    }

    pub fn digest(data: &[u8]) -> super::Md5Digest {
        let mut ctx = Md5Ctx::new();
        ctx.update(data);
        ctx.finalize()
    }
}

pub use backend::{digest, Md5Ctx};

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: both backends must agree with the RFC 1321 test vectors.
    /// We hard-code the expected digests rather than cross-comparing
    /// because in any single build only one backend is compiled in.
    #[test]
    fn known_md5_vectors() {
        // MD5("") = d41d8cd98f00b204e9800998ecf8427e
        assert_eq!(
            digest(b""),
            [
                0xd4, 0x1d, 0x8c, 0xd9, 0x8f, 0x00, 0xb2, 0x04, 0xe9, 0x80, 0x09, 0x98, 0xec, 0xf8,
                0x42, 0x7e,
            ]
        );
        // MD5("abc") = 900150983cd24fb0d6963f7d28e17f72
        assert_eq!(
            digest(b"abc"),
            [
                0x90, 0x01, 0x50, 0x98, 0x3c, 0xd2, 0x4f, 0xb0, 0xd6, 0x96, 0x3f, 0x7d, 0x28, 0xe1,
                0x7f, 0x72,
            ]
        );
    }

    #[test]
    fn streaming_matches_one_shot() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let one_shot = digest(data);
        let streamed = {
            let mut ctx = Md5Ctx::new();
            // Update in two chunks to exercise the streaming path.
            ctx.update(&data[..15]);
            ctx.update(&data[15..]);
            ctx.finalize()
        };
        assert_eq!(one_shot, streamed);
    }
}
