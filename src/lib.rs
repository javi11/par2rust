//! Create-only Rust port of [par2cmdline](https://github.com/Parchive/par2cmdline).
//!
//! Use [`run_create`] to produce a PAR2 recovery set programmatically. Output
//! is byte-compatible with PAR 2.0 — `par2 v` / `par2 r` from upstream will
//! accept it.
//!
//! ```no_run
//! use std::path::PathBuf;
//! use par2rust::{run_create, CreateOptions, SourceFile};
//!
//! # fn main() -> par2rust::Result<()> {
//! let path = PathBuf::from("data.bin");
//! let name = path.file_name().unwrap().as_encoded_bytes().to_vec();
//! let source = SourceFile::scan(&path, name, 4096)?;
//!
//! let written = run_create(
//!     &CreateOptions {
//!         output: PathBuf::from("backup.par2"),
//!         slice_size: 4096,
//!         recovery_block_count: 10,
//!         ..Default::default()
//!     },
//!     &[source],
//! )?;
//! for p in &written {
//!     println!("wrote {}", p.display());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! For a fully runnable version see [`examples/create_from_lib.rs`](https://github.com/javi11/par2rust/blob/main/examples/create_from_lib.rs).
//!
//! Library consumers should disable the default `cli` feature to avoid
//! pulling in `clap`:
//!
//! ```toml
//! [dependencies]
//! par2rust = { version = "0.1", default-features = false }
//! ```

pub mod creator;
pub mod error;
pub mod format;
pub mod galois;
pub mod galois_simd;
pub(crate) mod md5_impl;
pub mod packet;
pub mod progress;
pub mod reedsolomon;
pub mod source;

pub use creator::{
    run_create, run_create_fused, run_create_with_progress, write_index_file, CreateOptions,
    VolumeScheme, MAX_FILES, MAX_INPUT_BLOCKS, MAX_RECOVERY_BLOCKS,
};
pub use error::{Par2Error, Result};
pub use progress::{ProgressEvent, ProgressReporter};
pub use source::SourceFile;
