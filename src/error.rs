use std::io;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Par2Error {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("invalid slice size {0}: must be > 0 and a multiple of 4")]
    InvalidSliceSize(u64),

    #[error("filename {0:?} is empty or contains a NUL byte; PAR2 does not permit either")]
    InvalidFileName(PathBuf),

    #[error("no input files supplied")]
    NoInputFiles,

    #[error("input file {0:?} is empty; PAR2 requires non-empty files")]
    EmptyFile(PathBuf),

    #[error("too many input files ({0}); PAR2 limit is 32768")]
    TooManyFiles(usize),

    #[error("recovery block count {0} exceeds PAR2 limit of 65535")]
    TooManyRecoveryBlocks(u32),

    #[error(
        "too many input blocks ({count}) at slice_size {slice_size}; \
         PAR2 caps total slices at 32768 — retry with slice_size >= {suggested}"
    )]
    TooManyInputBlocks {
        count: u64,
        slice_size: u64,
        suggested: u64,
    },

    #[error("invalid volume scheme: {0}")]
    InvalidVolumeScheme(String),
}

pub type Result<T> = std::result::Result<T, Par2Error>;
