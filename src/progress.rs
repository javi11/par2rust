//! Progress reporting for long-running operations.
//!
//! Implement [`ProgressReporter`] and pass it to
//! [`SourceFile::scan_with_progress`](crate::SourceFile::scan_with_progress)
//! or [`run_create_with_progress`](crate::run_create_with_progress) to
//! observe scanning and recovery-block encoding without changing the
//! existing entry points.

use std::path::Path;

/// Receiver for progress notifications emitted by `par2rust`.
///
/// Implementations must be cheap to call: events fire from inside tight
/// loops, although the library rate-limits them so the callback rate
/// stays at roughly ~50 Hz. Implementations are shared by reference and
/// must be `Send + Sync` so they can be reused across the rayon-driven
/// encoding stage.
pub trait ProgressReporter: Send + Sync {
    fn on_event(&self, event: ProgressEvent<'_>);
}

/// One observable progress event.
///
/// Marked `#[non_exhaustive]` so new phases can be added without
/// breaking implementations that exhaustively `match` on the variants.
#[non_exhaustive]
#[derive(Debug)]
pub enum ProgressEvent<'a> {
    /// A source file is about to be scanned.
    ScanStarted { path: &'a Path, total_slices: u64 },
    /// Progress within the scan of a single file.
    ScanProgress {
        path: &'a Path,
        slices_done: u64,
        total_slices: u64,
    },
    /// A source file has been fully scanned.
    ScanCompleted { path: &'a Path },

    /// Encoding of one recovery volume is starting.
    EncodeStarted {
        volume_index: u32,
        total_volumes: u32,
        input_blocks: u64,
        recovery_blocks: u32,
    },
    /// Progress within one recovery volume's encoding.
    EncodeProgress {
        volume_index: u32,
        input_block_done: u64,
        input_blocks: u64,
    },
    /// Encoding of one recovery volume has finished.
    EncodeCompleted { volume_index: u32 },

    /// The index (`.par2`) file has been written to disk.
    IndexWritten { path: &'a Path },
    /// A recovery volume file has been written to disk.
    VolumeWritten { path: &'a Path },
}

/// Convenience: forward `Arc<R>` so callers can share one reporter
/// across threads without re-implementing the trait manually.
impl<R: ProgressReporter + ?Sized> ProgressReporter for std::sync::Arc<R> {
    fn on_event(&self, event: ProgressEvent<'_>) {
        (**self).on_event(event);
    }
}

/// Compute a progress-tick stride that keeps callback rate near ~50 Hz
/// for a loop of `total` iterations.
pub(crate) fn tick_stride(total: u64) -> u64 {
    (total / 100).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Capture {
        log: Mutex<Vec<String>>,
    }

    impl ProgressReporter for Capture {
        fn on_event(&self, event: ProgressEvent<'_>) {
            self.log.lock().unwrap().push(format!("{event:?}"));
        }
    }

    #[test]
    fn arc_forwards_events() {
        let cap = std::sync::Arc::new(Capture {
            log: Mutex::new(Vec::new()),
        });
        let path = Path::new("x");
        ProgressReporter::on_event(
            &cap,
            ProgressEvent::ScanStarted {
                path,
                total_slices: 1,
            },
        );
        assert_eq!(cap.log.lock().unwrap().len(), 1);
    }

    #[test]
    fn tick_stride_keeps_rate_bounded() {
        assert_eq!(tick_stride(0), 1);
        assert_eq!(tick_stride(50), 1);
        assert_eq!(tick_stride(100), 1);
        assert_eq!(tick_stride(1000), 10);
        assert_eq!(tick_stride(10_000), 100);
    }
}
