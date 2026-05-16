//! Shared helpers for integration tests that exercise par2rust against the
//! upstream `par2cmdline` binary.
//!
//! Set `PAR2_BIN` to override the path to the upstream binary.

// Each `tests/*.rs` integration test compiles `common/mod.rs` as part of its
// own binary, and reports any helper it does not call as dead code. The
// helpers here are intentionally a shared toolbox, so silence that lint.
#![allow(dead_code)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn par2_bin() -> PathBuf {
    std::env::var_os("PAR2_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("par2"))
}

/// Probe the configured `par2` binary by running `par2 -V`. Returns `true` if
/// the binary exists and responds; otherwise the test should treat this as an
/// environment failure (not a code failure) and skip itself.
pub fn ensure_par2_available_or_skip(test_name: &str) -> bool {
    match Command::new(par2_bin()).arg("-V").output() {
        Ok(out) if out.status.success() => true,
        Ok(_) | Err(_) => {
            eprintln!(
                "[{test_name}] skipping: upstream par2 binary not found \
                 (set PAR2_BIN or install par2cmdline). On Linux: `apt install par2`, \
                 macOS: `brew install par2`, Windows: `choco install par2cmdline`.",
            );
            false
        }
    }
}

pub fn write_fixture(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
    let p = dir.join(name);
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(content).unwrap();
    f.flush().unwrap();
    p
}

pub fn deterministic_bytes(seed: u64, len: usize) -> Vec<u8> {
    // Simple LCG so fixtures are reproducible without pulling in `rand`.
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push((state >> 33) as u8);
    }
    out
}

/// Run `par2 v <archive>` from inside `dir` and return its stdout/exit.
pub fn run_par2_verify(dir: &Path, archive: &Path) -> (std::process::ExitStatus, String, String) {
    let output = Command::new(par2_bin())
        .arg("v")
        .arg(archive)
        .current_dir(dir)
        .output()
        .expect("failed to invoke par2 — set PAR2_BIN if it is not on PATH");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    (output.status, stdout, stderr)
}

/// Run `par2 r <archive>` from inside `dir`.
pub fn run_par2_repair(dir: &Path, archive: &Path) -> (std::process::ExitStatus, String, String) {
    let output = Command::new(par2_bin())
        .arg("r")
        .arg(archive)
        .current_dir(dir)
        .output()
        .expect("failed to invoke par2");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    (output.status, stdout, stderr)
}
