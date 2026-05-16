//! End-to-end tests that invoke the `par2rust` CLI binary as a subprocess and
//! then have upstream `par2 r` operate on the produced recovery set. Proves
//! the CLI wiring works on a real path.

mod common;

use std::path::PathBuf;
use std::process::Command;

use tempfile::tempdir;

use common::{deterministic_bytes, ensure_par2_available_or_skip, run_par2_repair, write_fixture};

fn par2rust_bin() -> PathBuf {
    std::env::var_os("CARGO_BIN_EXE_par2rust")
        .map(PathBuf::from)
        .expect("cargo did not export CARGO_BIN_EXE_par2rust — are tests run via cargo?")
}

#[test]
fn create_then_upstream_repair() {
    if !ensure_par2_available_or_skip("create_then_upstream_repair") {
        return;
    }
    let dir = tempdir().unwrap();
    let original = deterministic_bytes(0xBA5EBA11, 32 * 1024);
    write_fixture(dir.path(), "doc.bin", &original);

    let archive = dir.path().join("backup.par2");
    let status = Command::new(par2rust_bin())
        .args(["create", "-s", "4096", "-c", "3"])
        .arg(&archive)
        .arg("doc.bin")
        .current_dir(dir.path())
        .status()
        .expect("failed to spawn par2rust binary");
    assert!(status.success(), "par2rust create exited non-zero");

    // Corrupt one slice and let upstream repair.
    let payload_path = dir.path().join("doc.bin");
    let mut bytes = std::fs::read(&payload_path).unwrap();
    for b in &mut bytes[8192..8192 + 300] {
        *b = b.wrapping_add(13);
    }
    std::fs::write(&payload_path, &bytes).unwrap();

    let (status, stdout, stderr) = run_par2_repair(dir.path(), &archive);
    assert!(
        status.success(),
        "par2 r failed:\nstdout:\n{stdout}\nstderr:\n{stderr}",
    );
    assert_eq!(std::fs::read(&payload_path).unwrap(), original);
}

/// With `--single-volume`, par2rust must emit exactly one `vol0+N.par2`
/// regardless of the (otherwise exponential-by-default) CLI behaviour. Repair
/// must still succeed.
#[test]
fn single_volume_flag_emits_one_volume_file() {
    if !ensure_par2_available_or_skip("single_volume_flag_emits_one_volume_file") {
        return;
    }
    let dir = tempdir().unwrap();
    let original = deterministic_bytes(0xABCD_1234, 32 * 1024);
    write_fixture(dir.path(), "doc.bin", &original);

    let archive = dir.path().join("backup.par2");
    let status = Command::new(par2rust_bin())
        .args(["create", "--single-volume", "-s", "4096", "-c", "5"])
        .arg(&archive)
        .arg("doc.bin")
        .current_dir(dir.path())
        .status()
        .expect("failed to spawn par2rust binary");
    assert!(
        status.success(),
        "par2rust create --single-volume exited non-zero"
    );

    // Exactly one volume file, named vol0+5.par2.
    let vol_files: Vec<PathBuf> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.contains(".vol"))
        })
        .collect();
    assert_eq!(
        vol_files.len(),
        1,
        "--single-volume should produce one volume file, got {vol_files:?}"
    );
    assert_eq!(
        vol_files[0].file_name().unwrap().to_string_lossy(),
        "backup.vol0+5.par2"
    );

    // Sanity: corrupt and repair must still succeed end-to-end.
    let payload_path = dir.path().join("doc.bin");
    let mut bytes = std::fs::read(&payload_path).unwrap();
    for b in &mut bytes[10_000..10_000 + 200] {
        *b ^= 0xFF;
    }
    std::fs::write(&payload_path, &bytes).unwrap();

    let (status, stdout, stderr) = run_par2_repair(dir.path(), &archive);
    assert!(
        status.success(),
        "single-volume repair failed:\nstdout:\n{stdout}\nstderr:\n{stderr}",
    );
    assert_eq!(std::fs::read(&payload_path).unwrap(), original);
}
