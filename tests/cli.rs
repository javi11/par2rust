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

/// `--out` is a ParPar-style alias for the positional `<ARCHIVE>` and must
/// produce the same recovery set.
#[test]
fn out_flag_replaces_positional_archive() {
    let dir = tempdir().unwrap();
    write_fixture(
        dir.path(),
        "doc.bin",
        &deterministic_bytes(0x1234_5678, 4096),
    );

    let archive = dir.path().join("via-out.par2");
    let status = Command::new(par2rust_bin())
        .args(["create", "--out"])
        .arg(&archive)
        .arg("--quiet")
        .arg("doc.bin")
        .current_dir(dir.path())
        .status()
        .expect("failed to spawn par2rust binary");
    assert!(status.success(), "par2rust create --out exited non-zero");
    assert!(archive.exists(), "--out target was not written");
}

/// `--comment` should add ASCII comment packets that par2cmdline accepts.
#[test]
fn comment_flag_embeds_comment_packets() {
    let dir = tempdir().unwrap();
    write_fixture(dir.path(), "doc.bin", &deterministic_bytes(7, 8192));
    let archive = dir.path().join("with-comments.par2");
    let status = Command::new(par2rust_bin())
        .args([
            "create",
            "--comment",
            "hello from par2rust",
            "--comment",
            "second note",
            "--quiet",
        ])
        .arg(&archive)
        .arg("doc.bin")
        .current_dir(dir.path())
        .status()
        .expect("spawn failed");
    assert!(status.success(), "create with --comment failed");

    // Two ASCII comment packets must be present in the index file. We scan
    // for the literal type tag at the expected packet header offset 48.
    let bytes = std::fs::read(&archive).unwrap();
    let needle = b"PAR 2.0\0CommASCI";
    let count = bytes.windows(needle.len()).filter(|w| *w == needle).count();
    assert_eq!(count, 2, "expected 2 ASCII comment packets, found {count}");
}

/// `--recurse` walks directories; without it, a directory input must error.
#[test]
fn recurse_flag_walks_directories() {
    let dir = tempdir().unwrap();
    let sub = dir.path().join("subdir");
    std::fs::create_dir(&sub).unwrap();
    write_fixture(&sub, "a.bin", &deterministic_bytes(1, 512));
    write_fixture(&sub, "b.bin", &deterministic_bytes(2, 512));

    let archive = dir.path().join("recurse.par2");
    // Without --recurse, directory input should fail.
    let out = Command::new(par2rust_bin())
        .args(["create", "--quiet"])
        .arg(&archive)
        .arg(&sub)
        .current_dir(dir.path())
        .output()
        .expect("spawn failed");
    assert!(
        !out.status.success(),
        "directory without --recurse must error"
    );

    // With --recurse, success and both files protected.
    let status = Command::new(par2rust_bin())
        .args(["create", "--recurse", "--quiet"])
        .arg(&archive)
        .arg(&sub)
        .current_dir(dir.path())
        .status()
        .expect("spawn failed");
    assert!(status.success(), "--recurse create failed");
    assert!(archive.exists());
}

/// `--input-file` reads additional input paths from a list file.
#[test]
fn input_file_flag_reads_paths_from_list() {
    let dir = tempdir().unwrap();
    write_fixture(dir.path(), "a.bin", &deterministic_bytes(1, 1024));
    write_fixture(dir.path(), "b.bin", &deterministic_bytes(2, 1024));
    let list_path = dir.path().join("inputs.txt");
    std::fs::write(&list_path, "a.bin\nb.bin\n").unwrap();

    let archive = dir.path().join("listed.par2");
    let status = Command::new(par2rust_bin())
        .args(["create", "--input-file"])
        .arg(&list_path)
        .arg("--quiet")
        .arg(&archive)
        .current_dir(dir.path())
        .status()
        .expect("spawn failed");
    assert!(status.success(), "create with --input-file failed");
    assert!(archive.exists());
}

/// `--quiet` suppresses stdout output.
#[test]
fn quiet_flag_suppresses_stdout() {
    let dir = tempdir().unwrap();
    write_fixture(dir.path(), "doc.bin", &deterministic_bytes(3, 4096));
    let archive = dir.path().join("silent.par2");
    let out = Command::new(par2rust_bin())
        .args(["create", "--quiet"])
        .arg(&archive)
        .arg("doc.bin")
        .current_dir(dir.path())
        .output()
        .expect("spawn failed");
    assert!(out.status.success());
    assert!(
        out.stdout.is_empty(),
        "--quiet should emit no stdout, got {} bytes",
        out.stdout.len()
    );
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
