//! Golden tests that exercise par2rust's output against the upstream `par2`
//! binary installed on the system. Each test:
//!   1. Writes one or more fixture files into a tempdir.
//!   2. Runs par2rust to produce an index `.par2`.
//!   3. Invokes upstream `par2 v` and asserts it accepts the file.
//!
//! Set `PAR2_BIN` to override the path to the upstream binary.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use par2rust::{run_create, write_index_file, CreateOptions, SourceFile, VolumeScheme};
use tempfile::tempdir;

fn par2_bin() -> PathBuf {
    std::env::var_os("PAR2_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("par2"))
}

/// Probe the configured `par2` binary by running `par2 -V`. Returns `Ok(())`
/// if the binary exists and responds; otherwise the test should treat this as
/// an environment failure (not a code failure) and skip itself.
fn ensure_par2_available_or_skip(test_name: &str) -> bool {
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

fn write_fixture(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
    let p = dir.join(name);
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(content).unwrap();
    f.flush().unwrap();
    p
}

fn deterministic_bytes(seed: u64, len: usize) -> Vec<u8> {
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
fn run_par2_verify(dir: &Path, archive: &Path) -> (std::process::ExitStatus, String, String) {
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
fn run_par2_repair(dir: &Path, archive: &Path) -> (std::process::ExitStatus, String, String) {
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

#[test]
fn index_only_par2_is_recognised_by_upstream() {
    if !ensure_par2_available_or_skip("index_only_par2_is_recognised_by_upstream") {
        return;
    }
    let dir = tempdir().unwrap();
    let data = deterministic_bytes(1, 32 * 1024);
    write_fixture(dir.path(), "hello.bin", &data);

    let src = SourceFile::scan(&dir.path().join("hello.bin"), b"hello.bin".to_vec(), 4096).unwrap();
    let archive = dir.path().join("recovery.par2");

    write_index_file(
        &CreateOptions {
            output: archive.clone(),
            slice_size: 4096,
            recovery_block_count: 0,
            volume_scheme: VolumeScheme::Single,
        },
        &[src],
    )
    .unwrap();

    let (status, stdout, stderr) = run_par2_verify(dir.path(), &archive);
    assert!(
        status.success(),
        "upstream par2 v rejected our index file\n\
         exit: {status}\n\
         stdout:\n{stdout}\n\
         stderr:\n{stderr}",
    );
    // Sanity: the upstream output should mention our filename.
    assert!(
        stdout.contains("hello.bin"),
        "par2 v stdout did not mention input file:\n{stdout}",
    );
}

#[test]
fn index_with_multiple_files_is_recognised_by_upstream() {
    if !ensure_par2_available_or_skip("index_with_multiple_files_is_recognised_by_upstream") {
        return;
    }
    let dir = tempdir().unwrap();
    write_fixture(dir.path(), "a.bin", &deterministic_bytes(11, 12_000));
    write_fixture(dir.path(), "b.bin", &deterministic_bytes(22, 5_555));
    write_fixture(dir.path(), "c.bin", &deterministic_bytes(33, 70_000));

    let mut srcs = Vec::new();
    for name in ["a.bin", "b.bin", "c.bin"] {
        srcs.push(
            SourceFile::scan(&dir.path().join(name), name.as_bytes().to_vec(), 4096).unwrap(),
        );
    }

    let archive = dir.path().join("set.par2");
    write_index_file(
        &CreateOptions {
            output: archive.clone(),
            slice_size: 4096,
            recovery_block_count: 0,
            volume_scheme: VolumeScheme::Single,
        },
        &srcs,
    )
    .unwrap();

    let (status, stdout, stderr) = run_par2_verify(dir.path(), &archive);
    assert!(
        status.success(),
        "upstream par2 v rejected our multi-file index\n\
         exit: {status}\n\
         stdout:\n{stdout}\n\
         stderr:\n{stderr}",
    );
    for name in ["a.bin", "b.bin", "c.bin"] {
        assert!(
            stdout.contains(name),
            "missing {name} in par2 v output:\n{stdout}"
        );
    }
}

/// End-to-end test via the `par2rust` CLI binary: invoke it as a subprocess to
/// generate the recovery set, then have upstream `par2 r` restore a corrupted
/// data file. Proves the CLI wiring works on a real path.
#[test]
fn cli_create_then_upstream_repair() {
    if !ensure_par2_available_or_skip("cli_create_then_upstream_repair") {
        return;
    }
    let dir = tempdir().unwrap();
    let original = deterministic_bytes(0xBA5EBA11, 32 * 1024);
    write_fixture(dir.path(), "doc.bin", &original);

    // Locate our compiled binary. `cargo test` builds it in `target/<profile>/par2rust`;
    // we look up the manifest's target dir via `CARGO_MANIFEST_DIR`.
    let bin = std::env::var_os("CARGO_BIN_EXE_par2rust")
        .map(PathBuf::from)
        .expect("cargo did not export CARGO_BIN_EXE_par2rust — are tests run via cargo?");

    let archive = dir.path().join("backup.par2");
    let status = Command::new(&bin)
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

/// End-to-end: create a recovery set, corrupt the data file, and have upstream
/// `par2 r` restore it byte-for-byte. This proves both the recovery slice math
/// and the volume-file framing are correct.
#[test]
fn corruption_repair_round_trip_against_upstream() {
    if !ensure_par2_available_or_skip("corruption_repair_round_trip_against_upstream") {
        return;
    }
    let dir = tempdir().unwrap();
    let original = deterministic_bytes(0xC0FFEE, 64 * 1024);
    write_fixture(dir.path(), "payload.bin", &original);

    let src = SourceFile::scan(
        &dir.path().join("payload.bin"),
        b"payload.bin".to_vec(),
        4096,
    )
    .unwrap();

    let archive = dir.path().join("recovery.par2");
    let files = run_create(
        &CreateOptions {
            output: archive.clone(),
            slice_size: 4096,
            recovery_block_count: 4,
            volume_scheme: VolumeScheme::Single,
        },
        &[src],
    )
    .unwrap();
    // index + 1 volume file
    assert_eq!(files.len(), 2);

    // Sanity: upstream verifies the pristine pair.
    let (status, stdout, _) = run_par2_verify(dir.path(), &archive);
    assert!(
        status.success(),
        "upstream rejected our recovery set BEFORE corruption:\n{stdout}",
    );

    // Corrupt one slice (offset 4096 .. 4096+200) by flipping bytes — within
    // the recovery budget of 4 blocks.
    let payload_path = dir.path().join("payload.bin");
    let mut bytes = std::fs::read(&payload_path).unwrap();
    for b in &mut bytes[4096..4096 + 200] {
        *b ^= 0xFF;
    }
    std::fs::write(&payload_path, &bytes).unwrap();
    assert_ne!(
        bytes, original,
        "corruption did not actually change the file"
    );

    // Upstream attempts repair using our recovery set.
    let (status, stdout, stderr) = run_par2_repair(dir.path(), &archive);
    assert!(
        status.success(),
        "upstream par2 r failed to repair using our recovery slices\n\
         stdout:\n{stdout}\n\
         stderr:\n{stderr}",
    );

    // The repaired file must equal the original.
    let restored = std::fs::read(&payload_path).unwrap();
    assert_eq!(
        restored, original,
        "repair returned wrong bytes — recovery slice math is incorrect",
    );
}

/// Multi-file: protect three files at once, corrupt one of them, and let
/// upstream repair using recovery slices that span all three.
#[test]
fn multi_file_corruption_repair_round_trip() {
    if !ensure_par2_available_or_skip("multi_file_corruption_repair_round_trip") {
        return;
    }
    let dir = tempdir().unwrap();
    let a = deterministic_bytes(1, 8 * 1024);
    let b = deterministic_bytes(2, 12 * 1024);
    let c = deterministic_bytes(3, 5 * 1024);
    write_fixture(dir.path(), "a.bin", &a);
    write_fixture(dir.path(), "b.bin", &b);
    write_fixture(dir.path(), "c.bin", &c);

    let mut srcs = Vec::new();
    for name in ["a.bin", "b.bin", "c.bin"] {
        srcs.push(
            SourceFile::scan(&dir.path().join(name), name.as_bytes().to_vec(), 4096).unwrap(),
        );
    }

    let archive = dir.path().join("multi.par2");
    run_create(
        &CreateOptions {
            output: archive.clone(),
            slice_size: 4096,
            recovery_block_count: 5,
            volume_scheme: VolumeScheme::Single,
        },
        &srcs,
    )
    .unwrap();

    // Damage `b.bin` — flip an entire slice.
    let b_path = dir.path().join("b.bin");
    let mut damaged = std::fs::read(&b_path).unwrap();
    for byte in &mut damaged[4096..8192] {
        *byte ^= 0x5A;
    }
    std::fs::write(&b_path, &damaged).unwrap();

    let (status, stdout, stderr) = run_par2_repair(dir.path(), &archive);
    assert!(
        status.success(),
        "repair failed:\nstdout:\n{stdout}\nstderr:\n{stderr}",
    );
    assert_eq!(std::fs::read(&b_path).unwrap(), b, "b.bin not restored");
    // The intact files must remain untouched.
    assert_eq!(std::fs::read(dir.path().join("a.bin")).unwrap(), a);
    assert_eq!(std::fs::read(dir.path().join("c.bin")).unwrap(), c);
}

/// Multi-volume distribution: produce a recovery set using the par2cmdline
/// exponential split, then verify upstream accepts both verify and repair.
/// Proves recovery exponents are assigned correctly across volumes with a
/// non-zero `first_exponent`.
#[test]
fn exponential_volume_split_verifies_and_repairs() {
    if !ensure_par2_available_or_skip("exponential_volume_split_verifies_and_repairs") {
        return;
    }
    let dir = tempdir().unwrap();
    let original = deterministic_bytes(0xDEAD_BEEF, 64 * 1024);
    write_fixture(dir.path(), "payload.bin", &original);

    let src = SourceFile::scan(
        &dir.path().join("payload.bin"),
        b"payload.bin".to_vec(),
        4096,
    )
    .unwrap();

    let archive = dir.path().join("recovery.par2");
    let files = run_create(
        &CreateOptions {
            output: archive.clone(),
            slice_size: 4096,
            recovery_block_count: 10,
            volume_scheme: VolumeScheme::Exponential,
        },
        &[src],
    )
    .unwrap();
    // Expected layout: index + 5 volume files (1, 1, 2, 4, 2).
    assert_eq!(
        files.len(),
        6,
        "expected 1 index + 5 volume files, got {files:?}"
    );
    let names: Vec<String> = files
        .iter()
        .skip(1)
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        vec![
            "recovery.vol0+1.par2",
            "recovery.vol1+1.par2",
            "recovery.vol2+2.par2",
            "recovery.vol4+4.par2",
            "recovery.vol8+2.par2",
        ]
    );

    // Upstream verifies the pristine pair.
    let (status, stdout, stderr) = run_par2_verify(dir.path(), &archive);
    assert!(
        status.success(),
        "upstream rejected our multi-volume recovery set\n\
         stdout:\n{stdout}\n\
         stderr:\n{stderr}",
    );

    // Corrupt one slice. Recovery budget = 10, well above what we need.
    let payload_path = dir.path().join("payload.bin");
    let mut bytes = std::fs::read(&payload_path).unwrap();
    for b in &mut bytes[8192..8192 + 400] {
        *b ^= 0xA5;
    }
    std::fs::write(&payload_path, &bytes).unwrap();

    let (status, stdout, stderr) = run_par2_repair(dir.path(), &archive);
    assert!(
        status.success(),
        "upstream repair failed using multi-volume recovery set\n\
         stdout:\n{stdout}\n\
         stderr:\n{stderr}",
    );
    assert_eq!(
        std::fs::read(&payload_path).unwrap(),
        original,
        "repair returned wrong bytes — recovery exponents across volumes are incorrect",
    );
}

/// Deeper exponential split: 50 recovery blocks produce 7 volume files
/// (1, 1, 2, 4, 8, 16, 18). Exercises larger first_exponent values (up to 32)
/// to catch any off-by-one in the per-volume exponent bookkeeping.
#[test]
fn exponential_split_with_deep_chain_repairs() {
    if !ensure_par2_available_or_skip("exponential_split_with_deep_chain_repairs") {
        return;
    }
    let dir = tempdir().unwrap();
    let original = deterministic_bytes(0x1234_5678, 256 * 1024);
    write_fixture(dir.path(), "payload.bin", &original);

    let src = SourceFile::scan(
        &dir.path().join("payload.bin"),
        b"payload.bin".to_vec(),
        4096,
    )
    .unwrap();

    let archive = dir.path().join("recovery.par2");
    let files = run_create(
        &CreateOptions {
            output: archive.clone(),
            slice_size: 4096,
            recovery_block_count: 50,
            volume_scheme: VolumeScheme::Exponential,
        },
        &[src],
    )
    .unwrap();
    let names: Vec<String> = files
        .iter()
        .skip(1)
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    // 1 + 1 + 2 + 4 + 8 + 16 + 18 = 50 across 7 volumes.
    assert_eq!(
        names,
        vec![
            "recovery.vol0+1.par2",
            "recovery.vol1+1.par2",
            "recovery.vol2+2.par2",
            "recovery.vol4+4.par2",
            "recovery.vol8+8.par2",
            "recovery.vol16+16.par2",
            "recovery.vol32+18.par2",
        ],
        "deeper exponential split produced an unexpected volume layout"
    );

    // Verify pristine, then corrupt and repair through the multi-volume set.
    let (status, _, _) = run_par2_verify(dir.path(), &archive);
    assert!(status.success(), "upstream rejected 7-volume recovery set");

    let payload_path = dir.path().join("payload.bin");
    let mut bytes = std::fs::read(&payload_path).unwrap();
    // Damage two distant regions to force the repair to use blocks from
    // multiple volume files at once.
    for b in &mut bytes[16_384..16_384 + 800] {
        *b ^= 0xC3;
    }
    for b in &mut bytes[180_000..180_000 + 1200] {
        *b = b.wrapping_add(97);
    }
    std::fs::write(&payload_path, &bytes).unwrap();

    let (status, stdout, stderr) = run_par2_repair(dir.path(), &archive);
    assert!(
        status.success(),
        "repair failed across 7-volume set:\nstdout:\n{stdout}\nstderr:\n{stderr}",
    );
    assert_eq!(std::fs::read(&payload_path).unwrap(), original);
}

/// Caller-supplied volume layout via `VolumeScheme::Explicit`. Proves that
/// arbitrary (first_exponent, count) ranges survive end-to-end — upstream
/// reads recovery blocks from each volume by their stored exponent, so any
/// off-by-one in our per-volume exponent assignment would surface here.
#[test]
fn explicit_volume_layout_repairs_through_upstream() {
    if !ensure_par2_available_or_skip("explicit_volume_layout_repairs_through_upstream") {
        return;
    }
    let dir = tempdir().unwrap();
    let original = deterministic_bytes(0xC0DE_C0DE, 96 * 1024);
    write_fixture(dir.path(), "payload.bin", &original);

    let src = SourceFile::scan(
        &dir.path().join("payload.bin"),
        b"payload.bin".to_vec(),
        4096,
    )
    .unwrap();

    let archive = dir.path().join("recovery.par2");
    // Non-power-of-two split that the Exponential scheme would never produce.
    let layout = vec![2u32, 3, 5];
    let files = run_create(
        &CreateOptions {
            output: archive.clone(),
            slice_size: 4096,
            recovery_block_count: layout.iter().sum(),
            volume_scheme: VolumeScheme::Explicit(layout.clone()),
        },
        &[src],
    )
    .unwrap();
    let names: Vec<String> = files
        .iter()
        .skip(1)
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        vec![
            "recovery.vol0+2.par2",
            "recovery.vol2+3.par2",
            "recovery.vol5+5.par2",
        ]
    );

    let (status, stdout, stderr) = run_par2_verify(dir.path(), &archive);
    assert!(
        status.success(),
        "upstream rejected explicit-layout recovery set\n\
         stdout:\n{stdout}\n\
         stderr:\n{stderr}",
    );

    // Corrupt one slice and let upstream repair using the explicit split.
    let payload_path = dir.path().join("payload.bin");
    let mut bytes = std::fs::read(&payload_path).unwrap();
    for b in &mut bytes[40_000..40_000 + 500] {
        *b ^= 0x5A;
    }
    std::fs::write(&payload_path, &bytes).unwrap();

    let (status, stdout, stderr) = run_par2_repair(dir.path(), &archive);
    assert!(
        status.success(),
        "repair failed with explicit layout:\nstdout:\n{stdout}\nstderr:\n{stderr}",
    );
    assert_eq!(std::fs::read(&payload_path).unwrap(), original);
}

/// CLI `--single-volume` flag: with the flag, par2rust must emit exactly one
/// `vol0+N.par2` regardless of the (otherwise exponential-by-default) CLI
/// behaviour. Repair must still succeed.
#[test]
fn cli_single_volume_flag_emits_one_volume_file() {
    if !ensure_par2_available_or_skip("cli_single_volume_flag_emits_one_volume_file") {
        return;
    }
    let dir = tempdir().unwrap();
    let original = deterministic_bytes(0xABCD_1234, 32 * 1024);
    write_fixture(dir.path(), "doc.bin", &original);

    let bin = std::env::var_os("CARGO_BIN_EXE_par2rust")
        .map(PathBuf::from)
        .expect("cargo did not export CARGO_BIN_EXE_par2rust — are tests run via cargo?");

    let archive = dir.path().join("backup.par2");
    let status = Command::new(&bin)
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
