//! Interop tests: create recovery sets via the par2rust library API, corrupt
//! the protected data, then have upstream `par2 r` restore it byte-for-byte.
//! Proves both the recovery slice math and volume-file framing are correct
//! across the different `VolumeScheme`s.

mod common;

use par2rust::{run_create, CreateOptions, SourceFile, VolumeScheme};
use tempfile::tempdir;

use common::{
    deterministic_bytes, ensure_par2_available_or_skip, run_par2_repair, run_par2_verify,
    write_fixture,
};

#[test]
fn single_file_repair_round_trip() {
    if !ensure_par2_available_or_skip("single_file_repair_round_trip") {
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
            comments: Vec::new(),
        },
        &[src],
    )
    .unwrap();
    // index + 1 volume file
    assert_eq!(files.len(), 2);

    let (status, stdout, _) = run_par2_verify(dir.path(), &archive);
    assert!(
        status.success(),
        "upstream rejected our recovery set BEFORE corruption:\n{stdout}",
    );

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

    let (status, stdout, stderr) = run_par2_repair(dir.path(), &archive);
    assert!(
        status.success(),
        "upstream par2 r failed to repair using our recovery slices\n\
         stdout:\n{stdout}\n\
         stderr:\n{stderr}",
    );

    let restored = std::fs::read(&payload_path).unwrap();
    assert_eq!(
        restored, original,
        "repair returned wrong bytes — recovery slice math is incorrect",
    );
}

#[test]
fn multi_file_repair_round_trip() {
    if !ensure_par2_available_or_skip("multi_file_repair_round_trip") {
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
            comments: Vec::new(),
        },
        &srcs,
    )
    .unwrap();

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
    assert_eq!(std::fs::read(dir.path().join("a.bin")).unwrap(), a);
    assert_eq!(std::fs::read(dir.path().join("c.bin")).unwrap(), c);
}

#[test]
fn exponential_split_repair_round_trip() {
    if !ensure_par2_available_or_skip("exponential_split_repair_round_trip") {
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
            comments: Vec::new(),
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

    let (status, stdout, stderr) = run_par2_verify(dir.path(), &archive);
    assert!(
        status.success(),
        "upstream rejected our multi-volume recovery set\n\
         stdout:\n{stdout}\n\
         stderr:\n{stderr}",
    );

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

#[test]
fn deep_exponential_split_repair_round_trip() {
    if !ensure_par2_available_or_skip("deep_exponential_split_repair_round_trip") {
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
            comments: Vec::new(),
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

#[test]
fn explicit_volume_layout_repair_round_trip() {
    if !ensure_par2_available_or_skip("explicit_volume_layout_repair_round_trip") {
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
            comments: Vec::new(),
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
