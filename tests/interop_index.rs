//! Interop tests: upstream `par2 v` must accept index files produced by
//! par2rust (no recovery slices).

mod common;

use par2rust::{write_index_file, CreateOptions, SourceFile, VolumeScheme};
use tempfile::tempdir;

use common::{deterministic_bytes, ensure_par2_available_or_skip, run_par2_verify, write_fixture};

#[test]
fn upstream_accepts_index_only_archive() {
    if !ensure_par2_available_or_skip("upstream_accepts_index_only_archive") {
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
            comments: Vec::new(),
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
    assert!(
        stdout.contains("hello.bin"),
        "par2 v stdout did not mention input file:\n{stdout}",
    );
}

#[test]
fn upstream_accepts_multi_file_index() {
    if !ensure_par2_available_or_skip("upstream_accepts_multi_file_index") {
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
            comments: Vec::new(),
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
