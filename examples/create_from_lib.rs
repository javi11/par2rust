//! End-to-end demonstration of calling par2rust as a library.
//!
//! Run with: `cargo run --example create_from_lib`
//!
//! Creates a small data file in a temporary directory, generates a PAR2
//! recovery set for it via the library API, and prints the resulting files.

use std::fs;

use par2rust::{run_create, CreateOptions, SourceFile};

fn main() -> par2rust::Result<()> {
    let dir = tempfile::tempdir().expect("create tempdir");
    let data_path = dir.path().join("data.bin");
    fs::write(&data_path, b"par2rust library API example payload")?;

    let display_name = data_path.file_name().unwrap().as_encoded_bytes().to_vec();
    let source = SourceFile::scan(&data_path, display_name, 4096)?;

    let output = dir.path().join("backup.par2");
    let written = run_create(
        &CreateOptions {
            output: output.clone(),
            slice_size: 4096,
            recovery_block_count: 10,
            ..Default::default()
        },
        &[source],
    )?;

    println!(
        "Generated {} PAR2 file(s) in {}:",
        written.len(),
        dir.path().display()
    );
    for p in &written {
        let size = fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        println!("  {} ({} bytes)", p.display(), size);
    }
    Ok(())
}
