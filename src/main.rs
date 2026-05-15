use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use par2rust::{
    run_create, CreateOptions, Par2Error, SourceFile, VolumeScheme, MAX_RECOVERY_BLOCKS,
};

#[derive(Parser, Debug)]
#[command(
    name = "par2rust",
    version,
    about = "Rust port of par2cmdline (create only).",
    long_about = "par2rust generates PAR2 recovery files. Output is byte-compatible \
                  with par2cmdline's PAR 2.0 format."
)]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand, Debug)]
enum CliCommand {
    /// Create a PAR2 recovery set. Alias: c
    #[command(alias = "c")]
    Create(CreateArgs),
}

#[derive(clap::Args, Debug)]
struct CreateArgs {
    /// Block (slice) size in bytes. Must be a positive multiple of 4.
    #[arg(short = 's', long = "slice-size", default_value_t = 4096)]
    slice_size: u64,

    /// Number of recovery blocks to generate.
    #[arg(short = 'c', long = "recovery-count", default_value_t = 0)]
    recovery_count: u32,

    /// Emit a single `vol0+N.par2` containing all recovery blocks instead of
    /// par2cmdline's default exponential split (`vol0+1`, `vol1+1`, `vol2+2`,
    /// `vol4+4`, …).
    #[arg(long = "single-volume", default_value_t = false)]
    single_volume: bool,

    /// Output PAR2 archive name. Volume files are derived from this name.
    archive: PathBuf,

    /// Input files to protect.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        CliCommand::Create(args) => match run_create_cli(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("par2rust: error: {e}");
                ExitCode::from(1)
            }
        },
    }
}

fn run_create_cli(args: CreateArgs) -> Result<(), Par2Error> {
    if args.recovery_count > MAX_RECOVERY_BLOCKS {
        return Err(Par2Error::TooManyRecoveryBlocks(args.recovery_count));
    }

    // Scan every input file before writing anything.
    let mut sources = Vec::with_capacity(args.inputs.len());
    for input in &args.inputs {
        let display = display_name_for(input);
        let src = SourceFile::scan(input, display, args.slice_size)?;
        sources.push(src);
    }

    let volume_scheme = if args.single_volume {
        VolumeScheme::Single
    } else {
        VolumeScheme::Exponential
    };

    let written = run_create(
        &CreateOptions {
            output: args.archive.clone(),
            slice_size: args.slice_size,
            recovery_block_count: args.recovery_count,
            volume_scheme,
        },
        &sources,
    )?;

    println!(
        "Wrote {} file{}:",
        written.len(),
        if written.len() == 1 { "" } else { "s" }
    );
    for p in &written {
        let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        println!("  {} ({} bytes)", p.display(), size);
    }
    Ok(())
}

/// Pick the bytes to record as the filename inside the PAR2 packet. We use the
/// file's basename so the recovery set is portable across machines — the user
/// typing `par2rust c set.par2 /home/me/data/x.bin` gets a `.par2` that refers
/// to `x.bin`, not the full path.
fn display_name_for(input: &Path) -> Vec<u8> {
    input
        .file_name()
        .map(|n| n.as_encoded_bytes().to_vec())
        .unwrap_or_else(|| input.as_os_str().as_encoded_bytes().to_vec())
}
