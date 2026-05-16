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

    /// par2cmdline `-u`: distribute recovery blocks uniformly across volume
    /// files instead of exponential growth. Combine with `-n` to set the
    /// volume count explicitly (defaults to par2cmdline's heuristic).
    #[arg(short = 'u', long = "uniform", default_value_t = false)]
    uniform: bool,

    /// par2cmdline `-l`: cap each volume file's size so it does not exceed
    /// the largest source file. Composes with `-u` and with the default
    /// exponential layout.
    #[arg(short = 'l', long = "limit-size", default_value_t = false)]
    limit_size: bool,

    /// par2cmdline `-n<count>`: explicit number of recovery volume files.
    /// Currently honoured only when combined with `-u`.
    #[arg(short = 'n', long = "volume-count")]
    volume_count: Option<u32>,

    /// Number of worker threads. 0 = auto (one per logical CPU).
    #[arg(short = 't', long = "threads", default_value_t = 0)]
    threads: usize,

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

    if args.threads > 0 {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
        {
            eprintln!("par2rust: warning: could not configure thread pool: {e}");
        }
    }

    // Scan every input file before writing anything.
    let mut sources = Vec::with_capacity(args.inputs.len());
    for input in &args.inputs {
        let display = display_name_for(input);
        let src = SourceFile::scan(input, display, args.slice_size)?;
        sources.push(src);
    }

    let volume_scheme = build_volume_scheme(&args, &sources)?;

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

/// par2cmdline's heuristic for the default number of uniform recovery volumes
/// when `-u` is passed without `-n`. Upstream caps the count at 15 by default.
fn default_uniform_count(recovery_count: u32) -> u32 {
    recovery_count.clamp(1, 15)
}

/// Assemble a [`VolumeScheme`] from the CLI flags. Mirrors par2cmdline's
/// flag interactions:
///   - `--single-volume`: standalone, conflicts with `-u`/`-l`/`-n`.
///   - `-u` → `Uniform { count }` where `count` is `-n` or the default.
///   - `-l` → wraps the chosen inner scheme in `Limited` with the cap
///     derived from the largest source file.
fn build_volume_scheme(
    args: &CreateArgs,
    sources: &[SourceFile],
) -> Result<VolumeScheme, Par2Error> {
    if args.single_volume && (args.uniform || args.limit_size || args.volume_count.is_some()) {
        return Err(Par2Error::InvalidVolumeScheme(
            "--single-volume cannot be combined with -u, -l, or -n".into(),
        ));
    }
    if args.single_volume {
        return Ok(VolumeScheme::Single);
    }
    if args.volume_count.is_some() && !args.uniform {
        return Err(Par2Error::InvalidVolumeScheme(
            "-n/--volume-count currently requires -u/--uniform".into(),
        ));
    }

    let inner = if args.uniform {
        let count = args
            .volume_count
            .unwrap_or_else(|| default_uniform_count(args.recovery_count));
        VolumeScheme::Uniform { count }
    } else {
        VolumeScheme::Exponential
    };

    if !args.limit_size {
        return Ok(inner);
    }

    let largest = sources.iter().map(|s| s.length).max().unwrap_or(0);
    if largest == 0 {
        return Err(Par2Error::InvalidVolumeScheme(
            "-l/--limit-size requires at least one non-empty source file".into(),
        ));
    }
    let cap_u64 = largest / args.slice_size;
    let cap: u32 = cap_u64.max(1).try_into().unwrap_or(u32::MAX);
    Ok(VolumeScheme::Limited {
        max_blocks_per_volume: cap,
        inner: Box::new(inner),
    })
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
