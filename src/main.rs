use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;

use clap::{Parser, Subcommand};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use par2rust::{
    run_create_with_progress, CreateOptions, Par2Error, ProgressEvent, ProgressReporter,
    SourceFile, VolumeScheme, MAX_RECOVERY_BLOCKS,
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
    /// Block (slice) size in bytes. Must be a positive multiple of 4. An
    /// optional suffix selects binary units: `b`/`B` = bytes (no-op),
    /// `k`/`K` = KiB, `m`/`M` = MiB, `g`/`G` = GiB. E.g. `-s 768000b`,
    /// `-s 750K`, `-s 1M`.
    #[arg(
        short = 's',
        long = "slice-size",
        default_value = "4096",
        value_parser = parse_slice_size
    )]
    slice_size: u64,

    /// Number of recovery blocks to generate.
    #[arg(short = 'c', long = "recovery-count", default_value_t = 0)]
    recovery_count: u32,

    /// par2cmdline `-r`: level of redundancy. Either a percentage of the input
    /// (e.g. `-r10` = 10%) or a target recovery-data size with a unit prefix:
    /// `-rk<N>`, `-rm<N>`, `-rg<N>` for KiB/MiB/GiB. Mutually exclusive with `-c`.
    #[arg(
        short = 'r',
        long = "redundancy",
        value_name = "PCT|[gmk]N",
        conflicts_with = "recovery_count"
    )]
    redundancy: Option<Redundancy>,

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
    if args.threads > 0 {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
        {
            eprintln!("par2rust: warning: could not configure thread pool: {e}");
        }
    }

    let reporter = CliReporter::new();

    // Scan every input file before writing anything.
    let mut sources = Vec::with_capacity(args.inputs.len());
    for input in &args.inputs {
        let display = display_name_for(input);
        let src = SourceFile::scan_with_progress(input, display, args.slice_size, Some(&reporter))?;
        sources.push(src);
    }

    let recovery_count = match args.redundancy {
        Some(r) => resolve_redundancy(r, &sources, args.slice_size)?,
        None => args.recovery_count,
    };
    if recovery_count > MAX_RECOVERY_BLOCKS {
        return Err(Par2Error::TooManyRecoveryBlocks(recovery_count));
    }

    let volume_scheme = build_volume_scheme(&args, &sources, recovery_count)?;

    let written = run_create_with_progress(
        &CreateOptions {
            output: args.archive.clone(),
            slice_size: args.slice_size,
            recovery_block_count: recovery_count,
            volume_scheme,
        },
        &sources,
        Some(&reporter),
    )?;

    reporter.finish();

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
    recovery_count: u32,
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
            .unwrap_or_else(|| default_uniform_count(recovery_count));
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

/// Two-bar progress reporter built on `indicatif`: one bar for the scan
/// phase (slices across all input files) and one for the encode phase
/// (input-block work across all recovery volumes).
struct CliReporter {
    multi: MultiProgress,
    scan: Mutex<Option<ProgressBar>>,
    scan_base: Mutex<u64>,
    encode: Mutex<Option<ProgressBar>>,
}

impl CliReporter {
    fn new() -> Self {
        Self {
            multi: MultiProgress::new(),
            scan: Mutex::new(None),
            scan_base: Mutex::new(0),
            encode: Mutex::new(None),
        }
    }

    fn finish(&self) {
        if let Some(pb) = self.scan.lock().unwrap().take() {
            pb.finish_and_clear();
        }
        if let Some(pb) = self.encode.lock().unwrap().take() {
            pb.finish_and_clear();
        }
    }

    fn ensure_scan_bar(&self, total: u64) -> ProgressBar {
        let mut guard = self.scan.lock().unwrap();
        if let Some(pb) = guard.as_ref() {
            return pb.clone();
        }
        let pb = self.multi.add(ProgressBar::new(total));
        pb.set_style(
            ProgressStyle::with_template(
                "scan   [{bar:30.cyan/blue}] {pos}/{len} slices ({eta}) {wide_msg}",
            )
            .unwrap()
            .progress_chars("=> "),
        );
        *guard = Some(pb.clone());
        pb
    }

    fn ensure_encode_bar(&self) -> ProgressBar {
        let mut guard = self.encode.lock().unwrap();
        if let Some(pb) = guard.as_ref() {
            return pb.clone();
        }
        let pb = self.multi.add(ProgressBar::new(0));
        pb.set_style(
            ProgressStyle::with_template(
                "encode [{bar:30.green/blue}] {pos}/{len} blocks ({eta}) {wide_msg}",
            )
            .unwrap()
            .progress_chars("=> "),
        );
        *guard = Some(pb.clone());
        pb
    }
}

impl ProgressReporter for CliReporter {
    fn on_event(&self, event: ProgressEvent<'_>) {
        match event {
            ProgressEvent::ScanStarted { path, total_slices } => {
                let pb = self.ensure_scan_bar(0);
                pb.inc_length(total_slices);
                pb.set_message(format!("{}", path.display()));
            }
            ProgressEvent::ScanProgress { slices_done, .. } => {
                if let Some(pb) = self.scan.lock().unwrap().as_ref() {
                    let base = *self.scan_base.lock().unwrap();
                    pb.set_position(base + slices_done);
                }
            }
            ProgressEvent::ScanCompleted { .. } => {
                // Advance the baseline so the next file picks up where this
                // one left off on the cumulative bar.
                if let Some(pb) = self.scan.lock().unwrap().as_ref() {
                    *self.scan_base.lock().unwrap() = pb.position();
                }
            }
            ProgressEvent::EncodeStarted {
                volume_index,
                total_volumes,
                input_blocks,
                ..
            } => {
                let pb = self.ensure_encode_bar();
                pb.set_length(input_blocks);
                pb.set_position(0);
                pb.set_message(format!("volume {}/{}", volume_index + 1, total_volumes));
            }
            ProgressEvent::EncodeProgress {
                input_block_done, ..
            } => {
                if let Some(pb) = self.encode.lock().unwrap().as_ref() {
                    pb.set_position(input_block_done);
                }
            }
            ProgressEvent::EncodeCompleted { .. } => {}
            ProgressEvent::IndexWritten { .. } | ProgressEvent::VolumeWritten { .. } => {}
            _ => {}
        }
    }
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

/// Parse a `--slice-size` argument. Accepts a bare integer (bytes) or an
/// integer with a trailing unit suffix: `b`/`B` (bytes), `k`/`K` (KiB),
/// `m`/`M` (MiB), `g`/`G` (GiB). This matches parpar's convention, so
/// wrappers built for parpar (e.g. Postie passing `768000b`) work unchanged.
fn parse_slice_size(s: &str) -> Result<u64, String> {
    if s.is_empty() {
        return Err("slice size is empty".into());
    }
    let last = *s.as_bytes().last().unwrap();
    let (digits, shift) = match last {
        b'b' | b'B' => (&s[..s.len() - 1], 0u32),
        b'k' | b'K' => (&s[..s.len() - 1], 10u32),
        b'm' | b'M' => (&s[..s.len() - 1], 20u32),
        b'g' | b'G' => (&s[..s.len() - 1], 30u32),
        _ => (s, 0u32),
    };
    if digits.is_empty() {
        return Err(format!("slice size '{s}' is missing a numeric value"));
    }
    let n: u64 = digits
        .parse()
        .map_err(|_| format!("slice size '{s}' has invalid number '{digits}'"))?;
    let multiplier = 1u64 << shift;
    n.checked_mul(multiplier)
        .ok_or_else(|| format!("slice size '{s}' overflows u64"))
}

/// par2cmdline `-r` value. Either a percentage of the total input data, or a
/// target recovery-data size in bytes (binary units, matching the convention
/// used by par2cmdline and friends).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Redundancy {
    Percent(u32),
    TargetBytes(u64),
}

impl std::str::FromStr for Redundancy {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        if s.is_empty() {
            return Err("redundancy value is empty".into());
        }
        let first = s.as_bytes()[0];
        let unit_shift: Option<u32> = match first {
            b'k' | b'K' => Some(10),
            b'm' | b'M' => Some(20),
            b'g' | b'G' => Some(30),
            _ => None,
        };
        if let Some(shift) = unit_shift {
            let rest = &s[1..];
            if rest.is_empty() {
                return Err(format!("redundancy '{s}' is missing a numeric value"));
            }
            let n: u64 = rest
                .parse()
                .map_err(|_| format!("redundancy '{s}' has invalid number '{rest}'"))?;
            let bytes = n
                .checked_shl(shift)
                .ok_or_else(|| format!("redundancy '{s}' overflows u64"))?;
            Ok(Redundancy::TargetBytes(bytes))
        } else {
            let n: u32 = s
                .parse()
                .map_err(|_| format!("redundancy '{s}' is not a percentage or [gmk]N value"))?;
            Ok(Redundancy::Percent(n))
        }
    }
}

/// Translate a `-r` redundancy specification into a concrete recovery-block
/// count, given the just-scanned input files and the chosen slice size.
fn resolve_redundancy(
    r: Redundancy,
    sources: &[SourceFile],
    slice_size: u64,
) -> Result<u32, Par2Error> {
    let total_input_blocks: u64 = sources.iter().map(|s| s.slice_checksums.len() as u64).sum();

    let blocks: u64 = match r {
        Redundancy::Percent(p) => {
            let p = u64::from(p);
            let numerator = total_input_blocks.saturating_mul(p);
            numerator.div_ceil(100)
        }
        Redundancy::TargetBytes(b) => b.div_ceil(slice_size.max(1)),
    };

    if blocks == 0 {
        return Err(Par2Error::InvalidVolumeScheme(format!(
            "-r {r:?} resolves to 0 recovery blocks (input is too small or value too low)"
        )));
    }
    let blocks_u32: u32 = blocks.try_into().unwrap_or(u32::MAX);
    Ok(blocks_u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn parses_percent() {
        assert_eq!(Redundancy::from_str("10").unwrap(), Redundancy::Percent(10));
        assert_eq!(Redundancy::from_str("0").unwrap(), Redundancy::Percent(0));
        assert_eq!(
            Redundancy::from_str("250").unwrap(),
            Redundancy::Percent(250)
        );
    }

    #[test]
    fn parses_size_targets() {
        assert_eq!(
            Redundancy::from_str("k500").unwrap(),
            Redundancy::TargetBytes(500 * 1024)
        );
        assert_eq!(
            Redundancy::from_str("m10").unwrap(),
            Redundancy::TargetBytes(10 * 1024 * 1024)
        );
        assert_eq!(
            Redundancy::from_str("g2").unwrap(),
            Redundancy::TargetBytes(2 * 1024 * 1024 * 1024)
        );
        // Case-insensitive on the unit letter.
        assert_eq!(
            Redundancy::from_str("M1").unwrap(),
            Redundancy::TargetBytes(1024 * 1024)
        );
    }

    #[test]
    fn rejects_bad_input() {
        assert!(Redundancy::from_str("").is_err());
        assert!(Redundancy::from_str("abc").is_err());
        assert!(Redundancy::from_str("k").is_err());
        assert!(Redundancy::from_str("-5").is_err());
    }

    #[test]
    fn slice_size_bare_integer() {
        assert_eq!(parse_slice_size("4096").unwrap(), 4096);
        assert_eq!(parse_slice_size("768000").unwrap(), 768000);
    }

    #[test]
    fn slice_size_byte_suffix() {
        assert_eq!(parse_slice_size("768000b").unwrap(), 768000);
        assert_eq!(parse_slice_size("768000B").unwrap(), 768000);
    }

    #[test]
    fn slice_size_binary_units() {
        assert_eq!(parse_slice_size("750K").unwrap(), 750 * 1024);
        assert_eq!(parse_slice_size("750k").unwrap(), 750 * 1024);
        assert_eq!(parse_slice_size("1M").unwrap(), 1 << 20);
        assert_eq!(parse_slice_size("1m").unwrap(), 1 << 20);
        assert_eq!(parse_slice_size("2G").unwrap(), 2u64 << 30);
        assert_eq!(parse_slice_size("2g").unwrap(), 2u64 << 30);
    }

    #[test]
    fn slice_size_rejects_bad_input() {
        assert!(parse_slice_size("").is_err());
        assert!(parse_slice_size("b").is_err());
        assert!(parse_slice_size("K").is_err());
        assert!(parse_slice_size("abc").is_err());
        assert!(parse_slice_size("12x").is_err());
        assert!(parse_slice_size("-5").is_err());
    }

    #[test]
    fn slice_size_rejects_overflow() {
        assert!(parse_slice_size("99999999999G").is_err());
    }

    fn fake_sources_with_blocks(total_blocks: usize) -> Vec<SourceFile> {
        use par2rust::source::SliceChecksum;
        let slice_checksums = vec![
            SliceChecksum {
                md5: [0u8; 16],
                crc32: 0,
            };
            total_blocks
        ];
        vec![SourceFile {
            name: b"x".to_vec(),
            path: PathBuf::from("x"),
            length: 0,
            hash_full: [0u8; 16],
            hash16k: [0u8; 16],
            file_id: [0u8; 16],
            slice_checksums,
        }]
    }

    #[test]
    fn resolves_percent() {
        let sources = fake_sources_with_blocks(100);
        assert_eq!(
            resolve_redundancy(Redundancy::Percent(10), &sources, 4096).unwrap(),
            10
        );
        // Non-exact percentages round up.
        let sources = fake_sources_with_blocks(33);
        assert_eq!(
            resolve_redundancy(Redundancy::Percent(10), &sources, 4096).unwrap(),
            4 // ceil(33 * 10 / 100) = ceil(3.3) = 4
        );
    }

    #[test]
    fn resolves_target_bytes() {
        let sources = fake_sources_with_blocks(100);
        assert_eq!(
            resolve_redundancy(Redundancy::TargetBytes(8192), &sources, 4096).unwrap(),
            2
        );
        assert_eq!(
            resolve_redundancy(Redundancy::TargetBytes(8193), &sources, 4096).unwrap(),
            3
        );
    }

    #[test]
    fn rejects_zero_result() {
        let sources = fake_sources_with_blocks(1);
        let err = resolve_redundancy(Redundancy::Percent(0), &sources, 4096).unwrap_err();
        assert!(matches!(err, Par2Error::InvalidVolumeScheme(_)));
    }
}
