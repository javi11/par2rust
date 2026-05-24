//! MD5 scan throughput bench.
//!
//! Measures the realistic PAR2 scan pattern: hash N independent
//! per-slice MD5s of a fixed slice size, parallelised across rayon
//! workers (mirroring `scan_via_mmap`'s inner loop). Also includes a
//! single-thread baseline so we can see how much of the win is rayon
//! parallelism vs SIMD per-stream throughput.
//!
//! ```bash
//! cargo bench --bench md5_scan                                # md-5 (default)
//! cargo bench --bench md5_scan --features fast-md5            # openssl asm
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use par2rust::format::md5_of as digest;
use rayon::prelude::*;

// Workload: 200 slices of 1 MiB each = 200 MiB total, matches the
// encoder bench's working set so the two are directly comparable.
const SLICE_BYTES: usize = 1024 * 1024;
const SLICE_COUNT: usize = 200;

fn bench_md5_scan(c: &mut Criterion) {
    // One big shared buffer split into SLICE_COUNT slices — same shape
    // as the mmap-backed `par_chunks(slice_size)` in `scan_via_mmap`.
    let buf: Vec<u8> = (0..SLICE_BYTES * SLICE_COUNT)
        .map(|i| (i & 0xFF) as u8)
        .collect();
    let total_bytes = (SLICE_BYTES * SLICE_COUNT) as u64;

    let mut g = c.benchmark_group("md5_scan");
    g.throughput(Throughput::Bytes(total_bytes));
    g.sample_size(20); // 200 MiB hashing is slow; keep iteration count modest

    // Single-thread serial: hash every slice on the calling thread. Lower
    // bound on rayon parallelism — shows raw per-stream throughput.
    g.bench_function("serial", |b| {
        b.iter(|| {
            let mut digests = Vec::with_capacity(SLICE_COUNT);
            for chunk in buf.chunks(SLICE_BYTES) {
                digests.push(digest(black_box(chunk)));
            }
            digests
        });
    });

    // Rayon parallel: mirrors `scan_via_mmap`'s `par_chunks(slice_size)`.
    // This is what par2rust actually runs today.
    g.bench_function("rayon", |b| {
        b.iter(|| {
            buf.par_chunks(SLICE_BYTES)
                .map(|s| digest(black_box(s)))
                .collect::<Vec<_>>()
        });
    });

    // 4-lane NEON multi-buffer MD5 (aarch64 only). The realistic
    // integration: batch slices in groups of 4 and lane them through
    // `digest4`. Caller threads collect digests in scan order; the
    // last incomplete batch (< 4 slices) would fall back to scalar
    // — here SLICE_COUNT=200 is divisible by 4 so the fallback is
    // exercised only by `_serial_mb` for the same workload.
    #[cfg(target_arch = "aarch64")]
    g.bench_function("serial_mb", |b| {
        b.iter(|| {
            let mut digests = Vec::with_capacity(SLICE_COUNT);
            let slices: Vec<&[u8]> = buf.chunks(SLICE_BYTES).collect();
            for batch in slices.chunks(4) {
                let mb = par2rust::md5_mb_neon::digest4([batch[0], batch[1], batch[2], batch[3]]);
                digests.extend_from_slice(black_box(&mb));
            }
            digests
        });
    });

    // Rayon + multi-buffer combined: outer rayon distributes batches
    // of 4 slices across workers, each worker SIMD-lanes its batch.
    // This is the target integration for scan_via_mmap.
    #[cfg(target_arch = "aarch64")]
    g.bench_function("rayon_mb", |b| {
        b.iter(|| {
            let slices: Vec<&[u8]> = buf.chunks(SLICE_BYTES).collect();
            slices
                .par_chunks(4)
                .flat_map_iter(|batch| {
                    let mb =
                        par2rust::md5_mb_neon::digest4([batch[0], batch[1], batch[2], batch[3]]);
                    mb.into_iter()
                })
                .collect::<Vec<_>>()
        });
    });

    g.finish();
}

criterion_group!(benches, bench_md5_scan);
criterion_main!(benches);
