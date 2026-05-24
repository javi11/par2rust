//! Encoder inner-loop benchmark.
//!
//! Simulates the realistic PAR2 encode pattern: one input slice
//! (`slice_size = 256 KiB`) XOR-multiplied into `N` recovery buffers
//! sequentially per input slice. This is where L2 cache-blocking matters
//! — the `gf_kernel` bench measures the raw kernel at one buffer pair,
//! which doesn't exercise the multi-buffer streaming pattern.
//!
//! Default tiling is enabled. Set `PAR2RUST_L2_BLOCK_BYTES=999999999`
//! (or any value ≥ slice_size) to effectively disable tiling and compare.
//!
//! ```bash
//! cargo bench --bench encoder_inner             # tiled (default)
//! PAR2RUST_L2_BLOCK_BYTES=999999999 \
//!   cargo bench --bench encoder_inner -- --save-baseline untiled
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use par2rust::galois_simd::{detect_dispatch, gf_mul_xor_with_tables, CoeffSimdTables};

// Match a realistic high-redundancy PAR2 workload: slice_size 1 MiB,
// 200 recovery blocks → 200 MiB working set, well past any L2/L3 on
// any current consumer CPU. This is the regime where cache-blocking
// matters; smaller workloads that fit in L2 see no win because the
// untiled access pattern already hits cache.
const SLICE_BYTES: usize = 1024 * 1024;
const RECOVERY_COUNT: usize = 200;
const BLOCK_BYTES: usize = 64 * 1024; // matches creator.rs default

fn bench_encoder(c: &mut Criterion) {
    let dispatch = detect_dispatch();

    let input = vec![0xA5u8; SLICE_BYTES];
    let mut recovery: Vec<Vec<u8>> = (0..RECOVERY_COUNT)
        .map(|_| vec![0u8; SLICE_BYTES])
        .collect();
    let tables: Vec<CoeffSimdTables> = (0..RECOVERY_COUNT)
        .map(|i| CoeffSimdTables::new(dispatch, 0x100 + i as u16))
        .collect();

    let total_bytes = (SLICE_BYTES * RECOVERY_COUNT) as u64;

    let block = if let Ok(s) = std::env::var("PAR2RUST_L2_BLOCK_BYTES") {
        s.parse::<usize>().unwrap_or(BLOCK_BYTES)
    } else {
        BLOCK_BYTES
    };

    let mut g = c.benchmark_group(format!("encoder_inner/{dispatch:?}"));
    g.throughput(Throughput::Bytes(total_bytes));

    // Untiled reference: each recovery buffer touched ONCE for the whole
    // input slice in turn. The input slice gets re-read from L2/DRAM
    // RECOVERY_COUNT times.
    g.bench_function("untiled", |b| {
        b.iter(|| {
            for (t, out) in tables.iter().zip(recovery.iter_mut()) {
                gf_mul_xor_with_tables(t, black_box(&input), out);
            }
        });
    });

    // Tiled: for each input block (size `block`), iterate all recovery
    // buffers. Input block stays L1-hot across RECOVERY_COUNT touches.
    g.bench_function("tiled", |b| {
        b.iter(|| {
            let mut off = 0;
            while off < SLICE_BYTES {
                let end = (off + block).min(SLICE_BYTES);
                let in_block = &input[off..end];
                for (t, out) in tables.iter().zip(recovery.iter_mut()) {
                    gf_mul_xor_with_tables(t, black_box(in_block), &mut out[off..end]);
                }
                off = end;
            }
        });
    });

    g.finish();
}

criterion_group!(benches, bench_encoder);
criterion_main!(benches);
