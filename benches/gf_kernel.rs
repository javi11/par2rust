//! GF(2^16) multiply-XOR kernel throughput benchmark.
//!
//! Measures wall-clock throughput of the dispatch-selected kernel
//! (`Dispatch::Gfni` on GFNI-capable x86, `Dispatch::Ssse3` on older
//! x86, `Dispatch::Neon` on aarch64, `Dispatch::TableScalar` elsewhere)
//! on a 1 MiB input/output pair. The input pair is L2-resident on every
//! modern CPU, so the measurement reflects per-iter SIMD throughput
//! without DRAM bandwidth confounding the result.
//!
//! ```text
//! cargo bench --bench gf_kernel
//! ```
//!
//! For comparing kernels across hardware, run the same workload on each
//! box and record the throughput line; the dispatch name in the group
//! label ("Gfni" / "Ssse3" / "Neon") makes the kernel obvious.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use par2rust::galois_simd::{detect_dispatch, gf_mul_xor_with_tables, CoeffSimdTables};

fn bench_kernel(c: &mut Criterion) {
    let dispatch = detect_dispatch();
    let coeff: u16 = 0xABCD;
    let input = vec![0xA5u8; 1024 * 1024];
    let mut output = vec![0u8; input.len()];
    let tables = CoeffSimdTables::new(dispatch, coeff);

    let mut g = c.benchmark_group(format!("gf_mul_xor/{dispatch:?}"));
    g.throughput(Throughput::Bytes(input.len() as u64));
    g.bench_function("1MiB", |b| {
        b.iter(|| {
            gf_mul_xor_with_tables(&tables, black_box(&input), black_box(&mut output));
        });
    });
    g.finish();
}

criterion_group!(benches, bench_kernel);
criterion_main!(benches);
