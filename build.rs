// par2rust build script.
//
// The only purpose right now is to compile the vendored ISA-L crypto
// `md5_mb` sources when the `mb-md5` cargo feature is enabled. Without
// the feature, the build is unchanged.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let mb_md5 = std::env::var_os("CARGO_FEATURE_MB_MD5").is_some();
    if !mb_md5 {
        return;
    }

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_arch != "aarch64" {
        // Phase 1 ships aarch64 only. On other architectures the
        // feature is a no-op for now; document and continue.
        println!(
            "cargo:warning=mb-md5 feature has no effect on target_arch={target_arch} (Phase 1 is aarch64-only)"
        );
        return;
    }
    if target_os == "macos" || target_os == "ios" {
        // ISA-L's aarch64 .S files use GNU-as ELF relocation syntax
        // (`adrp Xn, .label` + `:lo12:.label`) that Apple's Mach-O
        // assembler doesn't accept. See vendor/isa-l_crypto/README.md
        // for the full diagnosis; rewriting requires per-instruction
        // translation to Mach-O `@PAGE`/`@PAGEOFF` form. Until that
        // port lands, fall through to a no-op build on Apple targets.
        println!(
            "cargo:warning=mb-md5 not supported on target_os={target_os} yet (ELF relocation syntax in vendored .S files); using scalar md-5 instead. Track at https://github.com/javi11/par2rust"
        );
        return;
    }

    let vendor = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("vendor")
        .join("isa-l_crypto");

    let mut build = cc::Build::new();
    // ISA-L's `types.h` only emits the GCC `__attribute__((aligned(N)))`
    // form of DECLARE_ALIGNED when `__unix__` is defined. macOS / Darwin
    // doesn't predefine `__unix__`, so define it ourselves — clang on
    // Darwin understands GCC alignment attributes regardless.
    build.define("__unix__", None);
    build
        .include(vendor.join("include").join("isa-l_crypto"))
        .include(vendor.join("include").join("internal"))
        .include(vendor.join("md5_mb"))
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-variable")
        .flag_if_supported("-Wno-deprecated-declarations")
        .file(vendor.join("md5_mb").join("md5_ctx_base.c"))
        .file(vendor.join("md5_mb").join("md5_mb.c"))
        .file(vendor.join("md5_mb").join("md5_ref.c"))
        .file(vendor.join("md5_mb").join("md5_ctx_base_aliases.c"))
        .file(
            vendor
                .join("md5_mb")
                .join("aarch64")
                .join("md5_ctx_aarch64_asimd.c"),
        )
        .file(
            vendor
                .join("md5_mb")
                .join("aarch64")
                .join("md5_mb_mgr_aarch64_asimd.c"),
        )
        .file(
            vendor
                .join("md5_mb")
                .join("aarch64")
                .join("md5_mb_asimd_x1.S"),
        )
        .file(
            vendor
                .join("md5_mb")
                .join("aarch64")
                .join("md5_mb_asimd_x4.S"),
        );

    println!("cargo:rerun-if-changed={}", vendor.join("md5_mb").display());
    println!(
        "cargo:rerun-if-changed={}",
        vendor.join("include").display()
    );

    build.compile("isal_md5_mb");
}
