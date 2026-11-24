//! Generates `packages/yani-core/python/yani/_core.pyi` from the PyO3 bindings'
//! embedded type metadata (registered by the `#[gen_stub_*]` macros). Run with:
//!
//!   cargo run -p yani-python --bin stub_gen --features stub-gen
//!
//! Must be built WITHOUT the `extension-module` feature so it can link
//! libpython (the `stub-gen` feature gate keeps it out of normal builds).

fn main() -> pyo3_stub_gen::Result<()> {
    let stub = yani_python::gen_stub_info()?;
    stub.generate()?;
    Ok(())
}
