#!/usr/bin/env bash
# Build the wasm blobs the yamc-core wheel ships.
#
# These used to be committed binaries. They are not any more, because
# `python-source = "python"` copies that tree verbatim into the wheel, so a
# hand-built blob put its builder's home directory into every published wheel:
# 92 distinct `/home/jon/.cargo/...` and `/home/jon/.rustup/...` paths, found by
# `scripts/check_binary_paths.py`. CI builds them now (the `wasm-blobs` job in
# `.github/workflows/ci-python.yml`), and this script is the same build for
# anyone who needs them locally, for `maturin develop` or `Model.to_html()`.
#
# The remaps in `.cargo/config.toml` name the CI runner homes and cannot name
# yours, because cargo does not expand environment variables in `rustflags`. So
# this passes a `$HOME` remap on the command line, where the shell expands it.
# It has to re-supply the `getrandom_backend` cfg too: a `RUSTFLAGS` environment
# variable REPLACES the target-specific array rather than adding to it.
set -euo pipefail

cd "$(dirname "$0")/.."
dest="packages/yamc-core/python/yamc/_wasm"

if ! command -v wasm-pack >/dev/null 2>&1; then
    echo "wasm-pack is not installed. Install it with:" >&2
    echo "  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh" >&2
    exit 1
fi

# Same shape as the CI job, plus the remap of this machine's home. Not appended
# to the config array: this replaces it, which is why the cfg is repeated.
export RUSTFLAGS="--cfg getrandom_backend=\"wasm_js\" --remap-path-prefix=${HOME}=/build"

echo "building yamc_sim (the transport blob Model.to_html embeds)"
# `--out-name` is a wasm-pack flag and `--features` is not: anything wasm-pack
# does not recognise goes to `cargo build`, so reversing these hands cargo an
# `--out-name` it does not have and the build fails.
(cd crates/yamc && wasm-pack build --target web --out-name yamc_sim -- --features wasm)

echo "building yamt (the mesh viewer blob yamc-plot embeds under its mesh feature)"
# Keeps yamt's default features. `wasm` is not one of them, and without it the
# crate has no #[wasm_bindgen] exports and wasm-pack emits a 366-byte module
# that compiles in and does nothing. `--no-default-features` is also wrong: it
# would drop `arrow`, the only mesh input format yamt supports.
(cd crates/yamt && wasm-pack build --target web -- --features wasm)

mkdir -p "$dest"
# Named files only. Copying the whole `pkg/` directory would also bring
# wasm-pack's own package.json, its .d.ts pair and, worse, its .gitignore.
# A .gitignore in this directory silently drops the whole thing out of every
# wheel, because maturin honours gitignore when it copies `python-source`.
cp crates/yamc/pkg/yamc_sim.js      "$dest/yamc_sim.js"
cp crates/yamc/pkg/yamc_sim_bg.wasm "$dest/yamc_sim_bg.wasm"
cp crates/yamt/pkg/yamt.js          "$dest/yamt.js"
cp crates/yamt/pkg/yamt_bg.wasm     "$dest/yamt_bg.wasm"

echo
echo "wrote:"
ls -l "$dest"

# Guarded because the checker arrives with a separate change. Once it is here
# this runs automatically; until then, fall back to grepping for this machine's
# home, which is the leak the remap above exists to prevent.
echo
if [ -f scripts/check_binary_paths.py ]; then
    echo "checking them for builder paths"
    python3 scripts/check_binary_paths.py "$dest"/*.wasm
else
    echo "checking them for this machine's home directory"
    if grep -l "$HOME" "$dest"/*.wasm 2>/dev/null; then
        echo "the files above still carry $HOME, so the remap did not apply" >&2
        exit 1
    fi
    echo "ok: no $HOME in any blob"
fi
