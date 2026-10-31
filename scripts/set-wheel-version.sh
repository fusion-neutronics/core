#!/usr/bin/env bash
# Set the version a wheel will publish at, in the one place it lives: the crate
# manifest that its pyproject.toml names as `manifest-path` and reads through
# `dynamic = ["version"]`.
#
#   scripts/set-wheel-version.sh yamc-python 0.8.1
#
# Locally, prefer `cargo set-version -p yamc-python 0.8.1` (cargo-edit), which
# does the same thing. This script exists so the release workflow's
# dispatch-override path does not have to install cargo-edit in every wheel job,
# on three platforms.
#
# It patches Cargo.lock as well: the lock records each workspace member's
# version, and the license-bundle step runs `cargo about --locked`, which
# refuses a stale lock.
set -euo pipefail

crate="${1:?usage: set-wheel-version.sh <crate> <version>}"
version="${2:?usage: set-wheel-version.sh <crate> <version>}"
manifest="crates/$crate/Cargo.toml"

# `^version` matches only the [package] version: dependency versions in a Cargo
# manifest are nested inside their own tables, never at the start of a line.
sed "s/^version = \".*\"/version = \"$version\"/" "$manifest" > "$manifest.tmp"
mv "$manifest.tmp" "$manifest"

# In Cargo.lock the version line directly follows its package's name line.
sed "/^name = \"$crate\"\$/{n;s/^version = \".*\"/version = \"$version\"/;}" \
    Cargo.lock > Cargo.lock.tmp
mv Cargo.lock.tmp Cargo.lock

grep -m1 '^version' "$manifest"
