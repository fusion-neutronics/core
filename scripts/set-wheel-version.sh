#!/usr/bin/env bash
# Write the version a wheel will publish at into its package's pyproject.toml.
#
#   scripts/set-wheel-version.sh yamc-core 0.8.1
#
# The release workflow runs this in every wheel job, with the version the
# release tag (or a dispatch override) names, so the committed placeholder in
# `project.version` never reaches PyPI. Nothing is committed back.
#
# Only pyproject.toml changes (and git is told to ignore that, see below). The crate manifests and Cargo.lock are left
# alone: maturin takes a static `project.version` over Cargo.toml's, and
# leaving the lock untouched keeps `cargo about --locked` and friends happy.
set -euo pipefail

package="${1:?usage: set-wheel-version.sh <yamc-core|yani-core> <version>}"
version="${2:?usage: set-wheel-version.sh <yamc-core|yani-core> <version>}"
pyproject="packages/$package/pyproject.toml"

[ -f "$pyproject" ] || { echo "no such package: $pyproject" >&2; exit 1; }

# `project.version` is the only line in these files that starts with
# `version =`: everything else that mentions a version is nested in a value.
count="$(grep -c '^version = ' "$pyproject" || true)"
if [ "$count" != 1 ]; then
  echo "expected exactly one '^version = ' line in $pyproject, found $count" >&2
  exit 1
fi

sed "s/^version = \".*\"/version = \"$version\"/" "$pyproject" > "$pyproject.tmp"
mv "$pyproject.tmp" "$pyproject"

# The wheel records whether tracked files differed from the commit it was built
# from (`__commit_dirty__`). This edit is the release's own, not a change to
# the source, so tell git to leave the file out of that check.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git update-index --skip-worktree "$pyproject"
fi

grep '^version = ' "$pyproject"
