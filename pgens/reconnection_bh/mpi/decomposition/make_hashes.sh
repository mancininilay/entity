#!/usr/bin/env bash
# Fixes the hashes of the four inputs and of the analysis scripts (run in this folder).
set -euo pipefail
cd "$(dirname "$0")"
h() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
h r-low-mpi-a41.toml r-low-mpi-b22.toml r-low-mpi-b41.toml r-low-mpi-c22.toml > INPUTS.sha256
h check3_ensemble.py scalars.py interface.py verdict.py > SCRIPTS.sha256
