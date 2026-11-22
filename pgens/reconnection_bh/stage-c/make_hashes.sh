#!/usr/bin/env bash
# The hashes the driver and the analysis verify on Rusty, with the paths of the stage-C
# directory there (inputs/<name>.toml, scripts/<name>.py). Run in this directory.
set -euo pipefail
cd "$(dirname "$0")"
sum() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
: > INPUTS.sha256
for n in r-mid r-low-b r-low-c r-off-b r-off-c; do echo "$(sum "$n.toml")  inputs/$n.toml" >> INPUTS.sha256; done
: > SCRIPTS.sha256
for n in check3_ensemble check6_series check6_health; do echo "$(sum "../scripts/$n.py")  scripts/$n.py" >> SCRIPTS.sha256; done
cat INPUTS.sha256 SCRIPTS.sha256
