#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Existing SDK/torch installations are never upgraded by this installer.
"${PYTHON:-python3}" -m pip install --no-deps --no-build-isolation -e .
fxbx --help
