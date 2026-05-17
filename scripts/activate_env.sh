#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${PROJECT_ROOT}/.cache/matplotlib" "${PROJECT_ROOT}/.cache/fontconfig"

export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"

source "${PROJECT_ROOT}/.venv/bin/activate"
