#!/usr/bin/env bash
set -euo pipefail

# Build the package-local MMCV CUDA/C++ extensions for the active Python and
# PyTorch environment. This script deliberately uses only sources shipped in
# this package; CARLA is not needed for the build step.
CODES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DT_ROOT="${CODES_ROOT}/DriveTransformer"
PYTHON="${PYTHON:-python3}"

if [[ "${PYTHON}" != /* ]]; then
    if [[ "${PYTHON}" == */* ]]; then
        PYTHON="$(cd "$(dirname "${PYTHON}")" && pwd)/$(basename "${PYTHON}")"
    else
        PYTHON="$(command -v "${PYTHON}" || true)"
    fi
fi

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "PYTHON must name an executable Python interpreter" >&2
    exit 2
fi

test -f "${DT_ROOT}/setup.py"
cd "${DT_ROOT}"
"${PYTHON}" setup.py build_ext --inplace

for extension in \
    "mmcv/_ext" \
    "mmcv/ops/iou3d_det/iou3d_cuda" \
    "mmcv/ops/roiaware_pool3d/roiaware_pool3d_ext"; do
    if ! compgen -G "${DT_ROOT}/${extension}.cpython-*.so" >/dev/null; then
        echo "missing built extension: ${DT_ROOT}/${extension}.cpython-*.so" >&2
        exit 1
    fi
done

echo "package-local MMCV extensions are ready under ${DT_ROOT}/mmcv"
