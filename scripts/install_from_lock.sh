#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${REPO_ROOT}/run.sh"
ENV_NAME="${1:-comp560-bots}"

if [[ ! -f "${REPO_ROOT}/conda-lock.yml" ]]; then
  echo "ERROR: conda-lock.yml not found. Run scripts/lock_deps.sh first." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/requirements-lock.txt" ]]; then
  echo "ERROR: requirements-lock.txt not found. Run scripts/lock_deps.sh first." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found in PATH." >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo ">>> Installing conda base from lock into env: ${ENV_NAME}"
"${RUNNER}" -m conda_lock install --name "${ENV_NAME}" conda-lock.yml

echo ">>> Installing pip dependencies from requirements-lock.txt"
conda run -n "${ENV_NAME}" python -m pip install -r requirements-lock.txt

echo ">>> Environment ready: ${ENV_NAME}"