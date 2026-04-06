#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ML_ENV_FILE:-${SCRIPT_DIR}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -n "${ML_CONDA_ENV:-}" && -z "${ML_PYTHON_PATH:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
      conda activate "${ML_CONDA_ENV}"
    fi
  fi
fi

if [[ -z "${ML_PYTHON_PATH:-}" ]]; then
  if [[ -f "${SCRIPT_DIR}/.vscode/settings.json" ]]; then
    SETTINGS_PYTHON="$(grep -E '"python.defaultInterpreterPath"' "${SCRIPT_DIR}/.vscode/settings.json" | head -n 1 | sed -E 's/.*"python.defaultInterpreterPath"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')"
    if [[ -n "${SETTINGS_PYTHON}" ]]; then
      SETTINGS_PYTHON="${SETTINGS_PYTHON//\$\{workspaceFolder\}/${SCRIPT_DIR}}"
      if [[ -x "${SETTINGS_PYTHON}" ]]; then
        ML_PYTHON_PATH="${SETTINGS_PYTHON}"
      fi
    fi
  fi

fi

if [[ -z "${ML_PYTHON_PATH:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    ML_PYTHON_PATH="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    ML_PYTHON_PATH="${SCRIPT_DIR}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    ML_PYTHON_PATH="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    ML_PYTHON_PATH="$(command -v python)"
  else
    echo "ERROR: No Python interpreter found. Set ML_PYTHON_PATH in .env." >&2
    exit 1
  fi
fi

if [[ ! -x "${ML_PYTHON_PATH}" ]]; then
  echo "ERROR: ML_PYTHON_PATH is not executable: ${ML_PYTHON_PATH}" >&2
  exit 1
fi

export ML_PYTHON_PATH

if [[ "${1:-}" == "--print-python" ]]; then
  echo "${ML_PYTHON_PATH}"
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  cat <<'EOF'
Usage: ./run.sh [python args]

Examples:
  ./run.sh validate_env.py --profile core --require-cuda
  ./run.sh -u comp560-lamPham/common/train.py comp560-lamPham/4_digits_addition/config/basic.py
EOF
  exit 1
fi

exec "${ML_PYTHON_PATH}" "$@"