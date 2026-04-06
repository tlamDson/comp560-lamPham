#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROFILE_PACKAGES = {
    "core": ["torch", "numpy", "tqdm"],
    "bot": ["torch", "numpy", "tqdm", "optuna", "pynvml"],
    "phi": [
        "torch",
        "numpy",
        "tqdm",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
    ],
    "phi-train": [
        "torch",
        "numpy",
        "tqdm",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl",
        "deepspeed",
    ],
}


def resolve_path(path_value: str, repo_root: Path) -> Path:
    raw = Path(path_value).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (repo_root / raw).resolve()


def print_result(ok: bool, label: str, detail: str) -> None:
    tag = "OK" if ok else "FAIL"
    print(f"[{tag}] {label}: {detail}")


def check_python_version(min_major: int = 3, min_minor: int = 10) -> tuple[bool, str]:
    cur = sys.version_info
    ok = (cur.major, cur.minor) >= (min_major, min_minor)
    return ok, f"{cur.major}.{cur.minor}.{cur.micro}"


def check_expected_interpreter(expected: str | None, repo_root: Path) -> tuple[bool, str]:
    current = Path(sys.executable).resolve()
    if not expected:
        return True, f"current={current} (no expected path provided)"

    expected_path = resolve_path(expected, repo_root)
    ok = current == expected_path
    detail = f"current={current}, expected={expected_path}"
    return ok, detail


def check_pip() -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pip", "--version"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or "pip command failed"
        return False, err
    return True, proc.stdout.strip()


def check_imports(packages: list[str]) -> tuple[bool, str]:
    failed: list[str] = []
    for pkg in packages:
        cmd = [
            sys.executable,
            "-c",
            f"import importlib; importlib.import_module({pkg!r})",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip().splitlines()
            detail = err[-1] if err else "import failed"
            failed.append(f"{pkg} ({detail})")

    if failed:
        return False, "missing_or_broken=" + "; ".join(failed)
    return True, "all required packages imported"


def check_cuda(require_cuda: bool) -> tuple[bool, str]:
    if not require_cuda:
        return True, "not required"

    try:
        import torch
    except Exception as exc:
        return False, f"torch import failed: {exc}"

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"

    try:
        test = torch.tensor([1.0], device="cuda")
        _ = test * test
        torch.cuda.synchronize()
        return True, f"cuda_available=True, device_count={torch.cuda.device_count()}"
    except Exception as exc:
        return False, f"cuda probe failed: {exc}"


def check_phi_model_path(phi_model_path: str, repo_root: Path) -> tuple[bool, str]:
    model_dir = resolve_path(phi_model_path, repo_root)
    if not model_dir.exists() or not model_dir.is_dir():
        return False, f"model directory not found: {model_dir}"

    required = ["config.json", "tokenizer_config.json"]
    missing_required = [name for name in required if not (model_dir / name).exists()]
    if missing_required:
        return False, f"missing files in {model_dir}: {', '.join(missing_required)}"

    has_weights = any(model_dir.glob("*.safetensors")) or (model_dir / "model.safetensors.index.json").exists()
    if not has_weights:
        return False, f"no model weights detected in {model_dir}"

    return True, f"path={model_dir}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Environment gatekeeper for training/tuning scripts")
    parser.add_argument("--profile", choices=sorted(PROFILE_PACKAGES.keys()), default="core")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--expected-python", default=os.environ.get("ML_PYTHON_PATH", ""))
    parser.add_argument("--phi-model-path", default=os.environ.get("PHI_MODEL_PATH", "Phi-3-mini-4k-instruct"))
    parser.add_argument("--repo-root", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.repo_root:
        repo_root = Path(args.repo_root).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parent

    checks: list[tuple[str, tuple[bool, str]]] = []
    checks.append(("python_version", check_python_version()))
    checks.append(("python_executable", check_expected_interpreter(args.expected_python or None, repo_root)))
    checks.append(("pip", check_pip()))
    checks.append(("imports", check_imports(PROFILE_PACKAGES[args.profile])))
    checks.append(("cuda", check_cuda(args.require_cuda)))

    if args.profile.startswith("phi"):
        checks.append(("phi_model_path", check_phi_model_path(args.phi_model_path, repo_root)))

    failed = 0
    print("Environment validation summary")
    print(f"- profile: {args.profile}")
    print(f"- interpreter: {Path(sys.executable).resolve()}")
    print(f"- repo_root: {repo_root}")

    for label, (ok, detail) in checks:
        print_result(ok, label, detail)
        if not ok:
            failed += 1

    if failed:
        print(f"Validation failed with {failed} issue(s).")
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())