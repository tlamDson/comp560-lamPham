#!/usr/bin/env python3
"""
Hyperparameter Tuning Bot for 4-digit addition.

Goals:
- Avoid OOM with a VRAM estimator + batch-size binary search.
- Prune slow or stagnant runs early.
- Keep reproducibility metadata for every trial.
- Export trial leaderboard + champion report.

Usage example:
    python hyperopt_bot.py --n-trials 5 --max-iters 1200 --eval-interval 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import signal
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import optuna
    from optuna.exceptions import ExperimentalWarning, TrialPruned
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: optuna. Install with: pip install -r requirements_tuning.txt"
    ) from exc

try:
    import pynvml
except Exception:
    pynvml = None

try:
    from tqdm import tqdm
except Exception:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def update(self, n: int = 1) -> None:
            _ = n

        def set_description(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set_postfix(self, *args: Any, **kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass

PROJECT_DIR = Path(__file__).resolve().parent
COMMON_TRAIN = PROJECT_DIR.parent / "common" / "train.py"
DEFAULT_BASE_CONFIG = PROJECT_DIR / "config" / "basic.py"

ITER_RE = re.compile(
    r"iter\s+(\d+):\s+loss\s+([0-9.eE+-]+),\s+time\s+([0-9.]+)ms,\s+mfu\s+([-0-9.]+)%"
)
EVAL_RE = re.compile(
    r"step\s+(\d+):\s+train loss\s+([0-9.eE+-]+),\s+val loss\s+([0-9.eE+-]+),\s+val acc\s+([0-9.]+)%"
)
OOM_PATTERNS = (
    "out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)

TPE_STARTUP_TRIALS = 20
DEFAULT_MAX_TRIAL_SECONDS = 25.0
WIDE_BATCH_UTIL_MIN = 0.05
WIDE_BATCH_UTIL_MAX = 0.95
SPRINT_N_EMBD_MIN = 64
SPRINT_N_EMBD_MAX = 256
SPRINT_LR_MIN = 2e-3
SPRINT_LR_MAX = 2e-2
SPRINT_DEFAULT_MAX_ITERS = 3000


@dataclass
class TrainOutcome:
    status: str
    reason: str
    elapsed_sec: float
    time_to_target_sec: Optional[float]
    best_val_acc: float
    best_val_loss: float
    peak_vram_gb: float
    avg_iter_ms: Optional[float]
    time_weighted_convergence: Optional[float]
    throttled: bool


class GPUProbe:
    def __init__(self) -> None:
        self.backend = "none"
        self.handle = None
        self.nvml_initialized = False

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.backend = "nvml"
            except Exception:
                self.backend = "none"

    def close(self) -> None:
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml_initialized = False

    def sample(self) -> Optional[Dict[str, Any]]:
        if self.backend == "nvml" and self.handle is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                power_w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                temp_c = float(
                    pynvml.nvmlDeviceGetTemperature(
                        self.handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
                pstate = int(pynvml.nvmlDeviceGetPerformanceState(self.handle))

                return {
                    "memory_total_gb": mem.total / (1024.0**3),
                    "memory_used_gb": mem.used / (1024.0**3),
                    "power_w": power_w,
                    "temperature_c": temp_c,
                    "gpu_util": float(util.gpu),
                    "pstate": pstate,
                }
            except Exception:
                return self._sample_nvidia_smi()

        return self._sample_nvidia_smi()

    @staticmethod
    def _sample_nvidia_smi() -> Optional[Dict[str, Any]]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,power.draw,temperature.gpu,pstate,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

        if not out:
            return None

        first = out.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) < 6:
            return None

        try:
            mem_total_gb = float(parts[0]) / 1024.0
            mem_used_gb = float(parts[1]) / 1024.0
            power_w = float(parts[2])
            temp_c = float(parts[3])
            pstate_raw = parts[4]
            pstate = None
            if pstate_raw.upper().startswith("P"):
                pstate = int(pstate_raw[1:])
            gpu_util = float(parts[5])
        except Exception:
            return None

        return {
            "memory_total_gb": mem_total_gb,
            "memory_used_gb": mem_used_gb,
            "power_w": power_w,
            "temperature_c": temp_c,
            "gpu_util": gpu_util,
            "pstate": pstate,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_name(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_")
    return cleaned or "study"


def safe_unlink(path: Path) -> bool:
    try:
        if path.exists() and path.is_file():
            path.unlink()
            return True
    except Exception:
        return False
    return False


def safe_rmtree(path: Path) -> bool:
    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            return True
    except Exception:
        return False
    return False


def should_cleanup_status(status: str, cleanup_finished_no_target: bool) -> bool:
    if status in {"pruned", "oom", "throttled", "failed"}:
        return True
    if status == "finished_no_target":
        return cleanup_finished_no_target
    return False


def load_simple_config_vars(config_path: Path) -> Dict[str, Any]:
    scope: Dict[str, Any] = {}
    exec(config_path.read_text(), scope)
    out: Dict[str, Any] = {}
    for k, v in scope.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, bool, str)):
            out[k] = v
    return out


def append_overrides_to_config(base_path: Path, out_path: Path, overrides: Dict[str, Any]) -> None:
    base_text = base_path.read_text().rstrip()
    lines = [base_text, "", "# ---- Auto-generated overrides (Hyperparameter Tuning Bot) ----"]
    for key in sorted(overrides.keys()):
        lines.append(f"{key} = {repr(overrides[key])}")
    out_path.write_text("\n".join(lines) + "\n")


def detect_vocab_size(dataset_name: str) -> int:
    meta_path = PROJECT_DIR / "data" / dataset_name / "meta.pkl"
    if not meta_path.exists():
        return 16

    try:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = int(meta.get("vocab_size", 16))
        return max(8, vocab_size)
    except Exception:
        return 16


def flash_attn2_available() -> bool:
    try:
        import torch
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False

    return True


def detect_ac_power() -> Optional[bool]:
    base = Path("/sys/class/power_supply")
    if not base.exists():
        return None

    online_paths: List[Path] = []

    for item in base.iterdir():
        online = item / "online"
        type_file = item / "type"
        if not online.exists():
            continue

        name_low = item.name.lower()
        if any(tag in name_low for tag in ("ac", "adp", "mains")):
            online_paths.append(online)
            continue

        if type_file.exists():
            try:
                ps_type = type_file.read_text().strip().lower()
            except Exception:
                ps_type = ""
            if ps_type == "mains":
                online_paths.append(online)

    if not online_paths:
        return None

    values: List[int] = []
    for p in online_paths:
        try:
            values.append(int(p.read_text().strip()))
        except Exception:
            pass

    if not values:
        return None

    return any(v == 1 for v in values)


def detect_other_gpu_processes(current_pid: int) -> List[Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []

    rows: List[Dict[str, Any]] = []
    for raw in out.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3:
            continue

        try:
            pid = int(parts[0])
            used_mb = int(parts[2])
        except Exception:
            continue

        if pid == current_pid:
            continue

        rows.append({"pid": pid, "name": parts[1], "used_mb": used_mb})

    return rows


def current_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_DIR.parent,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def gpu_prewarm(seconds: float, matrix_dim: int) -> None:
    if seconds <= 0.0:
        return

    dim = max(512, int(matrix_dim))
    code = (
        "import time\n"
        "import torch\n"
        "if torch.cuda.is_available():\n"
        f"    n={dim}\n"
        "    a=torch.randn((n,n),device='cuda',dtype=torch.float16)\n"
        "    b=torch.randn((n,n),device='cuda',dtype=torch.float16)\n"
        "    torch.cuda.synchronize()\n"
        f"    t_end=time.time()+{seconds:.3f}\n"
        "    while time.time() < t_end:\n"
        "        _=a@b\n"
        "    torch.cuda.synchronize()\n"
    )

    timeout_sec = max(10, int(seconds) + 15)
    try:
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout_sec,
        )
    except Exception:
        return


def sample_gpu_stable(gpu_probe: GPUProbe, bursts: int = 3, gap_sec: float = 0.15) -> Optional[Dict[str, Any]]:
    snaps: List[Dict[str, Any]] = []
    for i in range(max(1, bursts)):
        s = gpu_probe.sample()
        if s is not None:
            snaps.append(s)
        if i + 1 < max(1, bursts):
            time.sleep(max(0.0, gap_sec))

    if not snaps:
        return None

    out = dict(snaps[-1])
    out["memory_used_gb"] = max(parse_float(s.get("memory_used_gb"), 0.0) for s in snaps)
    out["gpu_util"] = max(parse_float(s.get("gpu_util"), 0.0) for s in snaps)

    powers = [parse_float(s.get("power_w"), -1.0) for s in snaps if s.get("power_w") is not None]
    out["power_w"] = max(powers) if powers else None

    pstates = [parse_int(s.get("pstate"), 99) for s in snaps if s.get("pstate") is not None]
    out["pstate"] = min(pstates) if pstates else None
    return out


def start_keepalive_process(dim: int, sleep_sec: float) -> Optional[subprocess.Popen[str]]:
    matrix_dim = max(512, int(dim))
    sleep_time = max(0.0, float(sleep_sec))
    code = (
        "import time\n"
        "import torch\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit(0)\n"
        f"n={matrix_dim}\n"
        f"pause={sleep_time:.6f}\n"
        "a=torch.randn((n,n),device='cuda',dtype=torch.float16)\n"
        "b=torch.randn((n,n),device='cuda',dtype=torch.float16)\n"
        "torch.cuda.synchronize()\n"
        "while True:\n"
        "    _=a@b\n"
        "    torch.cuda.synchronize()\n"
        "    if pause > 0:\n"
        "        time.sleep(pause)\n"
    )

    preexec_fn = os.setsid if os.name == "posix" else None
    try:
        return subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            preexec_fn=preexec_fn,
        )
    except Exception:
        return None


def default_sprint_seed_params() -> List[Dict[str, Any]]:
    # Focused seeds around historically fast arithmetic regimes without loosening constraints.
    return [
        {
            "n_layer": 5,
            "n_embd": 96,
            "learning_rate": 3.5e-3,
            "beta2": 0.975,
            "warmup_iters": 50,
            "grad_clip": 0.78,
            "dropout": 0.073,
            "weight_decay": 0.0072,
            "batch_util": 0.95,
        },
        {
            "n_layer": 5,
            "n_embd": 128,
            "learning_rate": 2.9e-3,
            "beta2": 0.97,
            "warmup_iters": 75,
            "grad_clip": 0.85,
            "dropout": 0.05,
            "weight_decay": 0.005,
            "batch_util": 0.95,
        },
        {
            "n_layer": 4,
            "n_embd": 96,
            "learning_rate": 2.2e-3,
            "beta2": 0.985,
            "warmup_iters": 75,
            "grad_clip": 0.8,
            "dropout": 0.04,
            "weight_decay": 0.004,
            "batch_util": 0.95,
        },
        {
            "n_layer": 3,
            "n_embd": 192,
            "learning_rate": 2.2e-3,
            "beta2": 0.98,
            "warmup_iters": 75,
            "grad_clip": 1.0,
            "dropout": 0.0,
            "weight_decay": 0.001,
            "batch_util": 0.95,
        },
        {
            "n_layer": 3,
            "n_embd": 224,
            "learning_rate": 3.5e-3,
            "beta2": 0.975,
            "warmup_iters": 100,
            "grad_clip": 1.0,
            "dropout": 0.0,
            "weight_decay": 0.0005,
            "batch_util": 0.95,
        },
        {
            "n_layer": 3,
            "n_embd": 256,
            "learning_rate": 2.0e-3,
            "beta2": 0.99,
            "warmup_iters": 125,
            "grad_clip": 0.9,
            "dropout": 0.0,
            "weight_decay": 0.0003,
            "batch_util": 0.27,
        },
        {
            "n_layer": 4,
            "n_embd": 192,
            "learning_rate": 2.0e-3,
            "beta2": 0.985,
            "warmup_iters": 100,
            "grad_clip": 1.1,
            "dropout": 0.0,
            "weight_decay": 0.0015,
            "batch_util": 0.27,
        },
        {
            "n_layer": 4,
            "n_embd": 256,
            "learning_rate": 2.0e-3,
            "beta2": 0.99,
            "warmup_iters": 150,
            "grad_clip": 1.0,
            "dropout": 0.0,
            "weight_decay": 0.002,
            "batch_util": 0.27,
        },
    ]


def normalize_seed_params(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        n_layer = parse_int(raw.get("n_layer"), -1)
        n_embd = parse_int(raw.get("n_embd"), -1)
        learning_rate = parse_float(raw.get("learning_rate"), -1.0)
        beta2 = parse_float(raw.get("beta2"), -1.0)
        warmup_iters = parse_int(raw.get("warmup_iters"), -1)
        grad_clip = parse_float(raw.get("grad_clip"), -1.0)
        dropout = parse_float(raw.get("dropout"), -1.0)
        weight_decay = parse_float(raw.get("weight_decay"), -1.0)
        batch_util = parse_float(raw.get("batch_util"), -1.0)

        if not (2 <= n_layer <= 8):
            return None
        if n_embd < SPRINT_N_EMBD_MIN or n_embd > SPRINT_N_EMBD_MAX:
            return None
        if n_embd % 32 != 0:
            n_embd = max(
                SPRINT_N_EMBD_MIN,
                min(SPRINT_N_EMBD_MAX, int(round(n_embd / 32.0) * 32)),
            )
        if not (SPRINT_LR_MIN <= learning_rate <= SPRINT_LR_MAX):
            return None
        if not (0.95 <= beta2 <= 0.999):
            return None
        if not (50 <= warmup_iters <= 200):
            return None
        if warmup_iters % 25 != 0:
            warmup_iters = max(50, min(200, int(round(warmup_iters / 25.0) * 25)))
        if not (0.5 <= grad_clip <= 1.5):
            return None
        if not (0.0 <= dropout <= 0.08):
            return None
        if not (0.0 <= weight_decay <= 0.01):
            return None
        if not (WIDE_BATCH_UTIL_MIN <= batch_util <= WIDE_BATCH_UTIL_MAX):
            return None

        return {
            "n_layer": n_layer,
            "n_embd": n_embd,
            "learning_rate": learning_rate,
            "beta2": beta2,
            "warmup_iters": warmup_iters,
            "grad_clip": grad_clip,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "batch_util": batch_util,
        }
    except Exception:
        return None


def discover_elite_seed_params(results_history_dir: Path, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0 or not results_history_dir.exists():
        return []

    records: List[Dict[str, Any]] = []
    for csv_path in sorted(results_history_dir.glob("*/reports/trials.csv")):
        try:
            with csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    acc = parse_float(row.get("best_val_acc"), -1.0)
                    loss = parse_float(row.get("best_val_loss"), 1e9)
                    dur = parse_float(row.get("duration_sec"), 1e9)
                    state = str(row.get("state", "")).upper().strip()
                    reason = str(row.get("reason", "")).lower()

                    if acc < 0.0:
                        continue
                    if dur <= 0.0 or not math.isfinite(dur):
                        continue
                    if "incomplete_due_to_throttling" in reason:
                        continue
                    if state not in {"COMPLETE", "PRUNED"}:
                        continue

                    raw_params = {
                        "n_layer": row.get("param_n_layer"),
                        "n_embd": row.get("param_n_embd"),
                        "learning_rate": row.get("param_learning_rate"),
                        "beta2": row.get("param_beta2"),
                        "warmup_iters": row.get("param_warmup_iters"),
                        "grad_clip": row.get("param_grad_clip"),
                        "dropout": row.get("param_dropout"),
                        "weight_decay": row.get("param_weight_decay"),
                        "batch_util": row.get("param_batch_util"),
                    }
                    seed = normalize_seed_params(raw_params)
                    if seed is None:
                        continue

                    records.append(
                        {
                            "acc": acc,
                            "loss": loss,
                            "dur": dur,
                            "seed": seed,
                        }
                    )
        except Exception:
            continue

    records.sort(key=lambda r: (-r["acc"], r["loss"], r["dur"]))

    seen = set()
    out: List[Dict[str, Any]] = []
    for rec in records:
        key = json.dumps(rec["seed"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec["seed"])
        if len(out) >= limit:
            break

    return out


def enqueue_seed_trials(
    *,
    study: optuna.Study,
    history_dir: Path,
    include_default_sprint: bool,
    elite_limit: int,
    batch_util_min: float,
    batch_util_max: float,
) -> int:
    queued: List[Dict[str, Any]] = []
    if include_default_sprint:
        queued.extend(default_sprint_seed_params())
    queued.extend(discover_elite_seed_params(history_dir, elite_limit))

    seen = set()
    count = 0
    for cand in queued:
        seed = normalize_seed_params(cand)
        if seed is None:
            continue
        if not (float(batch_util_min) <= float(seed["batch_util"]) <= float(batch_util_max)):
            continue
        key = json.dumps(seed, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        try:
            study.enqueue_trial(seed)
            count += 1
        except Exception:
            continue
    return count


def estimate_vram_gb(
    *,
    n_layer: int,
    n_embd: int,
    block_size: int,
    batch_size: int,
    vocab_size: int,
    dtype_bytes: int = 2,
    activation_factor: float = 10.0,
    base_cuda_overhead_gb: float = 0.9,
) -> float:
    # Aggressive RTX 4060-oriented estimate:
    # VRAM ~= base CUDA overhead + model params + activation_per_sample * batch_size.
    attn_params = 4 * n_embd * n_embd
    mlp_params = 8 * n_embd * n_embd
    norm_params = 2 * n_embd
    per_layer_params = attn_params + mlp_params + norm_params

    token_emb = vocab_size * n_embd
    lm_head = vocab_size * n_embd
    pos_emb = block_size * n_embd

    total_params = n_layer * per_layer_params + token_emb + lm_head + pos_emb
    model_params_bytes = total_params * dtype_bytes
    activation_per_sample_bytes = block_size * n_embd * n_layer * dtype_bytes * activation_factor
    activ_bytes = activation_per_sample_bytes * batch_size

    total_bytes = model_params_bytes + activ_bytes
    total_gb = total_bytes / (1024.0**3)
    return total_gb + base_cuda_overhead_gb


def align_batch_size(raw_batch: int, max_batch: int) -> int:
    if raw_batch <= 0:
        return 8

    clipped = min(raw_batch, max_batch)
    if clipped > 4096 or max_batch > 4096:
        # Prefer power-of-two sizes first for large-batch memory alignment, else 128-multiple.
        power2 = [2048, 4096, 8192, 16384, 32768]
        valid_pow2 = [p for p in power2 if p <= max_batch and p <= clipped]
        if valid_pow2:
            return max(valid_pow2)
        return max(128, (clipped // 128) * 128)

    return max(8, (clipped // 8) * 8)


def find_safe_batch_size(
    *,
    n_layer: int,
    n_embd: int,
    block_size: int,
    vocab_size: int,
    min_batch: int,
    max_batch: int,
    max_vram_gb: float,
    safety_margin: float,
    vram_overhead_gb: float,
) -> int:
    step = 128 if max_batch > 4096 else 8
    lo = max(step, ((min_batch + step - 1) // step) * step)
    hi = max(lo, (max_batch // step) * step)
    safe = lo

    while lo <= hi:
        mid = ((lo + hi) // (2 * step)) * step
        mid = max(step, mid)

        est = estimate_vram_gb(
            n_layer=n_layer,
            n_embd=n_embd,
            block_size=block_size,
            batch_size=mid,
            vocab_size=vocab_size,
            base_cuda_overhead_gb=vram_overhead_gb,
        )

        if est <= max_vram_gb * safety_margin:
            safe = mid
            lo = mid + step
        else:
            hi = mid - step

    return align_batch_size(max(step, safe), max_batch=max_batch)


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return

    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
    else:
        try:
            proc.terminate()
        except Exception:
            pass

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)

    if proc.poll() is None:
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        else:
            try:
                proc.kill()
            except Exception:
                pass


def read_new_log_lines(log_path: Path, read_offset: int) -> tuple[List[str], int]:
    if not log_path.exists():
        return [], read_offset

    try:
        with log_path.open("r", errors="replace") as f:
            f.seek(read_offset)
            chunk = f.read()
            next_offset = f.tell()
    except Exception:
        return [], read_offset

    if not chunk:
        return [], next_offset

    return chunk.splitlines(), next_offset


def run_training_trial(
    *,
    trial: optuna.trial.Trial,
    config_path: Path,
    log_path: Path,
    target_acc: float,
    eval_stagnation_window: int,
    slow_iter_ms_ceiling: Optional[float],
    gpu_probe: GPUProbe,
    throttle_power_w: float,
    throttle_samples: int,
    hw_sample_sec: float,
    max_startup_seconds: float,
    max_trial_seconds: float,
    target_duration_sec: float,
    min_eval_loss_improvement_ratio: float,
    trial_seed: int,
    gpu_prewarm_sec: float,
    gpu_prewarm_dim: int,
    inductor_cache_dir: Optional[Path],
    anti_throttle_keepalive: bool,
    keepalive_dim: int,
    keepalive_sleep_sec: float,
) -> TrainOutcome:
    gpu_prewarm(gpu_prewarm_sec, gpu_prewarm_dim)
    keepalive_proc = (
        start_keepalive_process(keepalive_dim, keepalive_sleep_sec)
        if anti_throttle_keepalive
        else None
    )

    command = [sys.executable, "-u", str(COMMON_TRAIN), str(config_path)]

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(trial_seed)
    if inductor_cache_dir is not None:
        env["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache_dir)

    preexec_fn = os.setsid if os.name == "posix" else None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_sink = log_path.open("w")

    proc = subprocess.Popen(
        command,
        cwd=str(PROJECT_DIR),
        stdout=log_sink,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=env,
        preexec_fn=preexec_fn,
    )

    best_val_acc = 0.0
    best_val_loss = math.inf
    iter_times: List[float] = []
    eval_losses: List[float] = []
    eval_accs: List[float] = []
    eval_elapsed: List[float] = []

    throttled = False
    low_power_count = 0
    throttle_armed = False
    peak_vram_gb = 0.0

    reason = ""
    time_to_target: Optional[float] = None
    best_time_weighted_convergence = math.inf

    start = time.perf_counter()
    last_hw_sample_ts = 0.0
    last_log_sniff_ts = 0.0
    active_start_ts: Optional[float] = None
    log_read_offset = 0

    def sniff_trial_log(now_ts: float) -> bool:
        nonlocal best_val_acc
        nonlocal best_val_loss
        nonlocal active_start_ts
        nonlocal throttle_armed
        nonlocal time_to_target
        nonlocal best_time_weighted_convergence
        nonlocal reason
        nonlocal log_read_offset

        lines, log_read_offset = read_new_log_lines(log_path, log_read_offset)
        if not lines:
            return False

        for line in lines:
            lower = line.lower()
            if any(pat in lower for pat in OOM_PATTERNS):
                reason = "oom_runtime"
                return True

            m_iter = ITER_RE.search(line)
            if m_iter:
                if active_start_ts is None:
                    active_start_ts = now_ts
                iter_ms = float(m_iter.group(3))
                iter_times.append(iter_ms)

                if slow_iter_ms_ceiling is not None and len(iter_times) >= 30:
                    rolling_ms = sum(iter_times[-30:]) / 30.0
                    if rolling_ms > slow_iter_ms_ceiling:
                        reason = f"slow_trial_iter_ms_over_{slow_iter_ms_ceiling:.2f}"
                        return True

            m_eval = EVAL_RE.search(line)
            if m_eval:
                if active_start_ts is None:
                    active_start_ts = now_ts
                throttle_armed = True

                eval_step = int(m_eval.group(1))
                val_loss = float(m_eval.group(3))
                val_acc = float(m_eval.group(4)) / 100.0

                eval_losses.append(val_loss)
                eval_accs.append(val_acc)
                eval_elapsed.append(now_ts - start)

                if eval_stagnation_window > 0 and len(eval_losses) > eval_stagnation_window:
                    old_loss = eval_losses[-(eval_stagnation_window + 1)]
                    denom = max(abs(old_loss), 1e-12)
                    improvement_ratio = (old_loss - val_loss) / denom
                    if improvement_ratio < min_eval_loss_improvement_ratio:
                        reason = (
                            "eval_loss_stagnation_"
                            f"{improvement_ratio:.4f}_below_{min_eval_loss_improvement_ratio:.4f}"
                        )
                        return True

                best_val_acc = max(best_val_acc, val_acc)
                best_val_loss = min(best_val_loss, val_loss)

                if val_acc < target_acc and len(eval_accs) >= 3:
                    d_acc = eval_accs[-1] - eval_accs[-3]
                    d_t = eval_elapsed[-1] - eval_elapsed[-3]
                    if d_t > 1e-6 and d_acc > 0.0:
                        time_per_acc = d_t / d_acc
                        projected_total = eval_elapsed[-1] + (target_acc - val_acc) * time_per_acc
                        if projected_total > max_trial_seconds and projected_total > target_duration_sec:
                            reason = f"projected_target_miss_{projected_total:.2f}s"
                            return True

                elapsed_real_sec = max(1e-6, now_ts - start)
                time_factor = 1.0 + (elapsed_real_sec / max(1e-6, target_duration_sec))
                acc_gap = max(0.0, target_acc - val_acc)
                convergence_metric = ((0.7 * max(val_loss, 0.0)) + (0.3 * acc_gap)) * time_factor
                best_time_weighted_convergence = min(best_time_weighted_convergence, convergence_metric)

                # Report a time-aware convergence metric so pruning favors faster loss/acc progress per second.
                trial.report(convergence_metric, eval_step)
                if trial.should_prune():
                    reason = "hyperband_pruner"
                    return True

                if val_acc >= target_acc:
                    time_to_target = now_ts - start
                    reason = "target_reached"
                    return True

        return False

    try:
        while True:
            now = time.perf_counter()
            elapsed_real = now - start

            if active_start_ts is None and (now - start) > max_startup_seconds:
                reason = f"startup_timeout_exceeded_{max_startup_seconds:.1f}s"
                break
            if elapsed_real > max_trial_seconds:
                reason = f"time_limit_exceeded_{max_trial_seconds:.1f}s"
                break
            if (
                target_duration_sec > 0.0
                and elapsed_real > target_duration_sec
                and best_val_acc < target_acc
            ):
                reason = f"target_duration_exceeded_{target_duration_sec:.1f}s"
                break

            if now - last_hw_sample_ts >= hw_sample_sec:
                sample = sample_gpu_stable(gpu_probe, bursts=3, gap_sec=0.15)
                last_hw_sample_ts = now
                if sample is not None:
                    peak_vram_gb = max(peak_vram_gb, float(sample.get("memory_used_gb", 0.0)))
                    power_w = sample.get("power_w")
                    pstate = sample.get("pstate")
                    pstate_num = int(pstate) if isinstance(pstate, int) else None

                    if active_start_ts is not None and throttle_armed:
                        is_low_power = power_w is not None and float(power_w) < throttle_power_w
                        is_bad_pstate = pstate_num is not None and pstate_num > 2

                        if is_low_power or is_bad_pstate:
                            low_power_count += 1
                        else:
                            low_power_count = 0

                        if low_power_count >= throttle_samples:
                            throttled = True
                            reason = "incomplete_due_to_throttling"
                            break

            if now - last_log_sniff_ts >= 0.25:
                if sniff_trial_log(now):
                    break
                last_log_sniff_ts = now

            if proc.poll() is not None:
                break

            time.sleep(0.05)

        for _ in range(5):
            before = log_read_offset
            _ = sniff_trial_log(time.perf_counter())
            if log_read_offset == before:
                break
            time.sleep(0.02)

        if reason and proc.poll() is None:
            terminate_process(proc)

        if proc.poll() is None:
            try:
                proc.wait(timeout=2)
            except Exception:
                terminate_process(proc)
    finally:
        try:
            log_sink.close()
        except Exception:
            pass

        if keepalive_proc is not None and keepalive_proc.poll() is None:
            terminate_process(keepalive_proc)

    _ = sniff_trial_log(time.perf_counter())

    elapsed_sec = time.perf_counter() - start
    avg_iter_ms = (sum(iter_times) / len(iter_times)) if iter_times else None

    if time_to_target is not None:
        status = "success"
        reason = "target_reached"
    elif reason == "oom_runtime":
        status = "oom"
    elif reason == "incomplete_due_to_throttling":
        status = "throttled"
    elif reason:
        status = "pruned"
    else:
        rc = proc.returncode if proc.returncode is not None else -999
        if rc == 0:
            status = "finished_no_target"
            reason = "did_not_reach_target"
        else:
            status = "failed"
            reason = f"process_exit_{rc}"

    return TrainOutcome(
        status=status,
        reason=reason,
        elapsed_sec=elapsed_sec,
        time_to_target_sec=time_to_target,
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
        peak_vram_gb=peak_vram_gb,
        avg_iter_ms=avg_iter_ms,
        time_weighted_convergence=(
            None if not math.isfinite(best_time_weighted_convergence) else best_time_weighted_convergence
        ),
        throttled=throttled,
    )


def trial_duration_seconds(trial: optuna.trial.FrozenTrial) -> Optional[float]:
    if trial.datetime_start is None or trial.datetime_complete is None:
        return None
    return (trial.datetime_complete - trial.datetime_start).total_seconds()


def export_trials_csv(study: optuna.Study, out_csv: Path) -> None:
    param_keys = sorted({k for t in study.trials for k in t.params.keys()})

    fieldnames = [
        "number",
        "state",
        "value",
        "duration_sec",
        "reason",
        "eval_interval",
        "adaptive_eval_interval",
        "n_head",
        "head_size",
        "batch_size",
        "estimated_vram_gb",
        "peak_vram_gb",
        "best_val_acc",
        "best_val_loss",
        "avg_iter_ms",
        "flash_attn2_applied",
        "config_path",
        "log_path",
        "seed",
    ] + [f"param_{k}" for k in param_keys]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in study.trials:
            row: Dict[str, Any] = {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "duration_sec": trial_duration_seconds(t),
                "reason": t.user_attrs.get("reason"),
                "eval_interval": t.user_attrs.get("eval_interval"),
                "adaptive_eval_interval": t.user_attrs.get("adaptive_eval_interval"),
                "n_head": t.user_attrs.get("n_head"),
                "head_size": t.user_attrs.get("head_size"),
                "batch_size": t.user_attrs.get("batch_size"),
                "estimated_vram_gb": t.user_attrs.get("estimated_vram_gb"),
                "peak_vram_gb": t.user_attrs.get("peak_vram_gb"),
                "best_val_acc": t.user_attrs.get("best_val_acc"),
                "best_val_loss": t.user_attrs.get("best_val_loss"),
                "avg_iter_ms": t.user_attrs.get("avg_iter_ms"),
                "flash_attn2_applied": t.user_attrs.get("flash_attn2_applied"),
                "config_path": t.user_attrs.get("config_path"),
                "log_path": t.user_attrs.get("log_path"),
                "seed": t.user_attrs.get("seed"),
            }
            for pk in param_keys:
                row[f"param_{pk}"] = t.params.get(pk)
            writer.writerow(row)


def write_champion_report(
    *,
    study: optuna.Study,
    out_json: Path,
    failure_penalty: float,
) -> Dict[str, Any]:
    complete = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and math.isfinite(float(t.value))
        and float(t.value) < failure_penalty
    ]

    if not complete:
        payload = {
            "found": False,
            "message": "No completed trial reached target criteria.",
        }
        out_json.write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    best = min(complete, key=lambda t: float(t.value))

    payload = {
        "found": True,
        "trial_number": best.number,
        "objective_time_sec": float(best.value),
        "params": best.params,
        "batch_size": best.user_attrs.get("batch_size"),
        "estimated_vram_gb": best.user_attrs.get("estimated_vram_gb"),
        "peak_vram_gb": best.user_attrs.get("peak_vram_gb"),
        "best_val_acc": best.user_attrs.get("best_val_acc"),
        "best_val_loss": best.user_attrs.get("best_val_loss"),
        "avg_iter_ms": best.user_attrs.get("avg_iter_ms"),
        "config_path": best.user_attrs.get("config_path"),
        "log_path": best.user_attrs.get("log_path"),
        "seed": best.user_attrs.get("seed"),
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def get_success_trials(study: optuna.Study, failure_penalty: float) -> List[optuna.trial.FrozenTrial]:
    return [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and math.isfinite(float(t.value))
        and float(t.value) < failure_penalty
    ]


def get_top_champions(
    study: optuna.Study,
    failure_penalty: float,
    top_k: int = 3,
) -> List[optuna.trial.FrozenTrial]:
    complete = get_success_trials(study, failure_penalty)
    complete.sort(key=lambda t: float(t.value))
    return complete[: max(0, int(top_k))]


def print_top_champions(top_trials: List[optuna.trial.FrozenTrial]) -> None:
    print("\n" + "=" * 72)
    print("Top 3 Champions (100% Accuracy)")
    print("=" * 72)

    if not top_trials:
        print("No successful trials reached target accuracy.")
        return

    print(
        f"{'Rank':<6}{'Trial ID':<10}{'Time (sec)':<12}{'n_embd':<10}"
        f"{'n_layer':<10}{'batch_size':<12}{'learning_rate':<14}"
    )
    print("-" * 72)

    for rank, t in enumerate(top_trials, start=1):
        t_time = float(t.value) if t.value is not None else math.inf
        n_embd = t.params.get("n_embd")
        n_layer = t.params.get("n_layer")
        batch_size = t.user_attrs.get("batch_size")
        learning_rate = t.params.get("learning_rate")
        lr_text = "-" if learning_rate is None else f"{float(learning_rate):.6g}"

        print(
            f"{rank:<6}{t.number:<10}{t_time:<12.4f}{str(n_embd):<10}"
            f"{str(n_layer):<10}{str(batch_size):<12}{lr_text:<14}"
        )


def write_top_champions_report(
    out_json: Path,
    top_trials: List[optuna.trial.FrozenTrial],
) -> Dict[str, Any]:
    payload = {
        "count": len(top_trials),
        "top3": [
            {
                "rank": rank,
                "trial_number": t.number,
                "time_sec": float(t.value) if t.value is not None else None,
                "n_embd": t.params.get("n_embd"),
                "n_layer": t.params.get("n_layer"),
                "batch_size": t.user_attrs.get("batch_size"),
                "learning_rate": t.params.get("learning_rate"),
                "config_path": t.user_attrs.get("config_path"),
                "log_path": t.user_attrs.get("log_path"),
            }
            for rank, t in enumerate(top_trials, start=1)
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def cleanup_non_top_artifacts(
    *,
    study: optuna.Study,
    keep_trial_numbers: set[int],
    configs_dir: Path,
    logs_dir: Path,
    ckpt_dir: Path,
) -> Dict[str, int]:
    removed = {"configs": 0, "logs": 0, "checkpoints": 0}

    for t in study.trials:
        if t.number in keep_trial_numbers:
            continue

        cfg_main = configs_dir / f"trial_{t.number:04d}.py"
        cfg_retry = configs_dir / f"trial_{t.number:04d}_retry.py"
        log_main = logs_dir / f"trial_{t.number:04d}.log"
        log_retry = logs_dir / f"trial_{t.number:04d}_retry.log"
        trial_ckpt = ckpt_dir / f"trial_{t.number:04d}"

        for cfg_path in (cfg_main, cfg_retry):
            if safe_unlink(cfg_path):
                removed["configs"] += 1
        for log_path in (log_main, log_retry):
            if safe_unlink(log_path):
                removed["logs"] += 1
        if safe_rmtree(trial_ckpt):
            removed["checkpoints"] += 1

    return removed


def get_best_completed_iter_ms(study: optuna.Study) -> Optional[float]:
    best_ms: Optional[float] = None
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        avg = t.user_attrs.get("avg_iter_ms")
        if isinstance(avg, (int, float)) and avg > 0:
            best_ms = float(avg) if best_ms is None else min(best_ms, float(avg))
    return best_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Bot for 4-digit addition")

    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--study-name", type=str, default="4digit_hyperopt")
    parser.add_argument("--storage", type=str, default="", help="Optuna storage URI or sqlite file path")

    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout-min", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--target-acc", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=SPRINT_DEFAULT_MAX_ITERS)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument(
        "--suggest-eval-interval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let Optuna suggest eval_interval in [eval-interval-min, eval-interval-max].",
    )
    parser.add_argument("--eval-interval-min", type=int, default=100)
    parser.add_argument("--eval-interval-max", type=int, default=500)
    parser.add_argument(
        "--adaptive-eval-interval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start from eval_interval and increase it during a trial when train loss is low.",
    )
    parser.add_argument("--adaptive-eval-loss-threshold", type=float, default=0.5)
    parser.add_argument("--adaptive-eval-multiplier", type=float, default=2.0)
    parser.add_argument("--adaptive-eval-max", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=100)

    parser.add_argument("--max-vram-gb", type=float, default=7.6)
    parser.add_argument("--vram-safety-margin", type=float, default=0.98)
    parser.add_argument("--vram-overhead-gb", type=float, default=0.9)
    parser.add_argument("--batch-search-min", type=int, default=128)
    parser.add_argument("--batch-search-max", type=int, default=4096)
    parser.add_argument("--batch-util-min", type=float, default=WIDE_BATCH_UTIL_MIN)
    parser.add_argument("--batch-util-max", type=float, default=WIDE_BATCH_UTIL_MAX)

    parser.add_argument("--throttle-power-w", type=float, default=35.0)
    parser.add_argument("--throttle-samples", type=int, default=3)
    parser.add_argument("--hw-sample-sec", type=float, default=2.0)
    parser.add_argument("--max-startup-seconds", type=float, default=120.0)
    parser.add_argument(
        "--max-trial-seconds",
        "--max_trial_time_sec",
        dest="max_trial_seconds",
        type=float,
        default=DEFAULT_MAX_TRIAL_SECONDS,
    )
    parser.add_argument(
        "--target-duration-sec",
        type=float,
        default=18.0,
        help="Sprint leaderboard target. Trials that do not reach target_acc before this are pruned.",
    )
    parser.add_argument("--gpu-prewarm-sec", type=float, default=2.5)
    parser.add_argument("--gpu-prewarm-dim", type=int, default=4096)
    parser.add_argument(
        "--anti-throttle-keepalive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a lightweight CUDA keepalive sidecar to maintain P-state/power and reduce false throttle pruning.",
    )
    parser.add_argument("--keepalive-dim", type=int, default=1536)
    parser.add_argument("--keepalive-sleep-sec", type=float, default=0.01)

    parser.add_argument("--eval-stagnation-window", type=int, default=2)
    parser.add_argument("--min-eval-loss-improvement-ratio", type=float, default=0.05)
    parser.add_argument("--slow-trial-factor", type=float, default=2.2)
    parser.add_argument("--tpe-startup-trials", type=int, default=TPE_STARTUP_TRIALS)
    parser.add_argument("--enqueue-elite-trials", type=int, default=12)
    parser.add_argument(
        "--enqueue-default-sprint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Queue built-in fast-regime seed trials before TPE exploration.",
    )
    parser.add_argument(
        "--history-results-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "hyperopt",
        help="Directory to mine historical elite trials for enqueue seeding.",
    )

    parser.add_argument("--failure-penalty", type=float, default=1_000_000.0)

    parser.add_argument(
        "--allow-battery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow run even if AC power is not detected",
    )
    parser.add_argument(
        "--allow-shared-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow run even if other GPU compute processes are detected",
    )
    parser.add_argument(
        "--allow-suboptimal-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow run when initial GPU is not in high-performance condition.",
    )
    parser.add_argument(
        "--auto-reduce-on-throttle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retry a throttled trial once with half batch size",
    )
    parser.add_argument(
        "--compile-trials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile model during trials (usually slower for small fast runs)",
    )
    parser.add_argument(
        "--force-flash-attn2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject attn_implementation='flash_attention_2' when flash-attn2 is available.",
    )
    parser.add_argument(
        "--wandb-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable wandb_log in generated trial configs",
    )
    parser.add_argument(
        "--wandb-run-prefix",
        type=str,
        default="",
        help="Optional prefix for wandb_run_name. Trial id is always appended.",
    )
    parser.add_argument(
        "--cleanup-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete trial config/log/checkpoint artifacts for non-success trials.",
    )
    parser.add_argument(
        "--cleanup-finished-no-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also delete artifacts for trials that finished but never reached target.",
    )
    parser.add_argument(
        "--keep-failed-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep trial logs even when cleanup is enabled.",
    )
    parser.add_argument(
        "--keep-failed-configs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep generated trial config files even when cleanup is enabled.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.base_config.exists():
        raise SystemExit(f"Base config not found: {args.base_config}")

    if not args.compile_trials:
        print("compile-trials disabled: using eager mode to reduce startup latency")

    if args.batch_util_max < args.batch_util_min:
        raise SystemExit("Invalid batch util range: batch-util-max must be >= batch-util-min")

    if args.eval_interval < 1:
        raise SystemExit("eval-interval must be >= 1")
    if args.eval_interval_max < args.eval_interval_min:
        raise SystemExit("eval-interval-max must be >= eval-interval-min")
    if args.eval_interval_min < 1:
        raise SystemExit("eval-interval-min must be >= 1")
    if args.adaptive_eval_max < 1:
        raise SystemExit("adaptive-eval-max must be >= 1")
    if args.adaptive_eval_multiplier < 1.0:
        raise SystemExit("adaptive-eval-multiplier must be >= 1.0")

    if args.target_duration_sec <= 0.0:
        raise SystemExit("target-duration-sec must be > 0")

    if args.max_trial_seconds <= 0.0:
        raise SystemExit("max-trial-seconds must be > 0")

    if args.target_duration_sec > args.max_trial_seconds:
        print(
            "target-duration-sec was larger than max-trial-seconds; "
            f"clamping to {args.max_trial_seconds:.1f}s"
        )
        args.target_duration_sec = args.max_trial_seconds

    random.seed(args.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    ac_state = detect_ac_power()
    if ac_state is False and not args.allow_battery:
        raise SystemExit(
            "AC power check failed (running on battery). Plug in AC or re-run with --allow-battery."
        )

    other_gpu_procs = detect_other_gpu_processes(os.getpid())
    busy = [p for p in other_gpu_procs if int(p.get("used_mb", 0)) >= 150]
    if busy and not args.allow_shared_gpu:
        raise SystemExit(
            "Detected other GPU compute processes. Close them first or use --allow-shared-gpu."
        )

    base_cfg = load_simple_config_vars(args.base_config)
    dataset_name = str(base_cfg.get("dataset", "basic"))
    block_size = int(base_cfg.get("block_size", 20))
    vocab_size = detect_vocab_size(dataset_name)
    flash_attn2_ready = flash_attn2_available()

    safe_study_name = sanitize_name(args.study_name)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_stamp}_{safe_study_name}"

    run_dir = PROJECT_DIR / "results" / "hyperopt" / run_id
    configs_dir = run_dir / "configs"
    logs_dir = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    reports_dir = run_dir / "reports"
    inductor_cache_dir = run_dir / "inductor_cache"

    for d in (configs_dir, logs_dir, ckpt_dir, reports_dir, inductor_cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    history_dir = args.history_results_dir
    if not history_dir.is_absolute():
        history_dir = (PROJECT_DIR / history_dir).resolve()

    if args.storage:
        if "://" in args.storage:
            storage_uri = args.storage
        else:
            storage_uri = f"sqlite:///{Path(args.storage).resolve().as_posix()}"
    else:
        storage_uri = f"sqlite:///{(run_dir / 'study.db').resolve().as_posix()}"

    gpu_probe = GPUProbe()
    initial_gpu = gpu_probe.sample()

    if initial_gpu is not None and not args.allow_suboptimal_gpu:
        pstate = initial_gpu.get("pstate")
        pstate_num = int(pstate) if isinstance(pstate, int) else None
        power_w = initial_gpu.get("power_w")
        if pstate_num is not None and pstate_num > 2:
            raise SystemExit(
                f"Initial GPU P-state is P{pstate_num} (>P2). Stabilize GPU performance or use --allow-suboptimal-gpu."
            )
        if power_w is not None and float(power_w) < 35.0:
            raise SystemExit(
                f"Initial GPU power is {float(power_w):.1f}W (<35W). Check power/performance mode or use --allow-suboptimal-gpu."
            )

    json_safe_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    run_meta = {
        "started_utc": utc_now(),
        "study_name": args.study_name,
        "run_id": run_id,
        "base_config": str(args.base_config),
        "dataset": dataset_name,
        "block_size": block_size,
        "vocab_size": vocab_size,
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "git_commit": current_git_commit(),
        "args": json_safe_args,
        "ac_power": ac_state,
        "detected_other_gpu_processes": busy,
        "initial_gpu": initial_gpu,
        "flash_attn2_available": flash_attn2_ready,
        "target_duration_sec": args.target_duration_sec,
        "anti_throttle_keepalive": bool(args.anti_throttle_keepalive),
        "keepalive_dim": args.keepalive_dim,
        "keepalive_sleep_sec": args.keepalive_sleep_sec,
        "history_results_dir": str(history_dir),
        "storage_uri": storage_uri,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    print("=" * 72)
    print("Hyperparameter Tuning Bot (4-digit addition)")
    print("=" * 72)
    print(f"Run dir      : {run_dir}")
    print(f"Study        : {args.study_name}")
    print(f"Storage      : {storage_uri}")
    print(f"Trials       : {args.n_trials}")
    print(f"Timeout (min): {args.timeout_min}")
    print(f"Target acc   : {args.target_acc}")
    print(f"Target time  : {args.target_duration_sec:.2f} sec")
    print(f"VRAM limit   : {args.max_vram_gb:.2f} GB")
    print(
        "Keepalive   : "
        f"{bool(args.anti_throttle_keepalive)} "
        f"(dim={args.keepalive_dim}, sleep={args.keepalive_sleep_sec:.4f}s)"
    )
    print(f"AC power     : {ac_state}")
    if initial_gpu is not None:
        print(
            "GPU init     : "
            f"used={initial_gpu.get('memory_used_gb', 0.0):.2f}GB, "
            f"power={initial_gpu.get('power_w', 0.0):.1f}W, "
            f"temp={initial_gpu.get('temperature_c', 0.0):.1f}C"
        )
    print(f"FlashAttn2   : {flash_attn2_ready}")
    print("=" * 72)

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=TPE_STARTUP_TRIALS,
        multivariate=True,
        seed=42,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=200,
        max_resource=args.max_iters,
        reduction_factor=3,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_uri,
        load_if_exists=True,
    )

    queued_seed_count = enqueue_seed_trials(
        study=study,
        history_dir=history_dir,
        include_default_sprint=bool(args.enqueue_default_sprint),
        elite_limit=max(0, int(args.enqueue_elite_trials)),
        batch_util_min=float(args.batch_util_min),
        batch_util_max=float(args.batch_util_max),
    )
    print(f"Seed queue   : {queued_seed_count}")

    total_trials = max(0, int(args.n_trials))
    trial_progress = tqdm(total=total_trials, unit="trial", dynamic_ncols=True)

    progress_state: Dict[str, Any] = {
        "done": 0,
        "last_time": None,
        "best_time": None,
        "status": "Running",
    }

    def fmt_time(seconds: Optional[float]) -> str:
        if seconds is None or not math.isfinite(float(seconds)):
            return "--"
        return f"{float(seconds):.3f}s"

    trial_progress.set_description(f"Trial [0/{total_trials}]")
    trial_progress.set_postfix(
        {
            "Last Time": "--",
            "Best Time": "--",
            "Status": "Running",
        }
    )

    def objective(trial: optuna.trial.Trial) -> float:
        current_idx = min(total_trials, int(progress_state["done"]) + 1)
        trial_progress.set_description(f"Trial [{current_idx}/{total_trials}]")
        trial_progress.set_postfix(
            {
                "Last Time": fmt_time(progress_state.get("last_time")),
                "Best Time": fmt_time(progress_state.get("best_time")),
                "Status": "Running",
            },
            refresh=True,
        )

        trial_seed = args.seed + trial.number

        n_layer = trial.suggest_int("n_layer", 2, 8)
        n_embd = trial.suggest_int("n_embd", SPRINT_N_EMBD_MIN, SPRINT_N_EMBD_MAX, step=32)
        head_size = 64 if (n_embd % 64 == 0) else 32
        n_head = n_embd // head_size

        learning_rate = trial.suggest_float("learning_rate", SPRINT_LR_MIN, SPRINT_LR_MAX, log=True)
        beta2 = trial.suggest_float("beta2", 0.95, 0.999)
        warmup_iters = trial.suggest_int("warmup_iters", 50, 200, step=25)
        grad_clip = trial.suggest_float("grad_clip", 0.5, 1.5)
        dropout = trial.suggest_float("dropout", 0.0, 0.08)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.01)

        if args.suggest_eval_interval:
            trial_eval_interval = trial.suggest_int(
                "eval_interval",
                int(args.eval_interval_min),
                int(args.eval_interval_max),
            )
        else:
            trial_eval_interval = int(args.eval_interval)

        adaptive_eval_enabled = bool(args.adaptive_eval_interval and not args.suggest_eval_interval)

        max_safe_batch = find_safe_batch_size(
            n_layer=n_layer,
            n_embd=n_embd,
            block_size=block_size,
            vocab_size=vocab_size,
            min_batch=args.batch_search_min,
            max_batch=args.batch_search_max,
            max_vram_gb=args.max_vram_gb,
            safety_margin=args.vram_safety_margin,
            vram_overhead_gb=args.vram_overhead_gb,
        )

        batch_util = trial.suggest_float("batch_util", args.batch_util_min, args.batch_util_max)
        batch_size = align_batch_size(int(max_safe_batch * batch_util), max_batch=max_safe_batch)
        batch_size = max(args.batch_search_min, min(max_safe_batch, batch_size))
        batch_size = align_batch_size(batch_size, max_batch=max_safe_batch)

        est_vram = estimate_vram_gb(
            n_layer=n_layer,
            n_embd=n_embd,
            block_size=block_size,
            batch_size=batch_size,
            vocab_size=vocab_size,
            base_cuda_overhead_gb=args.vram_overhead_gb,
        )
        if est_vram > args.max_vram_gb:
            raise TrialPruned("estimated_vram_above_limit")

        trial_ckpt_dir = ckpt_dir / f"trial_{trial.number:04d}"
        trial_ckpt_dir.mkdir(parents=True, exist_ok=True)

        trial_config = configs_dir / f"trial_{trial.number:04d}.py"
        trial_log = logs_dir / f"trial_{trial.number:04d}.log"
        trial_generated_configs: List[Path] = [trial_config]
        trial_generated_logs: List[Path] = [trial_log]

        base_wandb_name = str(base_cfg.get("wandb_run_name", safe_study_name))
        if args.wandb_run_prefix.strip():
            base_wandb_name = sanitize_name(args.wandb_run_prefix.strip())
        trial_wandb_name = f"{base_wandb_name}-trial_{trial.number:04d}"

        overrides: Dict[str, Any] = {
            "out_dir": str(trial_ckpt_dir),
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "beta2": beta2,
            "warmup_iters": warmup_iters,
            "grad_clip": grad_clip,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "max_iters": args.max_iters,
            "lr_decay_iters": args.max_iters,
            "eval_interval": trial_eval_interval,
            "adaptive_eval_interval": adaptive_eval_enabled,
            "adaptive_eval_loss_threshold": float(args.adaptive_eval_loss_threshold),
            "adaptive_eval_multiplier": float(args.adaptive_eval_multiplier),
            "adaptive_eval_max": int(args.adaptive_eval_max),
            "eval_iters": args.eval_iters,
            "log_interval": args.log_interval,
            "early_stop_loss": 0.0,
            "early_stop_acc": args.target_acc,
            "wandb_log": bool(args.wandb_log),
            "wandb_run_name": trial_wandb_name,
            "compile": bool(args.compile_trials),
            "dtype": "bfloat16",
            "always_save_checkpoint": False,
            "random_seed": trial_seed,
        }
        if args.force_flash_attn2 and flash_attn2_ready:
            overrides["attn_implementation"] = "flash_attention_2"

        append_overrides_to_config(args.base_config, trial_config, overrides)

        best_iter_ms = get_best_completed_iter_ms(study)
        slow_ceiling = None
        if best_iter_ms is not None and best_iter_ms > 0:
            slow_ceiling = best_iter_ms * args.slow_trial_factor

        outcome = run_training_trial(
            trial=trial,
            config_path=trial_config,
            log_path=trial_log,
            target_acc=args.target_acc,
            eval_stagnation_window=args.eval_stagnation_window,
            slow_iter_ms_ceiling=slow_ceiling,
            gpu_probe=gpu_probe,
            throttle_power_w=args.throttle_power_w,
            throttle_samples=args.throttle_samples,
            hw_sample_sec=args.hw_sample_sec,
            max_startup_seconds=args.max_startup_seconds,
            max_trial_seconds=args.max_trial_seconds,
            target_duration_sec=args.target_duration_sec,
            min_eval_loss_improvement_ratio=args.min_eval_loss_improvement_ratio,
            trial_seed=trial_seed,
            gpu_prewarm_sec=args.gpu_prewarm_sec,
            gpu_prewarm_dim=args.gpu_prewarm_dim,
            inductor_cache_dir=inductor_cache_dir,
            anti_throttle_keepalive=bool(args.anti_throttle_keepalive),
            keepalive_dim=args.keepalive_dim,
            keepalive_sleep_sec=args.keepalive_sleep_sec,
        )

        if outcome.status == "throttled" and args.auto_reduce_on_throttle and batch_size >= 2 * args.batch_search_min:
            reduced_batch = int((batch_size // 2) // 8) * 8
            reduced_batch = max(args.batch_search_min, reduced_batch)
            overrides["batch_size"] = reduced_batch

            trial_config_retry = configs_dir / f"trial_{trial.number:04d}_retry.py"
            trial_log_retry = logs_dir / f"trial_{trial.number:04d}_retry.log"
            trial_generated_configs.append(trial_config_retry)
            trial_generated_logs.append(trial_log_retry)
            append_overrides_to_config(args.base_config, trial_config_retry, overrides)

            retry_est_vram = estimate_vram_gb(
                n_layer=n_layer,
                n_embd=n_embd,
                block_size=block_size,
                batch_size=reduced_batch,
                vocab_size=vocab_size,
                base_cuda_overhead_gb=args.vram_overhead_gb,
            )

            outcome = run_training_trial(
                trial=trial,
                config_path=trial_config_retry,
                log_path=trial_log_retry,
                target_acc=args.target_acc,
                eval_stagnation_window=args.eval_stagnation_window,
                slow_iter_ms_ceiling=slow_ceiling,
                gpu_probe=gpu_probe,
                throttle_power_w=args.throttle_power_w,
                throttle_samples=args.throttle_samples,
                hw_sample_sec=args.hw_sample_sec,
                max_startup_seconds=args.max_startup_seconds,
                max_trial_seconds=args.max_trial_seconds,
                target_duration_sec=args.target_duration_sec,
                min_eval_loss_improvement_ratio=args.min_eval_loss_improvement_ratio,
                trial_seed=trial_seed,
                gpu_prewarm_sec=args.gpu_prewarm_sec,
                gpu_prewarm_dim=args.gpu_prewarm_dim,
                inductor_cache_dir=inductor_cache_dir,
                anti_throttle_keepalive=bool(args.anti_throttle_keepalive),
                keepalive_dim=args.keepalive_dim,
                keepalive_sleep_sec=args.keepalive_sleep_sec,
            )

            trial.set_user_attr("retry_from_throttle", True)
            trial.set_user_attr("batch_size_before_retry", batch_size)
            trial.set_user_attr("batch_size", reduced_batch)
            trial.set_user_attr("estimated_vram_gb", round(retry_est_vram, 4))
            trial.set_user_attr("config_path", str(trial_config_retry))
            trial.set_user_attr("log_path", str(trial_log_retry))
        else:
            trial.set_user_attr("batch_size", batch_size)
            trial.set_user_attr("estimated_vram_gb", round(est_vram, 4))
            trial.set_user_attr("config_path", str(trial_config))
            trial.set_user_attr("log_path", str(trial_log))

        trial.set_user_attr("seed", trial_seed)
        trial.set_user_attr("n_head", n_head)
        trial.set_user_attr("head_size", head_size)
        trial.set_user_attr("eval_interval", trial_eval_interval)
        trial.set_user_attr("adaptive_eval_interval", adaptive_eval_enabled)
        trial.set_user_attr("peak_vram_gb", round(outcome.peak_vram_gb, 4))
        trial.set_user_attr("best_val_acc", round(outcome.best_val_acc, 6))
        trial.set_user_attr(
            "best_val_loss",
            None if not math.isfinite(outcome.best_val_loss) else round(outcome.best_val_loss, 6),
        )
        trial.set_user_attr(
            "avg_iter_ms",
            None if outcome.avg_iter_ms is None else round(outcome.avg_iter_ms, 4),
        )
        trial.set_user_attr(
            "time_weighted_convergence",
            None
            if outcome.time_weighted_convergence is None
            else round(outcome.time_weighted_convergence, 8),
        )
        trial.set_user_attr("status", outcome.status)
        trial.set_user_attr("reason", outcome.reason)
        trial.set_user_attr("wandb_run_name", trial_wandb_name)
        trial.set_user_attr("flash_attn2_applied", bool(args.force_flash_attn2 and flash_attn2_ready))

        cleaned_paths: List[str] = []
        if args.cleanup_artifacts and should_cleanup_status(
            outcome.status, cleanup_finished_no_target=args.cleanup_finished_no_target
        ):
            if not args.keep_failed_configs:
                for cfg_path in trial_generated_configs:
                    if safe_unlink(cfg_path):
                        cleaned_paths.append(str(cfg_path))
            if not args.keep_failed_logs:
                for log_path in trial_generated_logs:
                    if safe_unlink(log_path):
                        cleaned_paths.append(str(log_path))
            if safe_rmtree(trial_ckpt_dir):
                cleaned_paths.append(str(trial_ckpt_dir))

        trial.set_user_attr("artifacts_cleaned", bool(cleaned_paths))
        if cleaned_paths:
            trial.set_user_attr("cleaned_paths", cleaned_paths)

        if outcome.status == "success" and outcome.time_to_target_sec is not None:
            return float(outcome.time_to_target_sec)

        if outcome.status in {"pruned", "oom", "throttled"}:
            raise TrialPruned(outcome.reason)

        if outcome.status == "finished_no_target":
            return float(args.failure_penalty + outcome.elapsed_sec)

        # For hard failures, keep the study running with penalty.
        return float(args.failure_penalty + outcome.elapsed_sec)

    def on_trial_complete(study_ref: optuna.Study, finished_trial: optuna.trial.FrozenTrial) -> None:
        _ = study_ref
        reason_text = str(finished_trial.user_attrs.get("reason", "")).lower()
        duration_sec = trial_duration_seconds(finished_trial)

        status = "Pruned"
        last_time = duration_sec

        if finished_trial.state == optuna.trial.TrialState.COMPLETE:
            if (
                finished_trial.value is not None
                and math.isfinite(float(finished_trial.value))
                and float(finished_trial.value) < args.failure_penalty
            ):
                status = "Success"
                last_time = float(finished_trial.value)
                prev_best = progress_state.get("best_time")
                if prev_best is None:
                    progress_state["best_time"] = last_time
                else:
                    progress_state["best_time"] = min(float(prev_best), float(last_time))
            else:
                status = "Pruned"
        elif finished_trial.state == optuna.trial.TrialState.PRUNED:
            status = "Throttle" if "throttling" in reason_text else "Pruned"

        progress_state["done"] = int(progress_state["done"]) + 1
        progress_state["last_time"] = last_time
        progress_state["status"] = status

        trial_progress.update(1)
        done_idx = min(total_trials, int(progress_state["done"]))
        trial_progress.set_description(f"Trial [{done_idx}/{total_trials}]")
        trial_progress.set_postfix(
            {
                "Last Time": fmt_time(progress_state.get("last_time")),
                "Best Time": fmt_time(progress_state.get("best_time")),
                "Status": str(progress_state.get("status", "Pruned")),
            },
            refresh=True,
        )

        under_target_trials = [
            t
            for t in study_ref.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and math.isfinite(float(t.value))
            and float(t.value) < args.failure_penalty
            and float(t.value) < args.target_duration_sec
        ]
        if len(under_target_trials) >= 3:
            progress_state["status"] = "Top3Found"
            trial_progress.set_postfix(
                {
                    "Last Time": fmt_time(progress_state.get("last_time")),
                    "Best Time": fmt_time(progress_state.get("best_time")),
                    "Status": "Top3Found",
                },
                refresh=True,
            )
            study_ref.stop()

    timeout_sec = max(0, int(args.timeout_min * 60))

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=timeout_sec,
            callbacks=[on_trial_complete],
            gc_after_trial=True,
        )
    finally:
        trial_progress.close()
        gpu_probe.close()

    top3_trials = get_top_champions(study, args.failure_penalty, top_k=3)
    top3_json = reports_dir / "top3_champions.json"
    top3_payload = write_top_champions_report(top3_json, top3_trials)
    print_top_champions(top3_trials)

    keep_trials = {t.number for t in top3_trials}
    cleanup_summary = cleanup_non_top_artifacts(
        study=study,
        keep_trial_numbers=keep_trials,
        configs_dir=configs_dir,
        logs_dir=logs_dir,
        ckpt_dir=ckpt_dir,
    )
    print(
        "Artifact cleanup: "
        f"configs={cleanup_summary['configs']}, "
        f"logs={cleanup_summary['logs']}, "
        f"checkpoints={cleanup_summary['checkpoints']}"
    )

    trials_csv = reports_dir / "trials.csv"
    export_trials_csv(study, trials_csv)

    champion_json = reports_dir / "champion.json"
    champion = write_champion_report(
        study=study,
        out_json=champion_json,
        failure_penalty=args.failure_penalty,
    )

    try:
        best_trial_number = study.best_trial.number
        best_value = study.best_value
    except ValueError:
        best_trial_number = None
        best_value = None

    done_meta = {
        "finished_utc": utc_now(),
        "best_value": best_value,
        "best_trial": best_trial_number,
        "n_trials_total": len(study.trials),
        "top3_trial_numbers": [t.number for t in top3_trials],
        "artifact_cleanup": cleanup_summary,
        "reports": {
            "trials_csv": str(trials_csv),
            "champion_json": str(champion_json),
            "top3_json": str(top3_json),
        },
    }
    (run_dir / "run_done.json").write_text(json.dumps(done_meta, indent=2) + "\n")

    print("\n" + "=" * 72)
    print("Tuning finished")
    print("=" * 72)
    print(f"Trials CSV   : {trials_csv}")
    print(f"Champion JSON: {champion_json}")
    if champion.get("found"):
        print(
            "Champion     : "
            f"trial #{champion['trial_number']} | time={champion['objective_time_sec']:.3f}s | "
            f"batch={champion.get('batch_size')}"
        )
        print(f"Config       : {champion.get('config_path')}")
    else:
        print("Champion     : none (no completed target-reaching trial)")


if __name__ == "__main__":
    main()
