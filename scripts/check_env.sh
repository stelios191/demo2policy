#!/usr/bin/env bash
# Non-mutating environment diagnostics. Installs nothing and changes no settings.
# Usage:
#   scripts/check_env.sh            human-readable report
#   scripts/check_env.sh --json     JSON on stdout (used to produce artifacts/environment.json);
#                                   omits the CPU model, memory, core count and OS version
# Set PYTHON to choose the interpreter (default: .venv/bin/python, then python3).
# Set UPSTREAM_DIR to the directory holding the robomimic/robosuite checkouts (default: ../upstream).
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-$ROOT/../upstream}"
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="$(command -v python3)"; fi
fi

commit_of() {
  if [ -d "$1/.git" ]; then git -C "$1" rev-parse HEAD 2>/dev/null; else echo "unavailable"; fi
}
describe_of() {
  if [ -d "$1/.git" ]; then git -C "$1" describe --tags --always 2>/dev/null; else echo "unavailable"; fi
}

export D2P_ARCH="$(uname -m)"
export D2P_OS_NAME="$(sw_vers -productName 2>/dev/null || uname -s)"
export D2P_OS_VERSION="$(sw_vers -productVersion 2>/dev/null || uname -r)"
export D2P_CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
export D2P_NCPU="$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
export D2P_MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo unknown)"
export D2P_ROSETTA="$(sysctl -n sysctl.proc_translated 2>/dev/null || echo unknown)"
export D2P_ROBOMIMIC_COMMIT="$(commit_of "$UPSTREAM_DIR/robomimic")"
export D2P_ROBOMIMIC_DESCRIBE="$(describe_of "$UPSTREAM_DIR/robomimic")"
export D2P_ROBOSUITE_COMMIT="$(commit_of "$UPSTREAM_DIR/robosuite")"
export D2P_ROBOSUITE_DESCRIBE="$(describe_of "$UPSTREAM_DIR/robosuite")"
export D2P_ROOT="$ROOT"
export D2P_MODE="${1:-text}"

"$PYTHON" - <<'EOF' 2>/dev/null
import importlib.metadata as md, json, os, platform, sys

def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

root = os.environ["D2P_ROOT"]
exe = sys.executable
python_path = os.path.relpath(exe, root) if exe.startswith(root) else os.path.basename(exe)
mem = as_int(os.environ["D2P_MEM_BYTES"])
packages = ["torch", "torchvision", "robomimic", "robosuite", "mujoco", "numpy", "h5py",
            "numba", "scipy", "matplotlib", "imageio", "imageio-ffmpeg", "egl-probe", "pytest"]
report = {
    "machine": {
        "architecture": os.environ["D2P_ARCH"],
        "cpu": os.environ["D2P_CPU"],
        "logical_cpus": as_int(os.environ["D2P_NCPU"]),
        "memory_gb": round(mem / 2**30, 1) if mem else None,
        "os": f'{os.environ["D2P_OS_NAME"]} {os.environ["D2P_OS_VERSION"]}',
        "rosetta_translated": os.environ["D2P_ROSETTA"],
    },
    "python": {
        "executable": python_path,
        "version": platform.python_version(),
        "machine": platform.machine(),
    },
    "packages": {name: version(name) for name in packages},
    "upstream": {
        "robomimic": {"commit": os.environ["D2P_ROBOMIMIC_COMMIT"], "describe": os.environ["D2P_ROBOMIMIC_DESCRIBE"]},
        "robosuite": {"commit": os.environ["D2P_ROBOSUITE_COMMIT"], "describe": os.environ["D2P_ROBOSUITE_DESCRIBE"]},
    },
}
try:
    import torch
    report["torch"] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "num_threads": torch.get_num_threads(),
    }
except Exception as exc:  # diagnostics must report, not crash
    report["torch"] = {"error": f"{type(exc).__name__}: {exc}"}

if os.environ["D2P_MODE"] == "--json":
    report["machine"] = {"architecture": os.environ["D2P_ARCH"], "os": os.environ["D2P_OS_NAME"]}
    report["torch"].pop("num_threads", None)
    print(json.dumps(report, indent=2, sort_keys=True))
else:
    for section, values in report.items():
        print(f"[{section}]")
        for key, value in values.items():
            print(f"  {key}: {value}")
EOF
