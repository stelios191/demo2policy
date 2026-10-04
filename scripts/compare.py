"""Compare a reproduction with the committed results, file by file and number by number.

Values that are expected to differ between runs are ignored: file hashes of checkpoints and
state banks, run identifiers, runtimes, package versions, and the timestamps in run folder
names. Everything else, including every trial's outcome, must match.

    python -m scripts.compare reproduction
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from scripts.common import DIAGNOSIS, FIRST_EVALUATION, ROOT, TIMING_EXPERIMENT

VOLATILE_KEYS = {
    "checkpoint_sha256",
    "state_bank_sha256",
    "development_state_bank_sha256",
    "selection_sha256",
    "config_sha256",
    "sha256",
    "run_id",
    "runtime_seconds",
    "versions",
}
VOLATILE_COLUMNS = {"run_id", "checkpoint_sha256", "state_bank_sha256"}
TIMESTAMP = re.compile(r"\b\d{14}\b")
SKIPPED = {"environment.json"}  # describes the machine, compared separately
FULL_RUN_ONLY = (f"{DIAGNOSIS}/explore_", f"{DIAGNOSIS}/validation_vs_success.json")
SCOPES = ("quick", "core", "full")


def expected_in(relative, scope):
    """Whether a reproduction of the given scope should have produced this record."""
    if scope == "quick":  # the dataset records and the first model's evaluation
        first = relative.startswith(f"{FIRST_EVALUATION}/") and not relative.startswith(f"{TIMING_EXPERIMENT}/")
        return "/" not in relative or first
    if scope == "core":
        return not relative.startswith(FULL_RUN_ONLY)
    return True


FLOAT_TOLERANCE = 1e-6


def normalize(value):
    """Drop values that legitimately differ between runs."""
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, str):
        return TIMESTAMP.sub("<time>", value)
    return value


def differences(committed, reproduced, where=""):
    """Human-readable differences between two normalized values; empty when they match."""
    label = f"{where}: " if where else ""
    if isinstance(committed, dict) and isinstance(reproduced, dict):
        found = (
            [f"{label}keys differ: {sorted(set(committed) ^ set(reproduced))}"]
            if committed.keys() != reproduced.keys()
            else []
        )
        for key in committed.keys() & reproduced.keys():
            found += differences(committed[key], reproduced[key], f"{where}.{key}" if where else key)
        return found
    if isinstance(committed, list) and isinstance(reproduced, list):
        if len(committed) != len(reproduced):
            return [f"{label}length: committed {len(committed)}, reproduced {len(reproduced)}"]
        found = []
        for i, (a, b) in enumerate(zip(committed, reproduced, strict=True)):
            found += differences(a, b, f"{where}[{i}]")
        return found
    numbers = (int, float)
    if isinstance(committed, numbers) and isinstance(reproduced, numbers) and not isinstance(committed, bool):
        if abs(committed - reproduced) <= FLOAT_TOLERANCE * max(1.0, abs(committed)):
            return []
    elif committed == reproduced:
        return []
    return [f"{label}committed {committed}, reproduced {reproduced}"]


def load(path):
    if path.suffix == ".json":
        return normalize(json.loads(path.read_text(encoding="utf-8")))
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {k: TIMESTAMP.sub("<time>", v) for k, v in row.items() if k not in VOLATILE_COLUMNS}
            for row in csv.DictReader(f)
        ]


def compare_trees(committed_dir: Path, reproduced_dir: Path, scope="core"):
    """Compare every committed JSON and CSV record with its reproduction. Returns (rows, problem count)."""
    rows, problems = [], 0
    files = sorted(p for p in Path(committed_dir).rglob("*") if p.suffix in (".json", ".csv") and p.name not in SKIPPED)
    for path in files:
        relative = str(path.relative_to(committed_dir))
        twin = Path(reproduced_dir) / relative
        if not expected_in(relative, scope):
            rows.append(
                {
                    "file": relative,
                    "status": f"not run ({'quick check' if scope == 'quick' else 'needs --full'})",
                    "detail": "",
                }
            )
            continue
        if not twin.exists():
            rows.append({"file": relative, "status": "missing", "detail": ""})
            problems += 1
            continue
        found = differences(load(path), load(twin))
        rows.append({"file": relative, "status": "differs" if found else "same", "detail": "; ".join(found[:3])})
        problems += bool(found)
    return rows, problems


def package_differences(committed_dir: Path, reproduced_dir: Path):
    """Package versions that differ between the two environments, for context."""
    paths = [Path(d) / "environment.json" for d in (committed_dir, reproduced_dir)]
    if not all(p.exists() for p in paths):
        return None
    a, b = (json.loads(p.read_text(encoding="utf-8")).get("packages", {}) for p in paths)
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reproduction", type=Path, help="the folder a reproduction was written to")
    parser.add_argument("--committed", type=Path, default=ROOT)
    parser.add_argument("--scope", choices=SCOPES, default="core", help="which part of the work was reproduced")
    args = parser.parse_args(argv)
    committed, reproduced = args.committed / "artifacts", args.reproduction / "artifacts"
    rows, problems = compare_trees(committed, reproduced, args.scope)
    for row in rows:
        if row["status"] not in ("same",) and not row["status"].startswith("not run"):
            print(f"  {row['status']:24s} {row['file']}  {row['detail']}")
    counts = {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})}
    print(f"\n{len(rows)} recorded files compared: " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    packages = package_differences(committed, reproduced)
    if packages:
        print(f"package versions differ from the original run: {packages}")
    print(
        "REPRODUCED: every compared result matches"
        if problems == 0
        else f"NOT REPRODUCED: {problems} file(s) differ or are missing"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
