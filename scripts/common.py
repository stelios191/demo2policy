"""Helpers shared by every script: where inputs and outputs live, file hashes and JSON output."""

import hashlib
import json
import os
from pathlib import Path

# Code, configs and protocols are read from the repository.
ROOT = Path(__file__).resolve().parents[1]
# Everything the scripts produce (processed data, trained models, results, charts) is written
# under OUT. It defaults to the repository; set D2P_OUT to reproduce into a separate folder.
OUT = Path(os.environ.get("D2P_OUT", ROOT)).resolve()

# Recorded results, one folder per stage of the work, under OUT / "artifacts".
FIRST_EVALUATION = "1_first_evaluation"
TIMING_EXPERIMENT = f"{FIRST_EVALUATION}/timing_experiment"
DIAGNOSIS = "2_diagnosis"
FOLLOW_UP = "3_follow_up"
MULTI_SEED = "4_multi_seed"
ARTIFACTS = OUT / "artifacts"


def relative_path(path):
    """A path as recorded in result files: relative to OUT, else to the repository, else the file name."""
    path = Path(path).resolve()
    for base in (OUT, ROOT):
        if path.is_relative_to(base):
            return str(path.relative_to(base))
    return path.name


def sha256_file(path):
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    """Write JSON with a trailing newline, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
