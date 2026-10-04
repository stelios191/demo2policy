import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def paths_with(out):
    """OUT and two recorded paths, as seen by a fresh interpreter with D2P_OUT set."""
    code = (
        "import json; from scripts.common import OUT, ROOT, relative_path; "
        "print(json.dumps([str(OUT), relative_path(OUT / 'runs/x/model.pth'), relative_path(ROOT / 'configs/a.json')]))"
    )
    env = {**os.environ, "D2P_OUT": str(out)} if out else {k: v for k, v in os.environ.items() if k != "D2P_OUT"}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_outputs_default_to_the_repository():
    out, run, config = paths_with(None)
    assert out == str(ROOT)
    assert run == "runs/x/model.pth" and config == "configs/a.json"


def test_outputs_follow_d2p_out_and_recorded_paths_stay_relative(tmp_path):
    out, run, config = paths_with(tmp_path)
    assert out == str(tmp_path.resolve())
    assert run == "runs/x/model.pth"
    assert config == "configs/a.json"
