"""Launch robomimic's training loop on a project config.

This adds no training logic. It exists because the upstream entry point
(1) resolves relative output paths against the robomimic package directory,
(2) prompts interactively when the run directory exists, and
(3) catches training exceptions and still exits with status 0.
Here dataset and output paths resolve against the output root (the repository, or
D2P_OUT when set), an existing run directory is an error, and exceptions propagate.
After training, the expected checkpoints are verified and hashed into run_record.json.

    python -m scripts.train --config configs/bc_smoke.json
"""

import argparse
import importlib.metadata as metadata
import json
import platform
import sys
import time
from pathlib import Path

from scripts.common import OUT, relative_path, sha256_file, write_json


def load_config(config_path: Path, name: str | None = None, seed: int | None = None):
    """Build a locked robomimic config with repository-relative paths made absolute."""
    from robomimic.config import config_factory

    external = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = config_factory(external["algo_name"])
    with config.values_unlocked():
        config.update(external)
        if name is not None:
            config.experiment.name = name
        if seed is not None:
            config.train.seed = seed
        config.train.output_dir = str((OUT / config.train.output_dir).resolve())
        config.train.data = [
            {**dict(entry), "path": str((OUT / entry["path"]).resolve())} for entry in config.train.data
        ]
    config.lock()
    return config


def expected_checkpoints(config) -> list[str]:
    """File names of the checkpoints the config schedules."""
    return [f"model_epoch_{epoch}.pth" for epoch in config.experiment.save.epochs]


def write_run_record(config, config_path: Path, run_dir: Path, runtime_seconds: float) -> dict:
    """Verify the expected checkpoints exist and record their hashes, versions and runtime."""
    model_dir = run_dir / "models"
    missing = [name for name in expected_checkpoints(config) if not (model_dir / name).exists()]
    if missing:
        raise RuntimeError(f"training returned but expected checkpoints are missing: {missing}")

    steps = config.experiment.epoch_every_n_steps
    manifest_path = OUT / "artifacts" / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    record = {
        "experiment": config.experiment.name,
        "config_file": relative_path(config_path),
        "config_sha256": sha256_file(config_path),
        "run_dir": relative_path(run_dir),
        "seed": config.train.seed,
        "device": "cpu",
        "updates": config.train.num_epochs * steps,
        "updates_per_epoch": steps,
        "runtime_seconds": round(runtime_seconds, 1),
        "checkpoints": [
            {
                "file": name,
                "update": epoch * steps,
                "sha256": sha256_file(model_dir / name),
            }
            for epoch, name in zip(config.experiment.save.epochs, expected_checkpoints(config), strict=True)
        ],
        "source_data_sha256": manifest.get("source_sha256"),
        "working_data_sha256": sha256_file(config.train.data[0]["path"]),
        "versions": {
            "python": platform.python_version(),
            **{pkg: metadata.version(pkg) for pkg in ("torch", "robomimic", "robosuite", "mujoco", "numpy", "h5py")},
        },
    }
    write_json(run_dir / "run_record.json", record)
    return record


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--name", default=None, help="override experiment.name")
    parser.add_argument("--seed", type=int, default=None, help="override train.seed")
    args = parser.parse_args(argv)

    import torch
    from robomimic.scripts.train import train

    config = load_config(args.config, args.name, args.seed)
    experiment_dir = Path(config.train.output_dir) / config.experiment.name
    if experiment_dir.exists():
        print(f"error: {experiment_dir} exists; remove it or pass a different --name", file=sys.stderr)
        return 1
    if config.train.cuda:
        print("error: this project trains on CPU; set train.cuda to false", file=sys.stderr)
        return 1

    started = time.time()
    train(config, device=torch.device("cpu"))
    runtime = time.time() - started

    run_dir = sorted(p for p in experiment_dir.iterdir() if p.is_dir())[-1]
    record = write_run_record(config, args.config, run_dir, runtime)
    print(f"finished {record['updates']} updates in {record['runtime_seconds']} s")
    print(f"run record: {run_dir / 'run_record.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
