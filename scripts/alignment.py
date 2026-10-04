"""Timing experiment: pair each training observation with the action taken `offset` steps later.

Two derived datasets are written from the clean working copy:

  matched_aligned  observations 0..T-offset-1 with the actions from the same steps
  shift_plus_N     the same observations with the actions from `offset` steps later

Both contain identical observations and sample counts, so they differ only in the
labels. Only training episodes are changed; validation and test episodes are copied
unchanged. The shift is applied inside each episode and never crosses a boundary.

These files are supervised behavior-cloning data. Training episodes keep only
observations and actions, so the files are not valid for demonstration replay or
as transition data for reinforcement learning.

    python -m scripts.alignment --source data/processed/lift_clean.hdf5
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from scripts.common import OUT, sha256_file, write_json
from scripts.prepare_data import MASK_NAMES

USAGE_NOTE = (
    "supervised behavior-cloning data only; training episodes hold observations and actions and are "
    "not valid for demonstration replay or as reinforcement-learning transitions"
)


def make_alignment_pairs(
    obs: dict[str, np.ndarray], actions: np.ndarray, offset: int = 3
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return (observations 0:T-offset, aligned actions 0:T-offset, shifted actions offset:T)."""
    if offset <= 0:
        raise ValueError(f"offset must be positive, got {offset}")
    length = actions.shape[0]
    for key, value in obs.items():
        if value.shape[0] != length:
            raise ValueError(f"observation '{key}' has length {value.shape[0]} but actions have length {length}")
    if length <= offset:
        raise ValueError(f"episode of length {length} is too short for an offset of {offset}")
    keep = length - offset
    return (
        {key: value[:keep].copy() for key, value in obs.items()},
        actions[:keep].copy(),
        actions[offset:].copy(),
    )


def _check_paths(source, aligned_path, shifted_path):
    """Refuse colliding paths and existing outputs."""
    if len({source.resolve(), aligned_path.resolve(), shifted_path.resolve()}) != 3:
        raise ValueError("source, aligned and shifted paths must be three distinct files")
    for path in (aligned_path, shifted_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; derived datasets are not overwritten")


def _training_pairs(src, manifest, source_name, offset):
    """Build the trimmed observations and both label sets for every training episode."""
    for name, mask in MASK_NAMES.items():
        in_file = [x.decode() for x in src["mask"][mask][:]]
        if in_file != list(manifest[name]):
            raise ValueError(f"manifest '{name}' episodes do not match mask '{mask}' in {source_name}")
    pairs = {}
    for episode_id in manifest["train"]:
        episode = src["data"][episode_id]
        obs = {key: episode["obs"][key][:] for key in episode["obs"]}
        try:
            pairs[episode_id] = make_alignment_pairs(obs, episode["actions"][:], offset)
        except ValueError as exc:
            raise ValueError(f"{episode_id}: {exc}") from exc
    return pairs


def _write_condition(src, out, pairs, label_index):
    """Copy held-out episodes unchanged and write training episodes with the chosen labels."""
    src.copy("mask", out)
    data = out.create_group("data")
    for key, value in src["data"].attrs.items():
        data.attrs[key] = value
    total = 0
    for episode_id in src["data"]:
        if episode_id in pairs:
            group = data.create_group(episode_id)
            for key, value in src["data"][episode_id].attrs.items():
                group.attrs[key] = value
            labels = pairs[episode_id][label_index]
            group.attrs["num_samples"] = labels.shape[0]
            group.create_dataset("actions", data=labels)
            for key, value in pairs[episode_id][0].items():
                group.create_dataset(f"obs/{key}", data=value)
        else:
            src.copy(f"data/{episode_id}", data)
        total += data[episode_id]["actions"].shape[0]
    data.attrs["total"] = total


def write_alignment_datasets(
    source: Path, manifest: dict, aligned_path: Path, shifted_path: Path, offset: int = 3
) -> None:
    """Write the matched control and the shifted dataset. Nothing is written if any episode is unusable."""
    source, aligned_path, shifted_path = Path(source), Path(aligned_path), Path(shifted_path)
    _check_paths(source, aligned_path, shifted_path)
    checksum = sha256_file(source)
    with h5py.File(source, "r") as src:
        pairs = _training_pairs(src, manifest, source.name, offset)
        conditions = ((aligned_path, 1, "matched_aligned"), (shifted_path, 2, f"shift_plus_{offset}"))
        for path, label_index, condition in conditions:
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as out:
                out.attrs["d2p_condition"] = condition
                out.attrs["d2p_offset_steps"] = offset
                out.attrs["d2p_source_sha256"] = checksum
                out.attrs["d2p_usage"] = USAGE_NOTE
                _write_condition(src, out, pairs, label_index)
    if sha256_file(source) != checksum:
        raise RuntimeError(f"source checksum changed while writing derived datasets: {source}")


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=OUT / "data/processed/lift_clean.hdf5")
    parser.add_argument("--manifest", type=Path, default=OUT / "artifacts/split_manifest.json")
    parser.add_argument("--aligned", type=Path, default=OUT / "data/processed/lift_matched_aligned.hdf5")
    parser.add_argument("--shifted", type=Path, default=OUT / "data/processed/lift_shift_plus_3.hdf5")
    parser.add_argument("--offset", type=int, default=3)
    parser.add_argument(
        "--record", type=Path, default=OUT / "artifacts/1_first_evaluation/timing_experiment/datasets.json"
    )
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    write_alignment_datasets(args.source, manifest, args.aligned, args.shifted, args.offset)

    record = {"offset_control_steps": args.offset, "source_sha256": sha256_file(args.source), "usage": USAGE_NOTE}
    for name, path in (("matched_aligned", args.aligned), (f"shift_plus_{args.offset}", args.shifted)):
        with h5py.File(path, "r") as f:
            train = [x.decode() for x in f["mask"][MASK_NAMES["train"]][:]]
            record[name] = {
                "file": path.name,
                "sha256": sha256_file(path),
                "train_episodes": len(train),
                "train_samples": int(sum(f["data"][e]["actions"].shape[0] for e in train)),
                "total_samples": int(f["data"].attrs["total"]),
            }
    write_json(args.record, record)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
