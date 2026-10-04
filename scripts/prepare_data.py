r"""Audit a robomimic demonstration file, split it by episode, and write a working copy.

The raw download is opened read-only and never modified. The working copy is a
byte copy of the source plus three episode masks (d2p_train, d2p_val, d2p_test).

    python -m scripts.prepare_data \
        --source data/raw/lift/ph/low_dim_v15.hdf5 \
        --destination data/processed/lift_clean.hdf5
"""

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

from scripts.common import OUT, sha256_file, write_json

REQUIRED_OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
MASK_NAMES = {"train": "d2p_train", "val": "d2p_val", "test": "d2p_test"}
SPLIT_COUNTS = {"train": 160, "val": 20, "test": 20}
EXPECTED_EPISODES = sum(SPLIT_COUNTS.values())
# Actions are normalized controller inputs; values outside this range are reported, not rejected.
NOMINAL_ACTION_RANGE = (-1.0, 1.0)
SPLIT_ALGORITHM = (
    "sorted(episode_ids) using Python string ordering, then "
    "numpy.random.default_rng(seed).permutation over the sorted list; "
    "first 160 -> train, next 20 -> val, last 20 -> test"
)


class AuditError(ValueError):
    """The dataset has defects that must be resolved before training."""

    def __init__(self, problems):
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"dataset audit failed with {len(self.problems)} problem(s):\n{lines}")


def _episode_sort_key(episode_id):
    """Order demo_2 before demo_10."""
    suffix = episode_id.rsplit("_", 1)[-1]
    return (0, int(suffix), episode_id) if suffix.isdigit() else (1, 0, episode_id)


def _check_array(episode, name, expected_length, problems):
    """Return the array if it is usable, recording any defect in `problems`."""
    if name not in episode or not isinstance(episode[name], h5py.Dataset):
        problems.append(f"{episode.name.split('/')[-1]}: missing key '{name}'")
        return None
    array = episode[name][:]
    episode_id = episode.name.split("/")[-1]
    if array.ndim != 2:
        problems.append(f"{episode_id}: '{name}' has shape {array.shape}, expected 2 dimensions (time, features)")
        return None
    if expected_length is not None and array.shape[0] != expected_length:
        problems.append(f"{episode_id}: '{name}' has length {array.shape[0]} but actions have length {expected_length}")
        return None
    if not np.issubdtype(array.dtype, np.number):
        problems.append(f"{episode_id}: '{name}' has non-numeric dtype {array.dtype}")
        return None
    bad = int(np.count_nonzero(~np.isfinite(array)))
    if bad:
        problems.append(f"{episode_id}: '{name}' contains {bad} non-finite value(s) (NaN or Inf)")
        return None
    return array


def _audit_episode(episode, feature_dims, ranges, problems, warnings):
    """Check one episode, updating the running shapes and ranges. Returns its length, or None if unusable."""
    episode_id = episode.name.split("/")[-1]
    if "actions" in episode and episode["actions"].shape[0] == 0:
        problems.append(f"{episode_id}: zero-length episode")
        return None
    actions = _check_array(episode, "actions", None, problems)
    length = None if actions is None else actions.shape[0]
    arrays = {"actions": actions}
    for key in REQUIRED_OBS_KEYS:
        arrays[f"obs/{key}"] = _check_array(episode, f"obs/{key}", length, problems)

    for name, array in arrays.items():
        if array is None:
            continue
        dims = list(array.shape[1:])
        if feature_dims.setdefault(name, dims) != dims:
            problems.append(
                f"{episode_id}: '{name}' has shape {array.shape}, "
                f"inconsistent with feature shape {feature_dims[name]} seen in earlier episodes"
            )
            continue
        low, high = array.min(axis=0), array.max(axis=0)
        if name in ranges:
            low, high = np.minimum(ranges[name][0], low), np.maximum(ranges[name][1], high)
        ranges[name] = (low, high)

    if actions is None:
        return None
    low, high = NOMINAL_ACTION_RANGE
    outside = int(np.count_nonzero((actions < low) | (actions > high)))
    if outside:
        warnings.append(f"{episode_id}: 'actions' has {outside} value(s) outside the nominal range [{low}, {high}]")
    if "num_samples" in episode.attrs and int(episode.attrs["num_samples"]) != length:
        warnings.append(
            f"{episode_id}: attribute num_samples={int(episode.attrs['num_samples'])} "
            f"differs from actions length {length}"
        )
    return length


def _environment_summary(env_args):
    """The simulator settings a reader needs in order to interpret observations and actions."""
    env_args = env_args or {}
    env_kwargs = env_args.get("env_kwargs", {})
    controller = env_kwargs.get("controller_configs", {})
    arm = next(iter(controller.get("body_parts", {}).values()), controller)
    return {
        "env_name": env_args.get("env_name"),
        "env_version": env_args.get("env_version"),
        "robots": env_kwargs.get("robots"),
        "control_freq_hz": env_kwargs.get("control_freq"),
        "controller_type": arm.get("type"),
        "controller_control_delta": arm.get("control_delta"),
        "controller_input_range": [arm.get("input_min"), arm.get("input_max")],
        "controller_output_min": arm.get("output_min"),
        "controller_output_max": arm.get("output_max"),
        "gripper_controller": (arm.get("gripper") or {}).get("type"),
    }


def audit_dataset(path: Path) -> dict:
    """Check every episode and return a summary. Raises AuditError listing every defect found."""
    path = Path(path)
    problems, warnings = [], []
    lengths, feature_dims, ranges = {}, {}, {}

    with h5py.File(path, "r") as f:
        if "data" not in f or len(f["data"]) == 0:
            raise AuditError([f"{path.name}: no episodes found under 'data'"])
        data = f["data"]
        env_args = json.loads(data.attrs["env_args"]) if "env_args" in data.attrs else None
        if env_args is None:
            warnings.append("data group has no 'env_args' attribute; simulator metadata is unavailable")
        declared_total = int(data.attrs["total"]) if "total" in data.attrs else None
        existing_masks = {k: int(len(f["mask"][k])) for k in f["mask"]} if "mask" in f else {}
        for episode_id in sorted(data.keys(), key=_episode_sort_key):
            length = _audit_episode(data[episode_id], feature_dims, ranges, problems, warnings)
            if length is not None:
                lengths[episode_id] = length

    if problems:
        raise AuditError(problems)

    total = int(sum(lengths.values()))
    if declared_total is not None and declared_total != total:
        warnings.append(f"data attribute total={declared_total} differs from summed episode length {total}")
    length_values = np.array(list(lengths.values()))
    return {
        "source_file": path.name,
        "source_sha256": sha256_file(path),
        "num_episodes": len(lengths),
        "total_samples": total,
        "episode_length": {
            "min": int(length_values.min()),
            "max": int(length_values.max()),
            "mean": round(float(length_values.mean()), 3),
        },
        "action_dim": feature_dims["actions"][0],
        "obs_keys": list(REQUIRED_OBS_KEYS),
        "obs_shapes": {key: feature_dims[f"obs/{key}"] for key in REQUIRED_OBS_KEYS},
        "observed_ranges": {
            name: {"min": [round(float(v), 6) for v in low], "max": [round(float(v), 6) for v in high]}
            for name, (low, high) in ranges.items()
        },
        "environment": _environment_summary(env_args),
        "observation_source": "simulator state (low-dimensional); not derived from camera images",
        "upstream_masks": existing_masks,
        "warnings": warnings,
    }


def make_split(episode_ids: list[str], seed: int = 42) -> dict[str, list[str]]:
    """Deterministic whole-episode split into 160 train / 20 val / 20 test."""
    episode_ids = list(episode_ids)
    duplicates = sorted({e for e in episode_ids if episode_ids.count(e) > 1})
    if duplicates:
        raise ValueError(f"duplicate episode IDs: {duplicates[:5]}")
    if len(episode_ids) != EXPECTED_EPISODES:
        raise ValueError(
            f"expected {EXPECTED_EPISODES} episodes but found {len(episode_ids)}; "
            "resolve the dataset mismatch instead of choosing a different split"
        )
    ordered = sorted(episode_ids)
    order = np.random.default_rng(seed).permutation(len(ordered))
    shuffled = [ordered[i] for i in order]
    n_train, n_val = SPLIT_COUNTS["train"], SPLIT_COUNTS["val"]
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def prepare_dataset(
    source: Path,
    destination: Path,
    manifest_path: Path,
    audit_path: Path,
    overwrite: bool = False,
    seed: int = 42,
) -> None:
    """Audit `source`, then write a working copy with split masks plus the audit and manifest."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or (destination.exists() and os.path.samefile(source, destination)):
        raise ValueError(f"source and destination are the same file: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; pass overwrite=True (--overwrite) to replace it")

    checksum = sha256_file(source)
    audit = audit_dataset(source)
    with h5py.File(source, "r") as f:
        episode_ids = list(f["data"].keys())
    split = make_split(episode_ids, seed=seed)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o644)
    with h5py.File(destination, "r+") as f:
        masks = f.require_group("mask")
        for name, mask in MASK_NAMES.items():
            if mask in masks:
                del masks[mask]
            masks.create_dataset(mask, data=np.array(split[name], dtype="S"))

    if sha256_file(source) != checksum:
        raise RuntimeError(f"source checksum changed while preparing data: {source}")

    audit["split_masks"] = {MASK_NAMES[name]: len(split[name]) for name in MASK_NAMES}
    write_json(audit_path, audit)
    write_json(
        manifest_path,
        {
            "source_sha256": checksum,
            "split_seed": seed,
            "train": split["train"],
            "val": split["val"],
            "test": split["test"],
            "version_info": {
                "algorithm": SPLIT_ALGORITHM,
                "numpy": np.__version__,
                "h5py": h5py.__version__,
                "python": platform.python_version(),
                "source_file": source.name,
                "mask_names": MASK_NAMES,
            },
        },
    )


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=OUT / "data/processed/lift_clean.hdf5")
    parser.add_argument("--manifest", type=Path, default=OUT / "artifacts/split_manifest.json")
    parser.add_argument("--audit", type=Path, default=OUT / "artifacts/dataset_audit.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing working copy")
    args = parser.parse_args(argv)
    try:
        prepare_dataset(args.source, args.destination, args.manifest, args.audit, args.overwrite, args.seed)
    except (AuditError, ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    print(f"audited {audit['num_episodes']} episodes, {audit['total_samples']} samples")
    print(f"warnings: {len(audit['warnings'])}")
    print(f"working copy: {args.destination}")
    print(f"masks: {audit['split_masks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
