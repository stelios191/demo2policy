"""Held-out action scoring, fixed initial-state banks, checkpoint selection and closed-loop rollouts.

Policies and environments are always loaded through robomimic's checkpoint
utilities, so preprocessing and action interpretation are the same offline and
in the simulator.

    python -m scripts.evaluate state-bank --role development --run-dir runs/bc_lift_10k/<timestamp>
    python -m scripts.evaluate score-validation --run-dir runs/bc_lift_10k/<timestamp>
    python -m scripts.evaluate develop --run-dir runs/bc_lift_10k/<timestamp>
    python -m scripts.evaluate final
    python -m scripts.evaluate render --results-dir artifacts/3_follow_up --state-id state_30005 --video out.mp4

Held-out scoring always uses the clean working copy. Pass --tag <condition> before the
command to keep an alignment condition's outputs under artifacts/1_first_evaluation/timing_experiment/<condition>/.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

from scripts.common import ARTIFACTS, OUT, ROOT, TIMING_EXPERIMENT, relative_path, sha256_file, write_json
from scripts.prepare_data import REQUIRED_OBS_KEYS

ACTION_REPRESENTATION = (
    "dataset 'actions': normalized controller inputs as demonstrated; policy outputs are compared "
    "after robomimic's action unnormalization (identity in this project)"
)
RECORD_FIELDS = [
    "run_id",
    "state_id",
    "seed",
    "attempt",
    "outcome",
    "success",
    "steps",
    "first_success_step",
    "horizon",
    "checkpoint_sha256",
    "state_bank_sha256",
    "error",
]
SELECTION_EVIDENCE = {"offline_mask": "d2p_val", "state_bank_role": "development"}
STATE_TOLERANCE = 1e-8


def load_policy(checkpoint):
    """Load a policy through robomimic's checkpoint utilities."""
    import robomimic.utils.file_utils as FileUtils
    import torch

    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=str(checkpoint), device=torch.device("cpu"), verbose=False)
    return policy


def load_policy_and_env(checkpoint, render_offscreen=False):
    """Load a policy and the environment described in its checkpoint."""
    import robomimic.utils.file_utils as FileUtils
    import torch

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint), device=torch.device("cpu"), verbose=False
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, render=False, render_offscreen=render_offscreen, verbose=False
    )
    return policy, env


def episodes_for_mask(dataset, mask):
    """Episode IDs listed under a mask in the dataset."""
    with h5py.File(dataset, "r") as f:
        return [x.decode() for x in f["mask"][mask][:]]


# ---------------------------------------------------------------- offline scoring


def score_offline(checkpoint: Path, dataset: Path, episode_ids: list[str], load_policy=load_policy) -> dict:
    """Mean squared action error per episode, then averaged over episodes. Scores every sample."""
    episode_ids = list(episode_ids)
    if not episode_ids:
        raise ValueError("no episodes to score")
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("duplicate episode IDs in scoring request")

    policy = load_policy(checkpoint)
    episode_mse, episode_num_samples = {}, {}
    with h5py.File(dataset, "r") as f:
        for episode_id in episode_ids:
            if episode_id not in f["data"]:
                raise KeyError(f"episode {episode_id} is not in {Path(dataset).name}")
            episode = f["data"][episode_id]
            targets = episode["actions"][:]
            obs = {key: episode["obs"][key][:] for key in REQUIRED_OBS_KEYS}
            policy.start_episode()
            predictions = np.stack(
                [np.asarray(policy(ob={key: obs[key][t] for key in obs})) for t in range(len(targets))]
            )
            if not np.all(np.isfinite(predictions)):
                raise ValueError(f"policy produced non-finite actions for episode {episode_id}")
            episode_mse[episode_id] = float(np.mean((predictions - targets) ** 2))
            episode_num_samples[episode_id] = int(len(targets))

    return {
        "checkpoint": relative_path(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_sha256": sha256_file(dataset),
        "action_representation": ACTION_REPRESENTATION,
        "num_episodes": len(episode_ids),
        "num_samples": int(sum(episode_num_samples.values())),
        "episode_num_samples": episode_num_samples,
        "episode_mse": episode_mse,
        "mean_episode_mse": float(np.mean(list(episode_mse.values()))),
    }


# ---------------------------------------------------------------- selection


def select_checkpoint(candidates: list[dict]) -> dict:
    """Most development successes, then lowest validation MSE, then earliest update."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("no candidates to select from")
    for entry in candidates:
        forbidden = [key for key in entry if key.startswith(("final", "test"))]
        if forbidden:
            raise ValueError(f"candidate {entry.get('checkpoint')} carries final/test evidence: {forbidden}")
        if "evidence" not in entry:
            raise ValueError(f"candidate {entry.get('checkpoint')} has no evidence labels")
        if entry["evidence"] != SELECTION_EVIDENCE:
            raise ValueError(
                f"candidate {entry.get('checkpoint')} was scored on {entry['evidence']}; selection may only use "
                f"{SELECTION_EVIDENCE}, never final or test evidence"
            )
        if not math.isfinite(entry["validation_mean_episode_mse"]):
            raise ValueError(f"candidate {entry.get('checkpoint')} has a non-finite validation MSE")
    updates = [entry["update"] for entry in candidates]
    if len(set(updates)) != len(updates):
        raise ValueError(f"duplicate candidate updates: {sorted(updates)}")
    return min(
        candidates,
        key=lambda e: (-e["development_successes"], e["validation_mean_episode_mse"], e["update"]),
    )


# ---------------------------------------------------------------- state banks


def create_state_bank(checkpoint: Path, seeds: list[int], output: Path, role: str = "development") -> None:
    """Sample one initial state per seed and save everything needed to restore it."""
    import importlib.metadata as metadata

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; state banks are never regenerated in place")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds")
    _, env = load_policy_and_env(checkpoint)

    entries = []
    for seed in seeds:
        np.random.seed(seed)
        env.reset()
        state = env.get_state()
        obs = env.reset_to(state)
        entries.append((seed, state, {key: np.array(obs[key]) for key in REQUIRED_OBS_KEYS}))

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as f:
        f.attrs["role"] = role
        f.attrs["seeding"] = "numpy.random.seed(seed) immediately before env.reset()"
        f.attrs["env_metadata"] = json.dumps(env.serialize())
        f.attrs["versions"] = json.dumps(
            {pkg: metadata.version(pkg) for pkg in ("robomimic", "robosuite", "mujoco", "numpy")}
        )
        for seed, state, obs in entries:
            group = f.create_group(f"states/state_{seed}")
            group.attrs["seed"] = seed
            group.create_dataset("states", data=np.asarray(state["states"]))
            group.create_dataset("model", data=state["model"])
            for key, value in obs.items():
                group.create_dataset(f"obs/{key}", data=value)


def read_state_bank(path: Path) -> tuple[list[dict], str]:
    """Read saved initial states in seed order. Returns (entries, role)."""
    entries = []
    with h5py.File(path, "r") as f:
        role = str(f.attrs["role"])
        for state_id in sorted(f["states"], key=lambda s: int(f["states"][s].attrs["seed"])):
            group = f["states"][state_id]
            model = group["model"][()]
            entries.append(
                {
                    "state_id": state_id,
                    "seed": int(group.attrs["seed"]),
                    "states": group["states"][:],
                    "model": model.decode() if isinstance(model, bytes) else str(model),
                    "obs": {key: group["obs"][key][:] for key in group["obs"]} if "obs" in group else {},
                }
            )
    return entries, role


def verify_state_bank(checkpoint: Path, path: Path) -> dict:
    """Restore every saved state and confirm the simulator state and observations match."""
    _, env = load_policy_and_env(checkpoint)
    entries, role = read_state_bank(path)
    state_error = obs_error = 0.0
    for entry in entries:
        env.reset()
        obs = env.reset_to({"states": entry["states"], "model": entry["model"]})
        state_error = max(state_error, float(np.abs(env.get_state()["states"] - entry["states"]).max()))
        for key, saved in entry["obs"].items():
            obs_error = max(obs_error, float(np.abs(np.asarray(obs[key]) - saved).max()))
    result = {
        "state_bank": relative_path(path),
        "state_bank_sha256": sha256_file(path),
        "role": role,
        "num_states": len(entries),
        "max_state_error": state_error,
        "max_observation_error": obs_error,
        "tolerance": STATE_TOLERANCE,
    }
    if state_error > STATE_TOLERANCE or obs_error > STATE_TOLERANCE:
        raise RuntimeError(f"saved states do not restore consistently: {result}")
    return result


def min_distance_between(states_a, states_b) -> float:
    """Smallest Euclidean distance between any pair, ignoring the leading simulation-time entry."""
    a = np.stack([s[1:] for s in states_a])
    b = np.stack([s[1:] for s in states_b])
    return float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min())


def check_bank_is_distinct(path: Path, dataset: Path, other_banks: list[Path]) -> dict:
    """Confirm no bank state duplicates a demonstration's initial state or a state in another bank."""
    entries, _ = read_state_bank(path)
    bank_states = [e["states"] for e in entries]
    with h5py.File(dataset, "r") as f:
        demo_states = [f["data"][episode]["states"][0] for episode in f["data"]]
    result = {"min_distance_to_demonstration_initial_states": min_distance_between(bank_states, demo_states)}
    within = np.stack([s[1:] for s in bank_states])
    pairwise = np.sqrt(((within[:, None, :] - within[None, :, :]) ** 2).sum(-1))
    result["min_distance_within_bank"] = float(pairwise[~np.eye(len(within), dtype=bool)].min())
    for other in other_banks:
        other_states = [e["states"] for e in read_state_bank(other)[0]]
        result[f"min_distance_to_{relative_path(other)}"] = min_distance_between(bank_states, other_states)
    duplicates = {key: value for key, value in result.items() if value < 1e-9}
    if duplicates:
        raise RuntimeError(f"state bank {path} contains duplicated initial states: {duplicates}")
    return result


# ---------------------------------------------------------------- rollouts


def run_trials(
    policy,
    env,
    states,
    horizon,
    run_id,
    checkpoint_sha256,
    state_bank_sha256,
    previous_records=None,
    retry_state_ids=None,
):
    """Run one closed-loop trial per saved state. Every attempt yields a record, whatever happens.

    Adapted from the rollout loop in robomimic's scripts/run_trained_agent.py (MIT license), with
    saved initial states, per-trial records and explicit handling of invalid actions and errors.
    """
    by_id = {entry["state_id"]: entry for entry in states}
    if retry_state_ids is None:
        if previous_records:
            raise ValueError("previous records were given without retry_state_ids")
        records, to_run = [], list(states)
    else:
        unknown = [s for s in retry_state_ids if s not in by_id]
        if unknown:
            raise ValueError(f"cannot retry states that are not in the state bank: {unknown}")
        records = list(previous_records or [])
        to_run = [by_id[s] for s in retry_state_ids]

    trajectories = {}
    for entry in to_run:
        attempt = 1 + max((r["attempt"] for r in records if r["state_id"] == entry["state_id"]), default=0)
        record = {
            "run_id": run_id,
            "state_id": entry["state_id"],
            "seed": entry["seed"],
            "attempt": attempt,
            "outcome": None,
            "success": False,
            "steps": 0,
            "first_success_step": None,
            "horizon": horizon,
            "checkpoint_sha256": checkpoint_sha256,
            "state_bank_sha256": state_bank_sha256,
            "error": "",
        }
        actions, visited = [], [np.asarray(entry["states"])]
        try:
            policy.start_episode()
            env.reset()
            obs = env.reset_to({"states": entry["states"], "model": entry["model"]})
            record["outcome"] = "horizon_reached"
            for step in range(1, horizon + 1):
                action = np.asarray(policy(ob=obs))
                if not np.all(np.isfinite(action)):
                    record["outcome"] = "policy_failure"
                    record["error"] = f"non-finite action at step {step}: {action.tolist()}"
                    break
                obs, _, done, _ = env.step(action)
                record["steps"] = step
                actions.append(action)
                visited.append(np.asarray(env.get_state()["states"]))
                if env.is_success()["task"]:
                    record.update(outcome="success", success=True, first_success_step=step)
                    break
                if done:
                    record["outcome"] = "terminated"
                    break
        except Exception as exc:  # any failure here is infrastructure, and must stay visible
            record.update(outcome="infrastructure_error", success=None, error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        trajectories[(entry["state_id"], attempt)] = {
            "actions": np.stack(actions) if actions else np.zeros((0, 0)),
            "states": np.stack(visited),
            "model": entry["model"],
        }
    return records, trajectories


def latest_attempts(records) -> dict:
    """The latest attempt for each state, by state id."""
    latest = {}
    for record in records:
        if record["state_id"] not in latest or record["attempt"] > latest[record["state_id"]]["attempt"]:
            latest[record["state_id"]] = record
    return latest


def summarize_trials(records, planned: int) -> dict:
    """Counts use the latest attempt for each state; every attempt stays in the records."""
    latest = latest_attempts(records)
    outcomes = [record["outcome"] for record in latest.values()]
    completed = sum(outcome != "infrastructure_error" for outcome in outcomes)
    return {
        "planned": planned,
        "attempted": len(latest),
        "completed": completed,
        "successes": outcomes.count("success"),
        "policy_failures": outcomes.count("policy_failure"),
        "infrastructure_errors": outcomes.count("infrastructure_error"),
        "incomplete": completed != planned,
        "attempt_records": len(records),
    }


def write_records(path, records):
    """Write per-trial records as CSV, leaving absent values empty."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({k: ("" if record[k] is None else record[k]) for k in RECORD_FIELDS})


def read_records(path):
    """Read per-trial records back with their original types."""

    def optional_int(value):
        return None if value == "" else int(value)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["seed"], row["attempt"] = int(row["seed"]), int(row["attempt"])
        row["steps"], row["horizon"] = int(row["steps"]), int(row["horizon"])
        row["first_success_step"] = optional_int(row["first_success_step"])
        row["success"] = None if row["success"] == "" else row["success"] == "True"
    return rows


def evaluate_rollouts(
    checkpoint: Path,
    states_path: Path,
    horizon: int,
    output_dir: Path,
    load=load_policy_and_env,
    retry_state_ids=None,
) -> dict:
    """Run the policy from every saved state and write rollouts.csv, results.json and trajectories.hdf5."""
    output_dir = Path(output_dir)
    records_path = output_dir / "rollouts.csv"
    previous = None
    if retry_state_ids is None:
        if records_path.exists():
            raise FileExistsError(f"{records_path} exists; results are not overwritten (use a retry for failed states)")
    else:
        previous = read_records(records_path)

    states, role = read_state_bank(states_path)
    checkpoint_sha, bank_sha = sha256_file(checkpoint), sha256_file(states_path)
    run_id = f"{role}-{checkpoint_sha[:8]}-{bank_sha[:8]}"
    policy, env = load(checkpoint)
    records, trajectories = run_trials(
        policy, env, states, horizon, run_id, checkpoint_sha, bank_sha, previous, retry_state_ids
    )
    summary = summarize_trials(records, planned=len(states))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_records(records_path, records)
    with h5py.File(output_dir / "trajectories.hdf5", "a") as f:
        for (state_id, attempt), trajectory in trajectories.items():
            group = f.create_group(f"{state_id}/attempt_{attempt}")
            group.create_dataset("actions", data=trajectory["actions"])
            group.create_dataset("states", data=trajectory["states"])
            group.create_dataset("model", data=trajectory["model"])
    write_json(
        output_dir / "results.json",
        {
            "run_id": run_id,
            "checkpoint": relative_path(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "state_bank": relative_path(states_path),
            "state_bank_sha256": bank_sha,
            "state_bank_role": role,
            "horizon": horizon,
            "success_criterion": "env.is_success()['task']",
            "summary": summary,
            "success_rate_of_completed": (summary["successes"] / summary["completed"])
            if summary["completed"]
            else None,
        },
    )
    return summary


# ---------------------------------------------------------------- rendering


def render_trial(
    checkpoint: Path,
    results_dir: Path,
    state_id: str,
    video_path: Path,
    attempt: int = 1,
    caption: str | None = None,
    camera: str = "sideview",
) -> dict:
    """Re-render a recorded learned-policy trajectory by restoring its saved simulator states.

    Each frame carries one plain caption; which checkpoint and state it shows goes in the
    returned record, not on the frames.
    """
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    results_dir = Path(results_dir)
    results = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    if sha256_file(checkpoint) != results["checkpoint_sha256"]:
        raise ValueError("checkpoint does not match the one that produced these results")
    record = next(
        r for r in read_records(results_dir / "rollouts.csv") if r["state_id"] == state_id and r["attempt"] == attempt
    )
    with h5py.File(results_dir / "trajectories.hdf5", "r") as f:
        group = f[f"{state_id}/attempt_{attempt}"]
        visited, model = group["states"][:], group["model"][()].decode()

    _, env = load_policy_and_env(checkpoint, render_offscreen=True)
    env.reset()
    env.reset_to({"states": visited[0], "model": model})
    if caption is None:
        seconds = record["steps"] / 20
        caption = (
            f"cube lifted after {seconds:.1f} s"
            if record["success"]
            else f"cube not lifted in {seconds:.0f} s ({record['outcome'].replace('_', ' ')})"
        )
    font = ImageFont.load_default(size=22)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(video_path, fps=20)
    for step, state in enumerate(visited):
        env.reset_to({"states": state})
        frame = env.render(mode="rgb_array", height=512, width=512, camera_name=camera)
        image = Image.fromarray(np.ascontiguousarray(frame))
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 470, 512, 512], fill=(20, 20, 20))
        draw.text((12, 479), caption, fill=(255, 255, 255), font=font)
        draw.text((500, 479), f"{step / 20:4.1f} s", fill=(190, 190, 190), font=font, anchor="ra")
        writer.append_data(np.asarray(image))
    writer.close()
    return {
        "video": relative_path(video_path),
        "content": "learned-policy rollout in simulation, re-rendered from recorded simulator states",
        "task": "robosuite Lift",
        "checkpoint_sha256": results["checkpoint_sha256"],
        "run_id": results["run_id"],
        "state_id": state_id,
        "attempt": attempt,
        "outcome": record["outcome"],
        "steps": record["steps"],
        "frames": int(len(visited)),
        "camera": camera,
        "caption": caption,
    }


# ---------------------------------------------------------------- command line


def load_protocol(path="configs/protocol_first_evaluation.json"):
    """Read an evaluation protocol, given relative to the repository root."""
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def artifact_dir(tag, protocol=None):
    """Outputs live in the protocol's artifact directory; timing-experiment conditions get their own folder."""
    if tag:
        return ARTIFACTS / TIMING_EXPERIMENT / tag
    return OUT / protocol["artifacts"]


def scheduled_checkpoints(run_dir):
    """Yield (update, path) for each checkpoint of a run, verifying its recorded hash."""
    record = json.loads((Path(run_dir) / "run_record.json").read_text(encoding="utf-8"))
    for entry in record["checkpoints"]:
        path = Path(run_dir) / "models" / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"{path} does not match the hash recorded at training time")
        yield entry["update"], path


def command_state_bank(args):
    """Create a state bank, confirm it restores exactly, and check it duplicates no earlier state."""
    protocol = load_protocol(args.protocol)
    section = protocol[args.role]
    output = OUT / section["state_bank"]
    checkpoint = next(iter(scheduled_checkpoints(args.run_dir)))[1]
    create_state_bank(checkpoint, section["seeds"], output, role=args.role)
    if "distinct_from" in section:
        other = [OUT / p for p in section["distinct_from"]]
        missing = [str(p) for p in other if not p.exists()]
        if missing:
            raise FileNotFoundError(f"cannot check distinctness, state banks are absent: {missing}")
    else:
        other = [OUT / protocol[r]["state_bank"] for r in ("development", "final") if r != args.role]
        other = [p for p in other if p.exists()]
    report = verify_state_bank(checkpoint, output)
    report["distinctness"] = check_bank_is_distinct(output, OUT / args.dataset, other)
    write_json(artifact_dir("", protocol) / f"state_bank_{args.role}.json", report)
    print(json.dumps(report, indent=2))


def command_score_validation(args):
    """Score every scheduled checkpoint on the validation episodes."""
    protocol = load_protocol(args.protocol)
    dataset = OUT / args.dataset
    episodes = episodes_for_mask(dataset, protocol["training"]["validation_mask"])
    scores = []
    for update, path in scheduled_checkpoints(args.run_dir):
        result = score_offline(path, dataset, episodes)
        result.update(update=update, mask=protocol["training"]["validation_mask"])
        scores.append(result)
        print(f"update {update}: validation mean episode MSE {result['mean_episode_mse']:.6f}")
    write_json(
        artifact_dir(args.tag, protocol) / "validation_scores.json",
        {"run_dir": relative_path(args.run_dir), "scores": scores},
    )


def command_develop(args):
    """Run every scheduled checkpoint on the development states and select one."""
    protocol = load_protocol(args.protocol)
    bank = OUT / protocol["development"]["state_bank"]
    validation = json.loads((artifact_dir(args.tag, protocol) / "validation_scores.json").read_text(encoding="utf-8"))
    by_update = {score["update"]: score for score in validation["scores"]}
    candidates = []
    for update, path in scheduled_checkpoints(args.run_dir):
        out = artifact_dir(args.tag, protocol) / "development" / f"update_{update}"
        summary = evaluate_rollouts(path, bank, protocol["rollout"]["horizon"], out)
        if summary["incomplete"]:
            raise RuntimeError(f"development evaluation incomplete for update {update}: {summary}")
        score = by_update[update]
        if score["checkpoint_sha256"] != sha256_file(path):
            raise RuntimeError(f"validation score for update {update} belongs to a different checkpoint")
        candidates.append(
            {
                "checkpoint": relative_path(path),
                "checkpoint_sha256": score["checkpoint_sha256"],
                "update": update,
                "development_successes": summary["successes"],
                "development_completed": summary["completed"],
                "validation_mean_episode_mse": score["mean_episode_mse"],
                "evidence": {"offline_mask": score["mask"], "state_bank_role": read_state_bank(bank)[1]},
            }
        )
        print(f"update {update}: development successes {summary['successes']}/{summary['completed']}")
    selected = select_checkpoint(candidates)
    write_json(
        artifact_dir(args.tag, protocol) / "selection.json",
        {
            "rule": protocol["selection"]["rule"],
            "candidates": candidates,
            "selected": selected,
            "development_state_bank_sha256": sha256_file(bank),
        },
    )
    print(f"selected update {selected['update']} ({selected['checkpoint_sha256'][:12]})")


def command_final(args):
    """Evaluate the selected checkpoint once on the test episodes and the final states."""
    protocol = load_protocol(args.protocol)
    selection_path = OUT / args.selection if args.selection else artifact_dir(args.tag, protocol) / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected"]
    checkpoint = OUT / selected["checkpoint"]
    if sha256_file(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint has changed since selection")
    dataset = OUT / args.dataset
    bank = OUT / protocol["final"]["state_bank"]
    out = artifact_dir(args.tag, protocol) / "final" if args.tag else artifact_dir(args.tag, protocol)
    if args.name:
        out = artifact_dir(args.tag, protocol) / args.name
    offline = score_offline(checkpoint, dataset, episodes_for_mask(dataset, protocol["final"]["test_mask"]))
    offline["mask"] = protocol["final"]["test_mask"]
    summary = evaluate_rollouts(checkpoint, bank, protocol["rollout"]["horizon"], out, retry_state_ids=args.retry)
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    results["offline_test"] = offline
    results["selection_sha256"] = sha256_file(selection_path)
    results["update"] = selected["update"]
    write_json(out / "results.json", results)
    print(f"test mean episode MSE: {offline['mean_episode_mse']:.6f}")
    print(f"final rollouts: {summary}")


def command_render(args):
    """Re-render one recorded trial to video."""
    results = json.loads((Path(args.results_dir) / "results.json").read_text(encoding="utf-8"))
    info = render_trial(OUT / results["checkpoint"], args.results_dir, args.state_id, args.video, args.attempt)
    print(json.dumps(info, indent=2))


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data/processed/lift_clean.hdf5")
    parser.add_argument("--tag", default="", help="alignment condition name; empty for the baseline")
    parser.add_argument(
        "--protocol", default="configs/protocol_first_evaluation.json", help="evaluation protocol to follow"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    bank = commands.add_parser("state-bank")
    bank.add_argument("--role", choices=["development", "final"], required=True)
    bank.add_argument("--run-dir", type=Path, required=True)
    bank.set_defaults(func=command_state_bank)

    for name, func in (("score-validation", command_score_validation), ("develop", command_develop)):
        sub = commands.add_parser(name)
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.set_defaults(func=func)

    final = commands.add_parser("final")
    final.add_argument("--retry", nargs="+", default=None, metavar="STATE_ID")
    final.add_argument("--selection", default=None, help="evaluate the checkpoint chosen in this selection file")
    final.add_argument("--name", default=None, help="write results to a named subdirectory")
    final.set_defaults(func=command_final)

    render = commands.add_parser("render")
    render.add_argument("--results-dir", type=Path, required=True)
    render.add_argument("--state-id", required=True)
    render.add_argument("--attempt", type=int, default=1)
    render.add_argument("--video", type=Path, required=True)
    render.set_defaults(func=command_render)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
