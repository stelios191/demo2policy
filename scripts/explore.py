"""Exploratory evaluation on the exploration state bank. Results here are for diagnosis only.

The exploration bank (seeds 11000-11099) is separate from the development and final
banks. Anything chosen using these results must be confirmed on initial states that
have never been evaluated.

    python -m scripts.explore --run-dir runs/explore_mlp256_200k/<timestamp>
    python -m scripts.explore --run-dir runs/bc_lift_10k/<timestamp> --binary-gripper
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from scripts.common import ARTIFACTS, DIAGNOSIS, write_json
from scripts.evaluate import (
    evaluate_rollouts,
    latest_attempts,
    load_policy_and_env,
    read_records,
    scheduled_checkpoints,
)

BANK = ARTIFACTS / DIAGNOSIS / "initial_states_explore.hdf5"
# The same seeds later served as the follow-up's development states (configs/protocol_follow_up.json).
EXPLORATION_SEEDS = list(range(11000, 11100))
GRIPPER = 6


class BinaryGripperPolicy:
    """Wraps a policy and snaps the gripper command to -1 or +1, as in the demonstrations."""

    def __init__(self, policy):
        self.policy = policy

    def start_episode(self):
        self.policy.start_episode()

    def __call__(self, ob):
        action = np.array(self.policy(ob=ob), dtype=float)
        action[GRIPPER] = 1.0 if action[GRIPPER] > 0 else -1.0
        return action


def offset_at_close(results_dir, env):
    """Horizontal gripper-to-cube distance at the first close command, per trial outcome.

    Trials that never command a close are left out; scripts/analyze.py grasp-diagnosis counts them.
    """
    offsets = {True: [], False: []}
    records = latest_attempts(read_records(results_dir / "rollouts.csv")).values()
    with h5py.File(results_dir / "trajectories.hdf5", "r") as f:
        for record in records:
            if record["outcome"] == "infrastructure_error":
                continue
            group = f[f"{record['state_id']}/attempt_{record['attempt']}"]
            closes = np.where(group["actions"][:, GRIPPER] > 0)[0]
            if not len(closes):
                continue
            states = group["states"][:]
            env.reset()
            env.reset_to({"states": states[0], "model": group["model"][()].decode()})
            relative = env.reset_to({"states": states[closes[0]]})["object"][7:10]
            offsets[bool(record["success"])].append(float(np.hypot(relative[0], relative[1])))
    return offsets


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--binary-gripper", action="store_true", help="snap gripper commands to -1 or +1")
    parser.add_argument("--updates", type=int, nargs="+", default=None, help="only these checkpoints")
    args = parser.parse_args(argv)

    if not BANK.exists():
        parser.error(f"{BANK} is missing; create it with: python -m scripts.analyze exploration-bank")
    experiment = json.loads((args.run_dir / "run_record.json").read_text(encoding="utf-8"))["experiment"]
    label = experiment + ("_binary_gripper" if args.binary_gripper else "")

    def load(checkpoint):
        policy, env = load_policy_and_env(checkpoint)
        return (BinaryGripperPolicy(policy) if args.binary_gripper else policy), env

    rows = []
    for update, checkpoint in scheduled_checkpoints(args.run_dir):
        if args.updates and update not in args.updates:
            continue
        out = ARTIFACTS / DIAGNOSIS / label / f"update_{update}"
        if not (out / "rollouts.csv").exists():
            evaluate_rollouts(checkpoint, BANK, 400, out, load=load)
        summary = json.loads((out / "results.json").read_text(encoding="utf-8"))["summary"]
        offsets = offset_at_close(out, load_policy_and_env(checkpoint)[1])
        every = offsets[True] + offsets[False]
        rows.append(
            {
                "experiment": label,
                "update": update,
                "successes": summary["successes"],
                "completed": summary["completed"],
                "incomplete": summary["incomplete"],
                "median_offset_mm": round(1000 * float(np.median(every)), 1) if every else None,
                "p90_offset_mm": round(1000 * float(np.percentile(every, 90)), 1) if every else None,
            }
        )
        print(rows[-1], flush=True)
    path = ARTIFACTS / DIAGNOSIS / label / "summary.json"
    write_json(path, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
