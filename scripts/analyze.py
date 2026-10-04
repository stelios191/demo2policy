"""Analyses behind specific numbers in docs/results.md, and the videos and GIFs in the README.

Each command reads recorded trials and writes one small JSON record under artifacts/.

    python -m scripts.analyze exploration-bank        # the 100 exploration starting positions
    python -m scripts.analyze exploration-baseline    # the first model on those positions
    python -m scripts.analyze grasp-diagnosis         # gripper position at the close command, against the human
    python -m scripts.analyze failures                # how the first evaluation's failures ended
    python -m scripts.analyze follow-up-aiming        # aiming error on the follow-up final states
    python -m scripts.analyze validation-vs-success   # needs the long exploration runs (reproduce.sh --full)
    python -m scripts.analyze media                   # the two videos, their index, and the README GIFs
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from scripts.common import ARTIFACTS, DIAGNOSIS, FIRST_EVALUATION, FOLLOW_UP, OUT, write_json
from scripts.evaluate import (
    check_bank_is_distinct,
    create_state_bank,
    episodes_for_mask,
    evaluate_rollouts,
    latest_attempts,
    load_policy_and_env,
    read_records,
    render_trial,
    scheduled_checkpoints,
    score_offline,
    verify_state_bank,
)
from scripts.explore import BANK, EXPLORATION_SEEDS, GRIPPER

FIRST = ARTIFACTS / FIRST_EVALUATION
DATASET = OUT / "data" / "processed" / "lift_clean.hdf5"
EXPLORATION = ARTIFACTS / DIAGNOSIS
DEMONSTRATIONS_SAMPLED = 40


def first_model_checkpoint():
    """The checkpoint chosen in the first evaluation."""
    selection = json.loads((FIRST / "selection.json").read_text(encoding="utf-8"))
    return OUT / selection["selected"]["checkpoint"]


def millimetres(values):
    """Median and range of distances in metres, reported in millimetres; None when there are none."""
    if not len(values):
        return None
    values = np.asarray(values) * 1000
    return {
        "median": round(float(np.median(values)), 1),
        "min": round(float(values.min()), 1),
        "max": round(float(values.max()), 1),
    }


def median(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(float(np.median(values)), 1) if values else None


# ---------------------------------------------------------------- exploration states and baseline


def command_exploration_bank(_args):
    """Create the exploration starting positions if needed, and check them."""
    checkpoint = first_model_checkpoint()
    if not BANK.exists():
        create_state_bank(checkpoint, EXPLORATION_SEEDS, BANK, role="exploration")
    report = verify_state_bank(checkpoint, BANK)
    others = [FIRST / "initial_states_dev.hdf5", FIRST / "initial_states_final.hdf5"]
    report["distinctness"] = check_bank_is_distinct(BANK, DATASET, others)
    write_json(EXPLORATION / "state_bank_exploration.json", report)
    print(f"exploration states: {report['num_states']}, restore error {report['max_state_error']}")


def command_exploration_baseline(_args):
    """Run the first model from every exploration starting position."""
    out = EXPLORATION / "baseline"
    if (out / "rollouts.csv").exists():
        print(f"{out} already evaluated")
        return
    summary = evaluate_rollouts(first_model_checkpoint(), BANK, 400, out)
    print(f"first model on exploration states: {summary['successes']} / {summary['completed']}")


# ---------------------------------------------------------------- grasp diagnosis


def grasp_geometry(env, states, actions, model, window=120):
    """Where the gripper was, relative to the cube, when it tried to grasp.

    The grasp point is the first close command. A trial that never commands a close has no
    close point; for it, the grasp point is its closest approach to the cube. Only the first
    `window` steps are replayed: every grasp, by the human or the policy, falls well inside them.
    """
    env.reset()
    env.reset_to({"states": states[0], "model": model})
    obs = [env.reset_to({"states": s}) for s in states[: min(len(states), window)]]
    closes = np.where(actions[:, GRIPPER] > 0)[0]
    t_close = int(closes[0]) if len(closes) and closes[0] < len(obs) else None
    relative = np.array([o["object"][7:10] for o in obs])  # gripper-to-cube vector
    t_grasp = t_close if t_close is not None else int(np.linalg.norm(relative, axis=1).argmin())
    quaternion = obs[0]["object"][3:7]
    yaw = np.degrees(2 * np.arctan2(quaternion[2], quaternion[3])) % 90
    return {
        "closed": t_close is not None,
        "t_close": t_close,
        "t_grasp": t_grasp,
        "xy_off": float(np.hypot(*relative[t_grasp][:2])),
        "z_off": float(relative[t_grasp][2]),
        "yaw": float(min(yaw, 90 - yaw)),
    }


def trial_groups(results_dir):
    """Each state's latest trial record and its stored trajectory, skipping infrastructure errors."""
    with h5py.File(results_dir / "trajectories.hdf5", "r") as f:
        for record in latest_attempts(read_records(results_dir / "rollouts.csv")).values():
            if record["outcome"] != "infrastructure_error":
                yield record, f[f"{record['state_id']}/attempt_{record['attempt']}"]


def command_grasp_diagnosis(_args):
    """Gripper position at the grasp: first model's successes and failures, and the human."""
    _, env = load_policy_and_env(first_model_checkpoint())
    successes, failures = [], []
    for record, group in trial_groups(EXPLORATION / "baseline"):
        row = grasp_geometry(env, group["states"][:], group["actions"][:], group["model"][()].decode())
        (successes if record["success"] else failures).append({"state_id": record["state_id"], **row})
    demos = []
    with h5py.File(DATASET, "r") as f:
        for episode_id in [x.decode() for x in f["mask/d2p_train"][:DEMONSTRATIONS_SAMPLED]]:
            episode = f["data"][episode_id]
            demos.append(grasp_geometry(env, episode["states"][:], episode["actions"][:], episode.attrs["model_file"]))

    summary = {}
    for name, rows in (("demonstrations", demos), ("successes", successes), ("failures", failures)):
        closed = [r for r in rows if r["closed"]]
        summary[name] = {
            "count": len(rows),
            "closed": len(closed),
            "never_closed": len(rows) - len(closed),
            "sideways_offset_at_grasp_mm": millimetres([r["xy_off"] for r in rows]),
            "sideways_offset_at_close_mm": millimetres([r["xy_off"] for r in closed]),
            "sideways_offset_never_closed_mm": millimetres([r["xy_off"] for r in rows if not r["closed"]]),
            "median_grasp_step": median(rows, "t_grasp"),
            "median_cube_yaw_degrees": median(rows, "yaw"),
        }
    write_json(
        EXPLORATION / "grasp_diagnosis.json",
        {"summary": summary, "successes": successes, "failures": failures, "demos": demos},
    )
    for name, s in summary.items():
        print(f"{name:15s} closed {s['closed']}/{s['count']}, offset at grasp {s['sideways_offset_at_grasp_mm']} mm")


# ---------------------------------------------------------------- first-evaluation failures


def command_failures(_args):
    """How each failed trial of the first evaluation ended, from its recorded simulator states."""
    _, env = load_policy_and_env(first_model_checkpoint())
    records = read_records(FIRST / "rollouts.csv")
    with h5py.File(DATASET, "r") as f:
        highest_demonstrated = max(float(f["data"][e]["obs/robot0_eef_pos"][:, 2].max()) for e in f["data"])
    failures = []
    for record, group in trial_groups(FIRST):
        if record["success"]:
            continue
        states = group["states"][:]
        env.reset()
        env.reset_to({"states": states[0], "model": group["model"][()].decode()})
        obs = [env.reset_to({"states": s}) for s in states]
        cube_height = np.array([o["object"][2] for o in obs])
        gripper_height = np.array([o["robot0_eef_pos"][2] for o in obs])
        distance = np.array([np.linalg.norm(o["object"][7:10]) for o in obs])
        closes = np.where(group["actions"][:, GRIPPER] > 0)[0]
        failures.append(
            {
                "state_id": record["state_id"],
                "steps": record["steps"],
                "closest_approach_first_100_steps_m": round(float(distance[:100].min()), 4),
                "first_close_command_step": int(closes[0]) if len(closes) else None,
                "max_cube_rise_m": round(float(cube_height.max() - cube_height[0]), 4),
                "final_gripper_height_m": round(float(gripper_height[-1]), 3),
                "final_distance_to_cube_m": round(float(distance[-1]), 3),
                "left_demonstrated_range": bool(gripper_height.max() > highest_demonstrated),
            }
        )
    success_steps = [r["steps"] for r in records if r["success"]]
    result = {
        "failures": failures,
        "highest_gripper_position_in_demonstrations_m": round(highest_demonstrated, 3),
        "successful_trial_steps": {
            "min": min(success_steps),
            "median": float(np.median(success_steps)),
            "max": max(success_steps),
        },
    }
    write_json(FIRST / "failure_analysis.json", result)
    for row in failures:
        print(row)


# ---------------------------------------------------------------- follow-up aiming error


def command_follow_up_aiming(_args):
    """Aiming error at the grasp for both models on the follow-up final states."""
    _, env = load_policy_and_env(first_model_checkpoint())
    result = {}
    for name, directory in (
        ("longer_training", ARTIFACTS / FOLLOW_UP),
        ("first_model_same_states", ARTIFACTS / FOLLOW_UP / "reference_10k"),
    ):
        rows = [
            (record, grasp_geometry(env, g["states"][:], g["actions"][:], g["model"][()].decode()))
            for record, g in trial_groups(directory)
        ]
        every = [row["xy_off"] for _, row in rows]
        result[name] = {
            "median_mm": round(1000 * float(np.median(every)), 1),
            "p90_mm": round(1000 * float(np.percentile(every, 90)), 1),
            "failed_trials": [
                {"state_id": r["state_id"], "closed": row["closed"], "offset_mm": round(1000 * row["xy_off"], 1)}
                for r, row in rows
                if not r["success"]
            ],
        }
    write_json(ARTIFACTS / FOLLOW_UP / "aiming_error.json", result)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------- validation error against success


def command_validation_vs_success(_args):
    """Held-out action error and task success for every checkpoint of the two long exploration runs."""
    validation = episodes_for_mask(DATASET, "d2p_val")
    result = {}
    for experiment in ("explore_mlp256_200k", "explore_mlp1024_200k"):
        runs = sorted((OUT / "runs" / experiment).glob("*/run_record.json"))
        if not runs:
            sys.exit(f"{experiment} has not been trained; run scripts/reproduce.sh --full")
        summary = json.loads((EXPLORATION / experiment / "summary.json").read_text(encoding="utf-8"))
        successes = {row["update"]: row["successes"] for row in summary}
        result[experiment] = []
        for update, checkpoint in scheduled_checkpoints(runs[-1].parent):
            error = score_offline(checkpoint, DATASET, validation)["mean_episode_mse"]
            result[experiment].append(
                {
                    "update": update,
                    "validation_mean_episode_mse": round(error, 6),
                    "successes_of_100": successes[update],
                }
            )
    write_json(EXPLORATION / "validation_vs_success.json", result)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------- videos and GIFs


def to_gif(video, gif, max_frames, hold=8, size=320, step=2):
    """A small looping GIF from every second frame, pausing on the last one."""
    import imageio.v2 as imageio
    from PIL import Image

    frames = [frame for frame in imageio.get_reader(video)][:max_frames:step]  # noqa: C416 (the reader has no length)
    frames += [frames[-1]] * hold
    images = [Image.fromarray(frame).resize((size, size), Image.LANCZOS) for frame in frames]
    # one palette for all frames, built from several of them so small objects keep their colour
    samples = images[:: max(1, len(images) // 6)]
    mosaic = Image.new("RGB", (size * len(samples), size))
    for i, image in enumerate(samples):
        mosaic.paste(image, (size * i, 0))
    palette = mosaic.quantize(colors=160, method=Image.FASTOCTREE)
    images = [image.quantize(palette=palette, dither=Image.NONE) for image in images]
    Path(gif).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(gif, save_all=True, append_images=images[1:], duration=100, loop=0, optimize=True)


def command_media(_args):
    """Render both follow-up models from the first state where the first model failed, and make the GIFs."""
    longer, reference = ARTIFACTS / FOLLOW_UP, ARTIFACTS / FOLLOW_UP / "reference_10k"
    failed = [r["state_id"] for r in read_records(reference / "rollouts.csv") if not r["success"]]
    if not failed:
        sys.exit("the first model failed on no follow-up state; there is no contrasting pair to show")
    state_id = failed[0]
    videos = []
    for name, directory, trained in (
        ("longer_training", longer, "50,000 updates"),
        ("first_model_same_states", reference, "10,000 updates"),
    ):
        results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        record = next(r for r in read_records(directory / "rollouts.csv") if r["state_id"] == state_id)
        label = "success" if record["success"] else "failure"
        caption = f"{trained}: " + ("lifts the cube" if record["success"] else "misses the cube")
        video = ARTIFACTS / "videos" / f"{name}_{state_id}_{label}.mp4"
        info = render_trial(OUT / results["checkpoint"], directory, state_id, video, caption=caption)
        info["policy"] = name
        videos.append(info)
    write_json(
        ARTIFACTS / "videos" / "index.json",
        {
            "content": "learned-policy rollouts in simulation, re-rendered from recorded simulator states",
            "selection_rule": "the first follow-up final state, in state order, where the first model failed; "
            "both models are shown from that same state",
            "videos": videos,
        },
    )
    to_gif(OUT / videos[0]["video"], OUT / "docs" / "media" / "final_model_success.gif", 200)
    to_gif(OUT / videos[1]["video"], OUT / "docs" / "media" / "first_model_failure.gif", 160)
    print(f"videos and GIFs for {state_id}")


COMMANDS = {
    "exploration-bank": command_exploration_bank,
    "exploration-baseline": command_exploration_baseline,
    "grasp-diagnosis": command_grasp_diagnosis,
    "failures": command_failures,
    "follow-up-aiming": command_follow_up_aiming,
    "validation-vs-success": command_validation_vs_success,
    "media": command_media,
}


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(COMMANDS))
    args = parser.parse_args(argv)
    COMMANDS[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
