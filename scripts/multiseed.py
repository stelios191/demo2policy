"""Repeat each condition over several training seeds and compare them on one set of initial states.

Every model is evaluated at its last checkpoint, so nothing is selected after training.
All models start from the same saved initial states.

    python -m scripts.multiseed run                 # train and evaluate everything that is missing
    python -m scripts.multiseed run-one --condition bc_50k --seed 3
    python -m scripts.multiseed summarize           # artifacts/4_multi_seed/summary.json
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from scripts.common import ARTIFACTS, FIRST_EVALUATION, OUT, ROOT, sha256_file, write_json
from scripts.evaluate import (
    check_bank_is_distinct,
    create_state_bank,
    episodes_for_mask,
    evaluate_rollouts,
    latest_attempts,
    read_records,
    score_offline,
    summarize_trials,
    verify_state_bank,
)

PROTOCOL_PATH = "configs/protocol_multi_seed.json"
DATASET = "data/processed/lift_clean.hdf5"
COUNT_FIELDS = ["attempted", "completed", "successes", "policy_failures", "infrastructure_errors"]


class MultiSeedError(Exception):
    """Recorded multi-seed results are absent or inconsistent."""


def load_protocol():
    """Read the multi-seed protocol."""
    return json.loads((ROOT / PROTOCOL_PATH).read_text(encoding="utf-8"))


def sign_test(better, worse):
    """Exact two-sided sign test on the seeds where the two conditions differ."""
    n = better + worse
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(better, worse) + 1)) / 2**n)


def load_run(results_dir, condition, seed, expected_update):
    """Load one model's results and check them against the raw records. Returns None if absent."""
    out = Path(results_dir) / condition / f"seed_{seed}"
    if not (out / "results.json").exists():
        return None
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    where = f"{condition}, seed {seed}"
    if results["update"] != expected_update:
        raise MultiSeedError(f"{where}: evaluated update {results['update']} but the protocol says {expected_update}")
    if results["seed"] != seed or results["condition"] != condition:
        raise MultiSeedError(f"{where}: results are labelled {results['condition']}, seed {results['seed']}")
    stored = results["summary"]
    records = read_records(out / "rollouts.csv")
    recomputed = summarize_trials(records, planned=stored["planned"])
    for field in COUNT_FIELDS:
        if stored[field] != recomputed[field]:
            raise MultiSeedError(
                f"{where}: stored summary says {field}={stored[field]} but the raw records give {recomputed[field]}"
            )
    offline = results.get("offline_test")
    return {
        "successes": recomputed["successes"],
        "completed": recomputed["completed"],
        "planned": stored["planned"],
        "incomplete": recomputed["incomplete"],
        "test_mean_episode_mse": offline["mean_episode_mse"] if offline else None,
        "checkpoint_sha256": results["checkpoint_sha256"],
        "state_bank_sha256": results["state_bank_sha256"],
        "outcomes": {
            state: record["success"]
            for state, record in latest_attempts(records).items()
            if record["outcome"] != "infrastructure_error"
        },
    }


def outcome_table(runs):
    """Runs x states array of trial successes (1.0 or 0.0), on the states every run completed."""
    states = sorted(set.intersection(*(set(run["outcomes"]) for run in runs)))
    return np.array([[run["outcomes"][s] for s in states] for run in runs], dtype=float), states


def bootstrap_interval(table, baseline=None, repeats=10000, rng_seed=0):
    """95% interval for the pooled success rate, or for its difference from a seed-paired baseline.

    Every model is evaluated on the same initial states, so trials sharing a state are not
    independent, and neither are trials sharing a model. Both seeds (rows) and states (columns)
    are resampled with replacement, the same draws for both tables when comparing.
    """
    rng = np.random.default_rng(rng_seed)
    values = np.empty(repeats)
    for i in range(repeats):
        rows = rng.integers(0, table.shape[0], table.shape[0])
        columns = rng.integers(0, table.shape[1], table.shape[1])
        values[i] = table[np.ix_(rows, columns)].mean()
        if baseline is not None:
            values[i] -= baseline[np.ix_(rows, columns)].mean()
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 3), round(float(high), 3)]


def aggregate(results_dir: Path, protocol: dict) -> dict:
    """Per-condition statistics and seed-paired comparisons, computed from the raw records."""
    seeds = protocol["seeds"]
    runs = {
        name: {seed: load_run(results_dir, name, seed, condition["updates"]) for seed in seeds}
        for name, condition in protocol["conditions"].items()
    }
    present = [run for by_seed in runs.values() for run in by_seed.values() if run]
    if not present:
        raise MultiSeedError(f"no results found under {results_dir}")
    banks = sorted({run["state_bank_sha256"] for run in present})
    if len(banks) != 1:
        raise MultiSeedError(f"runs were evaluated on more than one state bank: {banks}")

    conditions = {}
    for name, by_seed in runs.items():
        done = {seed: run for seed, run in by_seed.items() if run}
        counts = [run["successes"] for run in done.values()]
        errors = [run["test_mean_episode_mse"] for run in done.values() if run["test_mean_episode_mse"] is not None]
        missing = [seed for seed in seeds if seed not in done]
        incomplete = [seed for seed, run in done.items() if run["incomplete"]]
        conditions[name] = {
            "status": "INCOMPLETE" if missing or incomplete or not done else "complete",
            "description": protocol["conditions"][name].get("description", name),
            "updates": protocol["conditions"][name]["updates"],
            "missing_seeds": missing,
            "incomplete_seeds": incomplete,
            "trials_per_seed": protocol["final"]["num_states"],
            "successes_by_seed": {str(seed): run["successes"] for seed, run in done.items()},
            "completed_by_seed": {str(seed): run["completed"] for seed, run in done.items()},
            "mean_successes": round(statistics.mean(counts), 3) if counts else None,
            "std_successes": round(statistics.stdev(counts), 3) if len(counts) > 1 else None,
            "min_successes": min(counts) if counts else None,
            "max_successes": max(counts) if counts else None,
            "pooled_successes": sum(counts),
            "pooled_completed": sum(run["completed"] for run in done.values()),
            "pooled_rate_interval_95": bootstrap_interval(outcome_table(list(done.values()))[0])
            if len(done) > 1
            else None,
            "mean_test_mse": round(statistics.mean(errors), 6) if errors else None,
        }

    comparisons = []
    for comparison in protocol["comparisons"]:
        control, treatment = runs[comparison["control"]], runs[comparison["treatment"]]
        shared = [seed for seed in seeds if control[seed] and treatment[seed]]
        differences = {str(seed): treatment[seed]["successes"] - control[seed]["successes"] for seed in shared}
        if len(shared) > 1:
            paired = [treatment[seed] for seed in shared] + [control[seed] for seed in shared]
            table, _ = outcome_table(paired)  # only states that every model in the pair completed
            rate_difference = bootstrap_interval(table[: len(shared)], table[len(shared) :])
        else:
            rate_difference = None
        better = sum(d > 0 for d in differences.values())
        worse = sum(d < 0 for d in differences.values())
        status = [conditions[comparison[k]]["status"] for k in ("control", "treatment")]
        comparisons.append(
            {
                "control": comparison["control"],
                "treatment": comparison["treatment"],
                "differs_in": comparison.get("differs_in"),
                "status": "complete" if status == ["complete", "complete"] else "INCOMPLETE",
                "difference_by_seed": differences,
                "seeds_treatment_better": better,
                "seeds_equal": len(differences) - better - worse,
                "seeds_control_better": worse,
                "mean_difference": round(statistics.mean(differences.values()), 3) if differences else None,
                "sign_test_p_value": round(sign_test(better, worse), 4),
                "rate_difference_interval_95": rate_difference,
            }
        )
    return {
        "seeds": seeds,
        "state_bank_sha256": banks[0],
        "conditions": conditions,
        "comparisons": comparisons,
        "note": (
            "Comparisons pair models that share a training seed. With 5 seeds, the smallest two-sided "
            "p-value a sign test can give is 0.0625. Pooled-rate intervals come from a bootstrap that "
            "resamples both training seeds and initial states (10000 repeats)."
        ),
    }


# ---------------------------------------------------------------- running


def study_dir(protocol):
    """Directory holding this protocol's recorded results."""
    return OUT / protocol["artifacts"]


def ensure_state_bank(protocol):
    """Create and check the shared final states if they do not exist yet."""
    final = protocol["final"]
    bank = OUT / final["state_bank"]
    if bank.exists():
        return bank
    reference = json.loads((ARTIFACTS / FIRST_EVALUATION / "selection.json").read_text(encoding="utf-8"))["selected"]
    checkpoint = OUT / reference["checkpoint"]
    create_state_bank(checkpoint, final["seeds"], bank, role="final")
    report = verify_state_bank(checkpoint, bank)
    report["distinctness"] = check_bank_is_distinct(bank, OUT / DATASET, [OUT / p for p in final["distinct_from"]])
    write_json(study_dir(protocol) / "state_bank_final.json", report)
    return bank


def run_one(condition, seed, protocol):
    """Train one condition with one seed if needed, then evaluate its last checkpoint."""
    out = study_dir(protocol) / condition / f"seed_{seed}"
    if (out / "results.json").exists():
        return
    spec = protocol["conditions"][condition]
    name = f"ms_{condition}_seed{seed}"
    if not list((OUT / "runs" / name).glob("*/run_record.json")):
        subprocess.run(
            [sys.executable, "-m", "scripts.train", "--config", spec["config"], "--name", name, "--seed", str(seed)],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    record_path = sorted((OUT / "runs" / name).glob("*/run_record.json"))[-1]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    last = max(record["checkpoints"], key=lambda c: c["update"])
    if last["update"] != spec["updates"] or record["seed"] != seed:
        raise MultiSeedError(
            f"{name}: trained run does not match the protocol: {last['update']}, seed {record['seed']}"
        )
    checkpoint = record_path.parent / "models" / last["file"]
    if sha256_file(checkpoint) != last["sha256"]:
        raise MultiSeedError(f"{checkpoint} does not match the hash recorded at training time")

    bank = OUT / protocol["final"]["state_bank"]
    dataset = OUT / DATASET
    evaluate_rollouts(checkpoint, bank, protocol["rollout"]["horizon"], out)
    offline = score_offline(checkpoint, dataset, episodes_for_mask(dataset, protocol["final"]["test_mask"]))
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    results.update(
        condition=condition,
        seed=seed,
        update=last["update"],
        offline_test=offline,
        runtime_seconds=record["runtime_seconds"],
    )
    write_json(out / "results.json", results)


def command_run(args):
    """Run every missing (condition, seed) pair, a few at a time."""
    protocol = load_protocol()
    ensure_state_bank(protocol)
    jobs = [(c, s) for s in protocol["seeds"] for c in protocol["conditions"]]
    jobs = [j for j in jobs if not (study_dir(protocol) / j[0] / f"seed_{j[1]}" / "results.json").exists()]
    print(f"{len(jobs)} runs to do, {args.workers} at a time", flush=True)

    def launch(job):
        condition, seed = job
        done = subprocess.run(
            [sys.executable, "-m", "scripts.multiseed", "run-one", "--condition", condition, "--seed", str(seed)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        status = "done" if done.returncode == 0 else f"FAILED: {done.stderr.strip().splitlines()[-1:]}"
        print(f"{condition} seed {seed}: {status}", flush=True)
        return done.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(launch, jobs))
    failed = sum(code != 0 for code in codes)
    print(f"finished: {len(jobs) - failed} succeeded, {failed} failed")
    return 1 if failed else 0


def command_run_one(args):
    """Run a single (condition, seed) pair."""
    run_one(args.condition, args.seed, load_protocol())
    return 0


def command_summarize(_args):
    """Aggregate the recorded results and print a short table."""
    protocol = load_protocol()
    summary = aggregate(study_dir(protocol), protocol)
    write_json(study_dir(protocol) / "protocol.json", protocol)  # the report reads the protocol from here
    write_json(study_dir(protocol) / "summary.json", summary)
    for name, c in summary["conditions"].items():
        print(f"{name:12s} {c['status']:10s} {list(c['successes_by_seed'].values())} mean {c['mean_successes']}")
    for c in summary["comparisons"]:
        print(f"{c['treatment']} vs {c['control']}: {c['difference_by_seed']} p={c['sign_test_p_value']}")
    return 0


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="models trained and evaluated at once (default: CPU cores minus two)",
    )
    run.set_defaults(func=command_run)
    one = commands.add_parser("run-one")
    one.add_argument("--condition", required=True)
    one.add_argument("--seed", type=int, required=True)
    one.set_defaults(func=command_run_one)
    commands.add_parser("summarize").set_defaults(func=command_summarize)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
