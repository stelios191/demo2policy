"""Check every recorded count against the per-trial records, and draw the charts.

Nothing here runs a policy or a simulator. Counts are recomputed from the per-trial
records and must agree with the stored summaries. Required inputs that are absent
raise an error; optional inputs that are absent are labelled "missing".

    python -m scripts.report --collect          # copy curves and run records out of runs/
    python -m scripts.report                    # charts in docs/report/
"""

import argparse
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

from scripts.common import FIRST_EVALUATION, FOLLOW_UP, MULTI_SEED, OUT, TIMING_EXPERIMENT, relative_path, write_json
from scripts.evaluate import latest_attempts, read_records, summarize_trials
from scripts.multiseed import MultiSeedError, aggregate, sign_test

MISSING = "missing"
CONTROL, TREATMENT = "matched_aligned", "shift_plus_3"
LONGER, REFERENCE = "longer_training", "first_model_same_states"
REFERENCE_DIR = "reference_10k"
COUNT_FIELDS = ["attempted", "completed", "successes", "policy_failures", "infrastructure_errors", "attempt_records"]

# chart colours: fixed categorical order, validated for colour-vision deficiency
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, TEXT, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


class ReportError(Exception):
    """The recorded outputs are absent or inconsistent; no report is produced."""


def wilson_interval(successes, trials, z=1.96):
    """95% Wilson score interval for a proportion, or None when there are no trials."""
    if trials == 0:
        return None
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return (max(0.0, centre - half), min(1.0, centre + half))


def require(path):
    """Return the path, or raise ReportError if it does not exist."""
    if not Path(path).exists():
        raise ReportError(f"missing required input: {path}")
    return Path(path)


def load_policy_results(name, final_dir, selection_path):
    """Load one policy's results, recompute its counts from the raw records, and check they agree."""
    results = json.loads(require(final_dir / "results.json").read_text(encoding="utf-8"))
    records = read_records(require(final_dir / "rollouts.csv"))
    selection = json.loads(require(selection_path).read_text(encoding="utf-8"))

    stored = results["summary"]
    recomputed = summarize_trials(records, planned=stored["planned"])
    for field in COUNT_FIELDS + ["incomplete"]:
        if stored[field] != recomputed[field]:
            raise ReportError(
                f"{name}: stored summary says {field}={stored[field]} but the raw records give {recomputed[field]}"
            )
    wrong = sorted({r["checkpoint_sha256"] for r in records} - {results["checkpoint_sha256"]})
    if wrong:
        raise ReportError(f"{name}: records reference checkpoints other than the evaluated one: {wrong}")

    latest = latest_attempts(records)
    success_steps = sorted(r["first_success_step"] for r in latest.values() if r["outcome"] == "success")
    interval = wilson_interval(recomputed["successes"], recomputed["completed"])
    offline = results.get("offline_test")
    entry = {
        "status": "INCOMPLETE" if recomputed["incomplete"] else "complete",
        "planned": stored["planned"],
        **{field: recomputed[field] for field in COUNT_FIELDS},
        "success_rate_of_completed": (
            round(recomputed["successes"] / recomputed["completed"], 4) if recomputed["completed"] else MISSING
        ),
        "wilson_95_interval": [round(v, 4) for v in interval] if interval else MISSING,
        "median_steps_to_success": statistics.median(success_steps) if success_steps else MISSING,
        "outcomes": {
            o: sum(r["outcome"] == o for r in latest.values()) for o in sorted({r["outcome"] for r in latest.values()})
        },
        "test_mean_episode_mse": offline["mean_episode_mse"] if offline else MISSING,
        "test_episodes": offline["num_episodes"] if offline else MISSING,
        "selected_update": results.get("update", MISSING),
        "checkpoint_sha256": results["checkpoint_sha256"],
        "state_bank_sha256": results["state_bank_sha256"],
        "selection_candidates": [
            {
                "update": c["update"],
                "development_successes": c["development_successes"],
                "development_completed": c.get("development_completed", MISSING),
                "validation_mean_episode_mse": c["validation_mean_episode_mse"],
            }
            for c in sorted(selection["candidates"], key=lambda c: c["update"])
        ],
    }
    return entry, latest


def paired_comparison(latest_control, latest_treatment, control=CONTROL, treatment=TREATMENT):
    """Trial-by-trial comparison on the states both policies completed."""
    shared = sorted(
        s
        for s in set(latest_control) & set(latest_treatment)
        if latest_control[s]["outcome"] != "infrastructure_error"
        and latest_treatment[s]["outcome"] != "infrastructure_error"
    )
    pairs = [(latest_control[s]["outcome"] == "success", latest_treatment[s]["outcome"] == "success") for s in shared]
    only_control = sum(c and not t for c, t in pairs)
    only_treatment = sum(t and not c for c, t in pairs)
    p_value = sign_test(only_treatment, only_control)  # exact two-sided, on the discordant pairs
    return {
        "control": control,
        "treatment": treatment,
        "shared_completed_states": len(shared),
        "both_succeed": sum(c and t for c, t in pairs),
        "both_fail": sum(not c and not t for c, t in pairs),
        "only_control_succeeds": only_control,
        "only_treatment_succeeds": only_treatment,
        "exact_sign_test_p_value": round(p_value, 4),
        "scope": "differences between these two trained models on these initial states; one training seed each",
    }


def print_summary(summary):
    """A few lines on the console: successes per policy, and per condition for the multi-seed study."""
    groups = [("first evaluation", summary["policies"])]
    if "follow_up" in summary:
        groups.append(("follow-up", summary["follow_up"]["policies"]))
    for title, policies in groups:
        print(title)
        for name, p in policies.items():
            print(f"  {name:26s} {p['successes']:3d} / {p['completed']:<3d} {p['status']}")
    if "multi_seed" in summary:
        print("multi-seed study")
        for name, c in summary["multi_seed"]["conditions"].items():
            counts = ", ".join(str(n) for n in c["successes_by_seed"].values())
            print(f"  {name:26s} {counts}  mean {c['mean_successes']}  {c['status']}")


def style_axes(ax):
    """Apply the shared, recessive axis styling."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(which="both", colors=TEXT_SECONDARY, length=0, labelsize=9)
    ax.set_axisbelow(True)


# condition names as used in the docs, by the condition key's first word
DISPLAY_NAMES = {"bc": "full episodes", "aligned": "timing control", "shifted": "shifted labels"}


def chart_label(name, condition):
    """'full episodes, 50,000 updates' for bc_50k; the protocol's description for unknown conditions."""
    display = DISPLAY_NAMES.get(name.split("_")[0])
    return f"{display}, {condition['updates']:,} updates" if display else condition.get("description", name)


def plot_multi_seed(study, path):
    """Dot plot with one dot per trained model and a bar at each condition's mean."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, c in study["conditions"].items() if c["successes_by_seed"]]
    trials = max(study["conditions"][n]["trials_per_seed"] for n in names)
    lowest = min(study["conditions"][n]["min_successes"] for n in names)
    fig, ax = plt.subplots(figsize=(8.5, 1.3 + 0.62 * len(names)), facecolor=SURFACE)
    style_axes(ax)
    positions = list(range(len(names)))[::-1]
    for position, name in zip(positions, names, strict=True):
        c = study["conditions"][name]
        counts = [c["successes_by_seed"][seed] for seed in sorted(c["successes_by_seed"], key=int)]
        seen = {}
        for count in counts:  # stack models with the same score so that every seed stays visible
            offset = seen.get(count, 0)
            seen[count] = offset + 1
            ax.plot(
                count,
                position + 0.24 * (offset - (counts.count(count) - 1) / 2),
                "o",
                color=SERIES[0],
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                zorder=3,
            )
        ax.plot([c["mean_successes"]] * 2, [position - 0.3, position + 0.3], color=TEXT, linewidth=2, zorder=2)
        ax.text(trials + 0.6, position, f"mean {c['mean_successes']:.1f}", va="center", color=TEXT, fontsize=9)
    labels = [chart_label(n, study["conditions"][n]) for n in names]
    ax.set_yticks(positions, labels, color=TEXT, fontsize=10)
    ax.set_ylim(-0.6, len(names) - 0.4)
    left = max(0, 5 * ((lowest - 3) // 5))
    ax.set_xlim(left, trials + 4.5)
    ax.set_xticks(list(range(left, trials + 1, 5)))
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel(f"cube lifted, out of {trials} trials", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title("Task success across training seeds", loc="left", color=TEXT, fontsize=11, pad=22)
    ax.text(
        0,
        1.04,
        "each dot is one trained model; the bar marks the mean of its condition",
        transform=ax.transAxes,
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_training_curves(curves, path):
    """Training loss and sampled validation loss against optimizer updates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    style_axes(ax)
    series = [
        ("train_loss", "training loss", SERIES[0]),
        ("validation_loss_sampled", "validation loss (sampled batches)", SERIES[1]),
    ]
    for key, label, colour in series:
        ax.plot(curves["updates"], curves[key], color=colour, linewidth=2, label=label, solid_capstyle="round")
        ax.plot(
            curves["updates"][-1],
            curves[key][-1],
            "o",
            color=colour,
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
    ax.set_yscale("log")
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xlabel("optimizer updates", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("mean squared action error (log scale)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title("Baseline training and sampled validation loss", loc="left", color=TEXT, fontsize=11, pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def load_timing_comparison(results_dir, policies, latest):
    """Add the two timing-experiment policies, if that experiment was run. Returns their names."""
    alignment = results_dir / TIMING_EXPERIMENT
    present = [n for n in (CONTROL, TREATMENT) if (alignment / n / "final" / "results.json").exists()]
    if present and present != [CONTROL, TREATMENT]:
        absent = sorted({CONTROL, TREATMENT} - set(present))
        raise ReportError(f"timing comparison is partial: {present} present but {absent} absent")
    for name in present:
        policies[name], latest[name] = load_policy_results(
            name, alignment / name / "final", alignment / name / "selection.json"
        )
    return present


def load_follow_up(results_dir, baseline):
    """The longer-trained model and the first model on the same new states, or None if not run."""
    follow_up_dir = results_dir / FOLLOW_UP
    if not (follow_up_dir / "results.json").exists():
        return None
    reference_dir = follow_up_dir / REFERENCE_DIR
    if not (reference_dir / "results.json").exists():
        raise ReportError(f"follow-up is partial: {REFERENCE_DIR} results are absent from {follow_up_dir}")
    longer, latest_longer = load_policy_results(LONGER, follow_up_dir, follow_up_dir / "selection.json")
    reference, latest_reference = load_policy_results(
        REFERENCE, reference_dir, results_dir / FIRST_EVALUATION / "selection.json"
    )
    if longer["state_bank_sha256"] != reference["state_bank_sha256"]:
        raise ReportError("follow-up policies were evaluated on different state banks and cannot be compared")
    if reference["checkpoint_sha256"] != baseline["checkpoint_sha256"]:
        raise ReportError("follow-up reference is not the baseline checkpoint from the first evaluation")
    return {
        "policies": {LONGER: longer, REFERENCE: reference},
        "paired_comparison": paired_comparison(latest_reference, latest_longer, REFERENCE, LONGER),
    }


def load_multi_seed(results_dir):
    """The multi-seed study recomputed from its raw records, or None if not run."""
    study_dir = results_dir / MULTI_SEED
    if not (study_dir.exists() and any(study_dir.glob("*/seed_*/results.json"))):
        return None
    protocol = json.loads(require(study_dir / "protocol.json").read_text(encoding="utf-8"))
    try:
        return aggregate(study_dir, protocol)
    except MultiSeedError as exc:
        raise ReportError(str(exc)) from exc


def build_report(results_dir: Path, output_dir: Path) -> None:
    """Validate every input first, then write summary.json and the plots."""
    results_dir, output_dir = Path(results_dir), Path(output_dir)
    require(results_dir)
    policies, latest = {}, {}
    first = results_dir / FIRST_EVALUATION
    policies["baseline"], latest["baseline"] = load_policy_results("baseline", first, first / "selection.json")
    timing = load_timing_comparison(results_dir, policies, latest)

    curves_path = first / "training_curves.json"
    curves = json.loads(curves_path.read_text(encoding="utf-8")) if curves_path.exists() else None
    summary = {
        "policies": policies,
        "training_curves": "training_curves.png" if curves else MISSING,
        "interval_note": (
            "Wilson 95% intervals describe sampling of initial states for one trained model. "
            "They say nothing about variation between training seeds."
        ),
    }
    if timing:
        summary["paired_comparison"] = paired_comparison(latest[CONTROL], latest[TREATMENT])
    follow_up = load_follow_up(results_dir, policies["baseline"])
    if follow_up:
        summary["follow_up"] = follow_up
    study = load_multi_seed(results_dir)
    if study:
        summary["multi_seed"] = study

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    if "multi_seed" in summary:
        plot_multi_seed(summary["multi_seed"], output_dir / "multi_seed_success.png")
    if curves:
        plot_training_curves(curves, output_dir / "training_curves.png")


# ---------------------------------------------------------------- collecting from runs/


def extract_training_curves(log_path: Path, updates_per_epoch: int) -> dict:
    """Read per-epoch losses from a robomimic training log."""
    text = Path(log_path).read_text(encoding="utf-8")

    def losses(kind):
        pattern = rf"^{kind} Epoch (\d+)\n\{{(.*?)\n\}}"
        found = {}
        for epoch, body in re.findall(pattern, text, flags=re.S | re.M):
            found[int(epoch)] = float(re.search(r'"Loss": ([0-9.eE+-]+)', body).group(1))
        return found

    train, validation = losses("Train"), losses("Validation")
    epochs = sorted(set(train) & set(validation))
    if not epochs:
        raise ReportError(f"no training epochs found in {log_path}")
    return {
        "source": "robomimic training log",
        "updates": [e * updates_per_epoch for e in epochs],
        "train_loss": [train[e] for e in epochs],
        "validation_loss_sampled": [validation[e] for e in epochs],
        "note": "validation loss here is computed on sampled batches each epoch; it is diagnostic only",
    }


def collect(results_dir: Path):
    """Copy small run records and loss curves from runs/ into the published artifacts."""
    destinations = {
        "bc_lift_10k": results_dir / FIRST_EVALUATION,
        "bc_matched_aligned_10k": results_dir / TIMING_EXPERIMENT / CONTROL,
        "bc_shift_plus_3_10k": results_dir / TIMING_EXPERIMENT / TREATMENT,
        "bc_lift_50k": results_dir / FOLLOW_UP,
    }
    for experiment, destination in destinations.items():
        run_dirs = sorted((OUT / "runs" / experiment).glob("*/run_record.json"))
        if not run_dirs:
            continue
        run_dir = run_dirs[-1].parent
        record = json.loads(run_dirs[-1].read_text(encoding="utf-8"))
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dirs[-1], destination / "run_record.json")
        curves = extract_training_curves(run_dir / "logs" / "log.txt", record["updates_per_epoch"])
        write_json(destination / "training_curves.json", curves)
        print(f"collected {experiment} -> {relative_path(destination)}")


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=OUT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=OUT / "docs" / "report")
    parser.add_argument("--collect", action="store_true", help="copy run records and loss curves from runs/ first")
    args = parser.parse_args(argv)
    try:
        if args.collect:
            collect(args.results_dir)
        build_report(args.results_dir, args.output_dir)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print_summary(json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
