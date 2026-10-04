import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate import RECORD_FIELDS
from scripts.multiseed import MultiSeedError, aggregate

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "seeds": [1, 2, 3],
    "conditions": {"short": {"updates": 100}, "long": {"updates": 500}},
    "comparisons": [{"control": "short", "treatment": "long"}],
    "final": {"num_states": 4},
}


def write_run(root, condition, seed, successes, planned=4, bank="bank", stored_successes=None, error_trials=0):
    out = root / condition / f"seed_{seed}"
    out.mkdir(parents=True)
    rows = []
    for i in range(planned):
        if i < error_trials:
            outcome, success = "infrastructure_error", ""
        elif i < error_trials + successes:
            outcome, success = "success", True
        else:
            outcome, success = "horizon_reached", False
        rows.append(
            {
                "run_id": "r",
                "state_id": f"state_{i}",
                "seed": i,
                "attempt": 1,
                "outcome": outcome,
                "success": success,
                "steps": 40,
                "first_success_step": 40 if outcome == "success" else "",
                "horizon": 400,
                "checkpoint_sha256": f"ckpt-{condition}-{seed}",
                "state_bank_sha256": bank,
                "error": "",
            }
        )
    with open(out / "rollouts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    completed = planned - error_trials
    results = {
        "condition": condition,
        "seed": seed,
        "update": PROTOCOL["conditions"][condition]["updates"],
        "checkpoint_sha256": f"ckpt-{condition}-{seed}",
        "state_bank_sha256": bank,
        "summary": {
            "planned": planned,
            "attempted": planned,
            "completed": completed,
            "successes": successes if stored_successes is None else stored_successes,
            "policy_failures": 0,
            "infrastructure_errors": error_trials,
            "incomplete": completed != planned,
            "attempt_records": planned,
        },
        "offline_test": {"mean_episode_mse": 0.03, "num_episodes": 20},
    }
    (out / "results.json").write_text(json.dumps(results))


def write_all(root, short=(2, 3, 2), long=(4, 3, 4)):
    for seed, n in zip(PROTOCOL["seeds"], short, strict=True):
        write_run(root, "short", seed, n)
    for seed, n in zip(PROTOCOL["seeds"], long, strict=True):
        write_run(root, "long", seed, n)


def test_per_condition_statistics(tmp_path):
    write_all(tmp_path)
    summary = aggregate(tmp_path, PROTOCOL)
    short, long = summary["conditions"]["short"], summary["conditions"]["long"]
    assert short["status"] == "complete" and long["status"] == "complete"
    assert short["successes_by_seed"] == {"1": 2, "2": 3, "3": 2}
    assert short["mean_successes"] == pytest.approx(7 / 3, abs=1e-3)
    assert (short["min_successes"], short["max_successes"]) == (2, 3)
    assert long["pooled_successes"] == 11 and long["pooled_completed"] == 12
    assert long["trials_per_seed"] == 4


def test_comparison_is_paired_by_seed(tmp_path):
    write_all(tmp_path)
    (comparison,) = aggregate(tmp_path, PROTOCOL)["comparisons"]
    assert comparison["control"] == "short" and comparison["treatment"] == "long"
    assert comparison["difference_by_seed"] == {"1": 2, "2": 0, "3": 2}
    assert comparison["seeds_treatment_better"] == 2
    assert comparison["seeds_equal"] == 1
    assert comparison["seeds_control_better"] == 0
    assert comparison["mean_difference"] == pytest.approx(4 / 3, abs=1e-3)
    assert comparison["sign_test_p_value"] == pytest.approx(0.5)


def test_sign_test_with_five_consistent_seeds_cannot_go_below_its_floor(tmp_path):
    protocol = dict(PROTOCOL, seeds=[1, 2, 3, 4, 5])
    for seed in protocol["seeds"]:
        write_run(tmp_path, "short", seed, 1)
        write_run(tmp_path, "long", seed, 4)
    (comparison,) = aggregate(tmp_path, protocol)["comparisons"]
    assert comparison["seeds_treatment_better"] == 5
    assert comparison["sign_test_p_value"] == pytest.approx(0.0625)


def test_missing_seed_is_reported_not_filled_in(tmp_path):
    write_all(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "long" / "seed_2")
    summary = aggregate(tmp_path, PROTOCOL)
    long = summary["conditions"]["long"]
    assert long["status"] == "INCOMPLETE" and long["missing_seeds"] == [2]
    assert long["successes_by_seed"] == {"1": 4, "3": 4}
    (comparison,) = summary["comparisons"]
    assert comparison["status"] == "INCOMPLETE"
    assert comparison["difference_by_seed"] == {"1": 2, "3": 2}


def test_run_with_simulator_errors_marks_the_condition_incomplete(tmp_path):
    write_all(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "short" / "seed_1")
    write_run(tmp_path, "short", 1, 2, error_trials=1)
    summary = aggregate(tmp_path, PROTOCOL)
    assert summary["conditions"]["short"]["status"] == "INCOMPLETE"
    assert summary["conditions"]["short"]["incomplete_seeds"] == [1]


def test_stored_summary_must_match_the_raw_records(tmp_path):
    write_all(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "short" / "seed_3")
    write_run(tmp_path, "short", 3, 2, stored_successes=4)
    with pytest.raises(MultiSeedError, match="short.*seed 3.*successes"):
        aggregate(tmp_path, PROTOCOL)


def test_all_runs_must_share_one_state_bank(tmp_path):
    write_all(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "long" / "seed_1")
    write_run(tmp_path, "long", 1, 4, bank="another")
    with pytest.raises(MultiSeedError, match="state bank"):
        aggregate(tmp_path, PROTOCOL)


def test_run_must_match_the_declared_condition(tmp_path):
    write_all(tmp_path)
    path = tmp_path / "long" / "seed_1" / "results.json"
    results = json.loads(path.read_text())
    results["update"] = 100
    path.write_text(json.dumps(results))
    with pytest.raises(MultiSeedError, match="long.*seed 1.*update"):
        aggregate(tmp_path, PROTOCOL)


def test_no_results_at_all_is_an_error(tmp_path):
    with pytest.raises(MultiSeedError, match="no results"):
        aggregate(tmp_path, PROTOCOL)


# ---------------------------------------------------------------- the committed protocol


def load(name):
    return json.loads((ROOT / "configs" / f"{name}.json").read_text())


def test_protocol_conditions_match_their_configs():
    protocol = load("protocol_multi_seed")
    assert protocol["seeds"] == [1, 2, 3, 4, 5]
    assert set(protocol["conditions"]) == {
        "bc_10k",
        "bc_50k",
        "aligned_10k",
        "shifted_10k",
        "aligned_50k",
        "shifted_50k",
    }
    for name, condition in protocol["conditions"].items():
        config = json.loads((ROOT / condition["config"]).read_text())
        updates = config["train"]["num_epochs"] * config["experiment"]["epoch_every_n_steps"]
        assert updates == condition["updates"], name
        assert max(config["experiment"]["save"]["epochs"]) == config["train"]["num_epochs"], name
        assert config["train"]["hdf5_filter_key"] == "d2p_train", name
    assert protocol["checkpoint_rule"] == "last"


def test_conditions_within_a_comparison_differ_only_as_declared():
    protocol = load("protocol_multi_seed")

    def stripped(condition):
        config = json.loads((ROOT / protocol["conditions"][condition]["config"]).read_text())
        config["experiment"].pop("name")
        return config

    for comparison in protocol["comparisons"]:
        control, treatment = stripped(comparison["control"]), stripped(comparison["treatment"])
        differs = comparison["differs_in"]
        if differs == "training length":
            for config in (control, treatment):
                config["train"].pop("num_epochs")
                config["experiment"]["save"].pop("epochs")
        elif differs == "action labels":
            assert control["train"].pop("data") != treatment["train"].pop("data")
        else:
            raise AssertionError(differs)
        assert control == treatment, comparison


def test_protocol_final_states_are_new():
    protocol = load("protocol_multi_seed")
    used = set(range(10000, 10010)) | set(range(20000, 20050)) | set(range(11000, 11100)) | set(range(30000, 30050))
    assert protocol["final"]["seeds"] == list(range(40000, 40050))
    assert not set(protocol["final"]["seeds"]) & used
    assert len(protocol["final"]["distinct_from"]) == 5


def test_bootstrap_interval_covers_the_observed_rate_and_is_repeatable():
    from scripts.multiseed import bootstrap_interval

    table = np.array([[1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]], dtype=float)
    low, high = bootstrap_interval(table)
    assert low <= table.mean() <= high
    assert bootstrap_interval(table) == [low, high]
    assert bootstrap_interval(np.ones((3, 4))) == [1.0, 1.0]


def test_bootstrap_difference_is_zero_for_identical_models_and_positive_for_a_clear_gain():
    from scripts.multiseed import bootstrap_interval

    table = np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=float)
    assert bootstrap_interval(table, table) == [0.0, 0.0]
    low, _ = bootstrap_interval(np.ones((5, 20)), np.zeros((5, 20)))
    assert low == 1.0


def test_summary_reports_intervals(tmp_path):
    for seed, (short, long) in zip([1, 2, 3], [(1, 3), (2, 4), (1, 4)], strict=True):
        write_run(tmp_path, "short", seed, short)
        write_run(tmp_path, "long", seed, long)
    summary = aggregate(tmp_path, PROTOCOL)
    low, high = summary["conditions"]["long"]["pooled_rate_interval_95"]
    assert low <= 11 / 12 <= high
    low, high = summary["comparisons"][0]["rate_difference_interval_95"]
    assert 0 < low <= high
