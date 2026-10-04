import csv
import json

import pytest

from scripts.compare import compare_trees, differences, normalize


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_normalize_drops_values_that_differ_between_runs():
    record = {
        "successes": 43,
        "checkpoint_sha256": "abc",
        "run_id": "final-abc",
        "runtime_seconds": 15.9,
        "checkpoint": "runs/bc_lift_10k/20260929183827/models/model_epoch_100.pth",
        "nested": [{"state_bank_sha256": "x", "steps": 40}],
        "source_sha256": "kept",
    }
    assert normalize(record) == {
        "successes": 43,
        "checkpoint": "runs/bc_lift_10k/<time>/models/model_epoch_100.pth",
        "nested": [{"steps": 40}],
        "source_sha256": "kept",
    }


def test_differences_allow_tiny_float_noise_but_not_real_changes():
    assert differences({"mse": 0.0312800001}, {"mse": 0.03128}) == []
    assert differences({"mse": 0.0313}, {"mse": 0.0320}) != []
    assert differences({"successes": 43}, {"successes": 42}) == ["successes: committed 43, reproduced 42"]
    assert differences([1, 2], [1, 2, 3]) == ["length: committed 2, reproduced 3"]
    assert differences({"a": 1}, {"b": 1}) != []


@pytest.fixture
def trees(tmp_path):
    committed, reproduced = tmp_path / "committed", tmp_path / "reproduced"
    for root, outcome in ((committed, "success"), (reproduced, "success")):
        write_json(root / "results.json", {"summary": {"successes": 1}, "checkpoint_sha256": str(root)})
        write_csv(root / "rollouts.csv", [{"state_id": "s0", "outcome": outcome, "run_id": str(root)}])
    return committed, reproduced


def test_identical_results_match_even_when_hashes_and_run_ids_differ(trees):
    rows, problems = compare_trees(*trees)
    assert problems == 0
    assert {row["status"] for row in rows} == {"same"}


def test_a_changed_trial_outcome_is_reported(trees):
    committed, reproduced = trees
    write_csv(reproduced / "rollouts.csv", [{"state_id": "s0", "outcome": "horizon_reached", "run_id": "r"}])
    rows, problems = compare_trees(committed, reproduced)
    assert problems == 1
    row = next(r for r in rows if r["file"] == "rollouts.csv")
    assert row["status"] == "differs" and "outcome" in row["detail"]


def test_missing_reproduction_is_a_problem_unless_it_needs_the_full_run(trees):
    committed, reproduced = trees
    write_json(committed / "extra.json", {"x": 1})
    write_json(committed / "2_diagnosis" / "explore_mlp256_200k" / "summary.json", [{"successes": 99}])
    rows, problems = compare_trees(committed, reproduced, scope="core")
    status = {row["file"]: row["status"] for row in rows}
    assert status["extra.json"] == "missing"
    assert status["2_diagnosis/explore_mlp256_200k/summary.json"] == "not run (needs --full)"
    assert problems == 1


def test_quick_scope_expects_only_the_first_evaluation(trees):
    committed, reproduced = trees
    write_json(committed / "1_first_evaluation" / "development" / "update_2500" / "results.json", {"x": 1})
    write_json(committed / "4_multi_seed" / "bc_50k" / "seed_1" / "results.json", {"x": 1})
    rows, problems = compare_trees(committed, reproduced, scope="quick")
    status = {row["file"]: row["status"] for row in rows}
    assert status["1_first_evaluation/development/update_2500/results.json"] == "missing"
    assert status["4_multi_seed/bc_50k/seed_1/results.json"] == "not run (quick check)"
    assert problems == 1


def test_full_scope_expects_everything(trees):
    committed, reproduced = trees
    write_json(committed / "2_diagnosis" / "explore_mlp256_200k" / "summary.json", [{"successes": 99}])
    _, problems = compare_trees(committed, reproduced, scope="full")
    assert problems == 1
