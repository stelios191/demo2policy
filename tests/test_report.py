import csv
import json

import pytest

from scripts.evaluate import RECORD_FIELDS
from scripts.common import FIRST_EVALUATION, FOLLOW_UP, MULTI_SEED, TIMING_EXPERIMENT
from scripts.report import ReportError, build_report, wilson_interval


def record(state, outcome, attempt=1, steps=40):
    success = {"success": True, "infrastructure_error": None}.get(outcome, False)
    return {
        "run_id": "run",
        "state_id": f"state_{state}",
        "seed": state,
        "attempt": attempt,
        "outcome": outcome,
        "success": success,
        "steps": steps,
        "first_success_step": steps if outcome == "success" else None,
        "horizon": 400,
        "checkpoint_sha256": "ckpt",
        "state_bank_sha256": "bank",
        "error": "RuntimeError: boom" if outcome == "infrastructure_error" else "",
    }


def write_evaluation(directory, records, planned, summary_overrides=None, mse=0.03):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "rollouts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow({k: ("" if row[k] is None else row[k]) for k in RECORD_FIELDS})
    latest = {}
    for row in records:
        latest[row["state_id"]] = row
    outcomes = [row["outcome"] for row in latest.values()]
    completed = sum(o != "infrastructure_error" for o in outcomes)
    summary = {
        "planned": planned,
        "attempted": len(latest),
        "completed": completed,
        "successes": outcomes.count("success"),
        "policy_failures": outcomes.count("policy_failure"),
        "infrastructure_errors": outcomes.count("infrastructure_error"),
        "incomplete": completed != planned,
        "attempt_records": len(records),
    }
    summary.update(summary_overrides or {})
    results = {
        "run_id": "run",
        "checkpoint": "runs/x/models/model_epoch_100.pth",
        "checkpoint_sha256": "ckpt",
        "state_bank_sha256": "bank",
        "state_bank_role": "final",
        "horizon": 400,
        "update": 10000,
        "summary": summary,
        "offline_test": {"mean_episode_mse": mse, "num_episodes": 20, "num_samples": 974, "mask": "d2p_test"},
    }
    (directory / "results.json").write_text(json.dumps(results))


def write_selection(directory):
    directory.mkdir(parents=True, exist_ok=True)
    candidates = [
        {
            "update": u,
            "development_successes": s,
            "development_completed": 10,
            "validation_mean_episode_mse": m,
            "checkpoint_sha256": f"sha{u}",
        }
        for u, s, m in ((2500, 4, 0.04), (5000, 9, 0.03))
    ]
    (directory / "selection.json").write_text(json.dumps({"candidates": candidates, "selected": candidates[1]}))


def write_curves(directory):
    curves = {"updates": [100, 200, 300], "train_loss": [0.2, 0.1, 0.05], "validation_loss_sampled": [0.21, 0.12, 0.07]}
    (directory / "training_curves.json").write_text(json.dumps(curves))


@pytest.fixture
def results_dir(tmp_path):
    directory = tmp_path / "artifacts"
    first = directory / FIRST_EVALUATION
    records = [record(i, "success") for i in range(4)] + [record(4, "horizon_reached", steps=400)]
    write_evaluation(first, records, planned=5)
    write_selection(first)
    write_curves(first)
    return directory


def test_report_contains_counts_from_the_records(results_dir, tmp_path):
    out = tmp_path / "report"
    build_report(results_dir, out)
    summary = json.loads((out / "summary.json").read_text())
    baseline = summary["policies"]["baseline"]
    assert baseline["successes"] == 4 and baseline["completed"] == 5 and baseline["planned"] == 5
    assert baseline["status"] == "complete"
    assert baseline["test_mean_episode_mse"] == 0.03
    assert not (out / "summary.md").exists()
    assert (out / "training_curves.png").stat().st_size > 0
    assert not (out / "task_success.png").exists()


def test_summary_that_disagrees_with_raw_records_is_rejected(tmp_path):
    directory = tmp_path / "artifacts"
    first = directory / FIRST_EVALUATION
    write_evaluation(first, [record(i, "success") for i in range(3)], planned=3, summary_overrides={"successes": 2})
    write_selection(first)
    out = tmp_path / "report"
    with pytest.raises(ReportError, match="successes"):
        build_report(directory, out)
    assert not out.exists()


def test_incomplete_run_stays_labelled_incomplete(tmp_path):
    directory = tmp_path / "artifacts"
    first = directory / FIRST_EVALUATION
    records = [record(0, "success"), record(1, "infrastructure_error", steps=3)]
    write_evaluation(first, records, planned=5)
    write_selection(first)
    out = tmp_path / "report"
    build_report(directory, out)
    baseline = json.loads((out / "summary.json").read_text())["policies"]["baseline"]
    assert baseline["status"] == "INCOMPLETE"
    assert baseline["attempted"] == 2 and baseline["completed"] == 1 and baseline["infrastructure_errors"] == 1
    assert baseline["planned"] == 5 and baseline["successes"] == 1


@pytest.mark.parametrize("missing", ["results.json", "rollouts.csv", "selection.json"])
def test_missing_required_input_raises_and_writes_nothing(results_dir, tmp_path, missing):
    (results_dir / FIRST_EVALUATION / missing).unlink()
    out = tmp_path / "report"
    with pytest.raises(ReportError, match=f"missing required input.*{missing}"):
        build_report(results_dir, out)
    assert not out.exists()


def test_missing_results_directory_raises(tmp_path):
    with pytest.raises(ReportError, match="missing required input"):
        build_report(tmp_path / "nowhere", tmp_path / "report")


def test_missing_optional_input_is_labelled_missing_not_invented(results_dir, tmp_path):
    (results_dir / FIRST_EVALUATION / "training_curves.json").unlink()
    out = tmp_path / "report"
    build_report(results_dir, out)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["training_curves"] == "missing"
    assert not (out / "training_curves.png").exists()


def test_missing_offline_metric_is_labelled_missing(results_dir, tmp_path):
    results = json.loads((results_dir / FIRST_EVALUATION / "results.json").read_text())
    del results["offline_test"]
    (results_dir / FIRST_EVALUATION / "results.json").write_text(json.dumps(results))
    out = tmp_path / "report"
    build_report(results_dir, out)
    baseline = json.loads((out / "summary.json").read_text())["policies"]["baseline"]
    assert baseline["test_mean_episode_mse"] == "missing"


def test_timing_comparison_is_left_out_when_it_was_not_run(results_dir, tmp_path):
    out = tmp_path / "report"
    build_report(results_dir, out)
    summary = json.loads((out / "summary.json").read_text())
    assert list(summary["policies"]) == ["baseline"]
    assert "paired_comparison" not in summary


def test_timing_comparison_shows_each_condition_with_counts(results_dir, tmp_path):
    for name, successes in (("matched_aligned", 3), ("shift_plus_3", 5)):
        records = [record(i, "success" if i < successes else "horizon_reached") for i in range(5)]
        write_evaluation(results_dir / TIMING_EXPERIMENT / name / "final", records, planned=5)
        write_selection(results_dir / TIMING_EXPERIMENT / name)
    out = tmp_path / "report"
    build_report(results_dir, out)
    summary = json.loads((out / "summary.json").read_text())
    assert list(summary["policies"]) == ["baseline", "matched_aligned", "shift_plus_3"]
    assert summary["policies"]["matched_aligned"]["successes"] == 3
    assert summary["policies"]["shift_plus_3"]["successes"] == 5
    paired = summary["paired_comparison"]
    assert paired["control"] == "matched_aligned" and paired["treatment"] == "shift_plus_3"
    assert paired["both_succeed"] == 3 and paired["only_treatment_succeeds"] == 2
    assert paired["only_control_succeeds"] == 0 and paired["both_fail"] == 0


def test_a_single_alignment_condition_is_an_error_not_a_comparison(results_dir, tmp_path):
    records = [record(i, "success") for i in range(5)]
    write_evaluation(results_dir / TIMING_EXPERIMENT / "shift_plus_3" / "final", records, planned=5)
    write_selection(results_dir / TIMING_EXPERIMENT / "shift_plus_3")
    with pytest.raises(ReportError, match="matched_aligned"):
        build_report(results_dir, tmp_path / "report")


def test_report_is_deterministic(results_dir, tmp_path):
    build_report(results_dir, tmp_path / "a")
    build_report(results_dir, tmp_path / "b")
    assert (tmp_path / "a" / "summary.json").read_text() == (tmp_path / "b" / "summary.json").read_text()


def test_wilson_interval():
    low, high = wilson_interval(43, 50)
    assert low == pytest.approx(0.738, abs=0.002) and high == pytest.approx(0.930, abs=0.002)
    assert wilson_interval(50, 50)[1] == pytest.approx(1.0)
    assert wilson_interval(0, 0) is None


# ---------------------------------------------------------------- follow-up (longer training, new states)


def write_follow_up(results_dir, longer=5, reference=3, with_reference=True):
    records = [record(100 + i, "success" if i < longer else "horizon_reached") for i in range(5)]
    write_evaluation(results_dir / FOLLOW_UP, records, planned=5, mse=0.029)
    write_selection(results_dir / FOLLOW_UP)
    if with_reference:
        records = [record(100 + i, "success" if i < reference else "horizon_reached") for i in range(5)]
        write_evaluation(results_dir / FOLLOW_UP / "reference_10k", records, planned=5)


def test_follow_up_is_left_out_when_it_was_not_run(results_dir, tmp_path):
    build_report(results_dir, tmp_path / "report")
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert "follow_up" not in summary


def test_follow_up_is_reported_separately_from_the_first_evaluation(results_dir, tmp_path):
    write_follow_up(results_dir)
    out = tmp_path / "report"
    build_report(results_dir, out)
    summary = json.loads((out / "summary.json").read_text())
    assert list(summary["policies"]) == ["baseline"]
    follow_up = summary["follow_up"]
    assert list(follow_up["policies"]) == ["longer_training", "first_model_same_states"]
    assert follow_up["policies"]["longer_training"]["successes"] == 5
    assert follow_up["policies"]["first_model_same_states"]["successes"] == 3
    paired = follow_up["paired_comparison"]
    assert paired["control"] == "first_model_same_states" and paired["treatment"] == "longer_training"
    assert paired["only_treatment_succeeds"] == 2 and paired["only_control_succeeds"] == 0
    assert not (out / "follow_up_success.png").exists()


def test_follow_up_policies_must_share_a_state_bank(results_dir, tmp_path):
    write_follow_up(results_dir)
    path = results_dir / FOLLOW_UP / "reference_10k" / "results.json"
    results = json.loads(path.read_text())
    results["state_bank_sha256"] = "another-bank"
    path.write_text(json.dumps(results))
    with pytest.raises(ReportError, match="state bank"):
        build_report(results_dir, tmp_path / "report")


def test_partial_follow_up_is_an_error(results_dir, tmp_path):
    write_follow_up(results_dir, with_reference=False)
    with pytest.raises(ReportError, match="reference_10k"):
        build_report(results_dir, tmp_path / "report")


# ---------------------------------------------------------------- multi-seed study


def write_multiseed(results_dir, drop=None):
    from tests.test_multiseed import PROTOCOL, write_all

    root = results_dir / MULTI_SEED
    write_all(root)
    (root / "protocol.json").write_text(json.dumps(PROTOCOL))
    if drop:
        import shutil

        shutil.rmtree(root / drop[0] / f"seed_{drop[1]}")


def test_multiseed_is_left_out_when_it_was_not_run(results_dir, tmp_path):
    build_report(results_dir, tmp_path / "report")
    assert "multi_seed" not in json.loads((tmp_path / "report" / "summary.json").read_text())
    assert not (tmp_path / "report" / "multi_seed_success.png").exists()


def test_multiseed_reports_every_seed_and_the_paired_comparison(results_dir, tmp_path):
    write_multiseed(results_dir)
    out = tmp_path / "report"
    build_report(results_dir, out)
    study = json.loads((out / "summary.json").read_text())["multi_seed"]
    assert study["conditions"]["short"]["successes_by_seed"] == {"1": 2, "2": 3, "3": 2}
    assert study["comparisons"][0]["difference_by_seed"] == {"1": 2, "2": 0, "3": 2}
    assert study["conditions"]["long"]["successes_by_seed"] == {"1": 4, "2": 3, "3": 4}
    assert (out / "multi_seed_success.png").stat().st_size > 0


def test_incomplete_multiseed_study_is_labelled(results_dir, tmp_path):
    write_multiseed(results_dir, drop=("long", 2))
    out = tmp_path / "report"
    build_report(results_dir, out)
    study = json.loads((out / "summary.json").read_text())["multi_seed"]
    assert study["conditions"]["long"]["status"] == "INCOMPLETE"
    assert study["conditions"]["long"]["missing_seeds"] == [2]
    assert study["comparisons"][0]["status"] == "INCOMPLETE"


def test_multiseed_results_without_their_protocol_are_rejected(results_dir, tmp_path):
    write_multiseed(results_dir)
    (results_dir / MULTI_SEED / "protocol.json").unlink()
    with pytest.raises(ReportError, match="missing required input.*protocol.json"):
        build_report(results_dir, tmp_path / "report")
