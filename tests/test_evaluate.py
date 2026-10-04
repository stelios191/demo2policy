import csv
import hashlib
import json

import h5py
import numpy as np
import pytest

from scripts import evaluate
from scripts.evaluate import (
    evaluate_rollouts,
    run_trials,
    score_offline,
    select_checkpoint,
    summarize_trials,
)

OBS_DIMS = {"robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2, "object": 10}
ACTION_DIM = 7


# ---------------------------------------------------------------- offline scoring


class OffsetPolicy:
    """Predicts a constant action and counts calls; the dataset stores target = prediction + offset."""

    def __init__(self):
        self.calls = 0
        self.episodes_started = 0

    def start_episode(self):
        self.episodes_started += 1

    def __call__(self, ob):
        assert set(ob) == set(OBS_DIMS)
        assert all(ob[k].shape == (OBS_DIMS[k],) for k in ob)
        self.calls += 1
        return np.zeros(ACTION_DIM)


def write_offline_dataset(path, episodes):
    """episodes: {episode_id: (length, constant_action_value)}"""
    with h5py.File(path, "w") as f:
        for episode_id, (length, value) in episodes.items():
            group = f.create_group(f"data/{episode_id}")
            group.create_dataset("actions", data=np.full((length, ACTION_DIM), float(value)))
            for key, dim in OBS_DIMS.items():
                group.create_dataset(f"obs/{key}", data=np.zeros((length, dim)))
    return path


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model.pth"
    path.write_bytes(b"not a real checkpoint")
    return path


def test_offline_mse_averages_episode_means_regardless_of_length(tmp_path, checkpoint):
    # error 1 -> MSE 1 over 3 samples; error 3 -> MSE 9 over 131 samples
    dataset = write_offline_dataset(tmp_path / "d.hdf5", {"demo_0": (3, 1.0), "demo_1": (131, 3.0)})
    policy = OffsetPolicy()
    result = score_offline(checkpoint, dataset, ["demo_0", "demo_1"], load_policy=lambda path: policy)
    assert result["episode_mse"] == {"demo_0": pytest.approx(1.0), "demo_1": pytest.approx(9.0)}
    assert result["mean_episode_mse"] == pytest.approx(5.0)
    sample_weighted = (3 * 1.0 + 131 * 9.0) / 134
    assert result["mean_episode_mse"] != pytest.approx(sample_weighted)


def test_offline_scoring_includes_every_sample(tmp_path, checkpoint):
    # lengths chosen so that no common batch size divides them
    dataset = write_offline_dataset(tmp_path / "d.hdf5", {"demo_0": (129, 1.0), "demo_1": (7, 1.0)})
    policy = OffsetPolicy()
    result = score_offline(checkpoint, dataset, ["demo_0", "demo_1"], load_policy=lambda path: policy)
    assert policy.calls == 136
    assert result["num_samples"] == 136
    assert result["episode_num_samples"] == {"demo_0": 129, "demo_1": 7}
    assert policy.episodes_started == 2


def test_offline_scoring_only_reads_requested_episodes(tmp_path, checkpoint):
    dataset = write_offline_dataset(tmp_path / "d.hdf5", {"demo_0": (4, 1.0), "demo_1": (4, 3.0), "demo_2": (4, 5.0)})
    result = score_offline(checkpoint, dataset, ["demo_1"], load_policy=lambda path: OffsetPolicy())
    assert list(result["episode_mse"]) == ["demo_1"]
    assert result["mean_episode_mse"] == pytest.approx(9.0)


def test_offline_result_identifies_checkpoint_and_representation(tmp_path, checkpoint):
    dataset = write_offline_dataset(tmp_path / "d.hdf5", {"demo_0": (4, 1.0)})
    result = score_offline(checkpoint, dataset, ["demo_0"], load_policy=lambda path: OffsetPolicy())
    assert result["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert "action_representation" in result


def test_offline_scoring_rejects_bad_inputs(tmp_path, checkpoint):
    dataset = write_offline_dataset(tmp_path / "d.hdf5", {"demo_0": (4, 1.0)})
    with pytest.raises(ValueError, match="no episodes"):
        score_offline(checkpoint, dataset, [], load_policy=lambda path: OffsetPolicy())
    with pytest.raises(KeyError, match="demo_9"):
        score_offline(checkpoint, dataset, ["demo_9"], load_policy=lambda path: OffsetPolicy())
    with pytest.raises(ValueError, match="duplicate"):
        score_offline(checkpoint, dataset, ["demo_0", "demo_0"], load_policy=lambda path: OffsetPolicy())


def test_offline_scoring_reports_non_finite_predictions(tmp_path, checkpoint):
    class NanPolicy(OffsetPolicy):
        def __call__(self, ob):
            return np.full(ACTION_DIM, np.nan)

    dataset = write_offline_dataset(tmp_path / "d.hdf5", {"demo_0": (4, 1.0)})
    with pytest.raises(ValueError, match="non-finite.*demo_0"):
        score_offline(checkpoint, dataset, ["demo_0"], load_policy=lambda path: NanPolicy())


# ---------------------------------------------------------------- selection


def candidate(update, successes, mse, **overrides):
    entry = {
        "checkpoint": f"model_{update}.pth",
        "checkpoint_sha256": f"sha-{update}",
        "update": update,
        "development_successes": successes,
        "validation_mean_episode_mse": mse,
        "evidence": {"offline_mask": "d2p_val", "state_bank_role": "development"},
    }
    entry.update(overrides)
    return entry


def test_selection_prefers_more_development_successes():
    chosen = select_checkpoint([candidate(2500, 3, 0.01), candidate(5000, 7, 0.05), candidate(7500, 5, 0.001)])
    assert chosen["update"] == 5000


def test_selection_breaks_success_ties_by_lower_validation_mse():
    chosen = select_checkpoint([candidate(2500, 7, 0.05), candidate(5000, 7, 0.02), candidate(7500, 6, 0.001)])
    assert chosen["update"] == 5000


def test_selection_breaks_exact_ties_by_earlier_update():
    chosen = select_checkpoint([candidate(7500, 7, 0.02), candidate(2500, 7, 0.02), candidate(5000, 7, 0.02)])
    assert chosen["update"] == 2500


def test_selection_does_not_depend_on_candidate_order():
    candidates = [candidate(2500, 7, 0.05), candidate(5000, 7, 0.02), candidate(7500, 9, 0.5)]
    assert select_checkpoint(candidates) == select_checkpoint(list(reversed(candidates)))


@pytest.mark.parametrize(
    "overrides",
    [
        {"evidence": {"offline_mask": "d2p_test", "state_bank_role": "development"}},
        {"evidence": {"offline_mask": "d2p_val", "state_bank_role": "final"}},
        {"test_mean_episode_mse": 0.01},
        {"final_successes": 40},
    ],
)
def test_selection_rejects_final_test_evidence(overrides):
    with pytest.raises(ValueError, match="final|test"):
        select_checkpoint([candidate(2500, 7, 0.05), candidate(5000, 7, 0.02, **overrides)])


def test_selection_rejects_unlabelled_or_invalid_candidates():
    unlabelled = candidate(2500, 7, 0.05)
    del unlabelled["evidence"]
    with pytest.raises(ValueError, match="evidence"):
        select_checkpoint([unlabelled])
    with pytest.raises(ValueError, match="no candidates"):
        select_checkpoint([])
    with pytest.raises(ValueError, match="finite"):
        select_checkpoint([candidate(2500, 7, float("nan"))])
    with pytest.raises(ValueError, match="duplicate"):
        select_checkpoint([candidate(2500, 7, 0.05), candidate(2500, 6, 0.01)])


# ---------------------------------------------------------------- rollouts (mock environment)


class MockEnv:
    """Scripted environment. `script[state_value]` describes how the trial starting there behaves."""

    rollout_exceptions = (RuntimeError,)

    def __init__(self, script):
        self.script = script
        self.current = None
        self.steps = 0
        self.reset_to_calls = []

    def reset(self):
        return self._obs()

    def reset_to(self, state):
        self.current = self.script[int(state["states"][0])]
        self.steps = 0
        self.reset_to_calls.append(int(state["states"][0]))
        if self.current.get("raise_on_reset"):
            raise RuntimeError("reset failed")
        return self._obs()

    def step(self, action):
        assert np.all(np.isfinite(action)), "a non-finite action must never reach the simulator"
        self.steps += 1
        if self.current.get("raise_at") == self.steps:
            raise RuntimeError("simulator exploded")
        done = self.current.get("done_at") == self.steps
        return self._obs(), 0.0, done, {}

    def is_success(self):
        at = self.current.get("success_at")
        return {"task": at is not None and self.steps >= at}

    def get_state(self):
        return {"states": np.array([float(self.steps)])}

    def _obs(self):
        return {"obs": np.array([float(self.steps)])}


class MockPolicy:
    def __init__(self, nan_on_trial=None, nan_at_call=2):
        self.starts = 0
        self.calls_this_episode = 0
        self.nan_on_trial = nan_on_trial
        self.nan_at_call = nan_at_call

    def start_episode(self):
        self.starts += 1
        self.calls_this_episode = 0

    def __call__(self, ob):
        self.calls_this_episode += 1
        if self.starts == self.nan_on_trial and self.calls_this_episode == self.nan_at_call:
            return np.full(ACTION_DIM, np.nan)
        return np.zeros(ACTION_DIM)


def states(*values):
    return [
        {"state_id": f"state_{v}", "seed": 1000 + v, "states": np.array([float(v)]), "model": "<xml/>"} for v in values
    ]


IDS = {"run_id": "run", "checkpoint_sha256": "ckpt-sha", "state_bank_sha256": "bank-sha"}


def test_successful_trial_stops_at_first_success():
    env = MockEnv({0: {"success_at": 12}})
    records, _ = run_trials(MockPolicy(), env, states(0), horizon=400, **IDS)
    (record,) = records
    assert record["outcome"] == "success" and record["success"] is True
    assert record["steps"] == 12 and record["first_success_step"] == 12
    assert record["error"] == ""
    assert record["checkpoint_sha256"] == "ckpt-sha" and record["state_bank_sha256"] == "bank-sha"
    assert record["state_id"] == "state_0" and record["seed"] == 1000


def test_trial_without_success_runs_to_the_horizon():
    env = MockEnv({0: {}})
    records, _ = run_trials(MockPolicy(), env, states(0), horizon=400, **IDS)
    assert records[0]["outcome"] == "horizon_reached"
    assert records[0]["success"] is False
    assert records[0]["steps"] == 400 and records[0]["first_success_step"] is None


def test_environment_termination_ends_the_trial_early():
    env = MockEnv({0: {"done_at": 30}})
    records, _ = run_trials(MockPolicy(), env, states(0), horizon=400, **IDS)
    assert records[0]["outcome"] == "terminated"
    assert records[0]["success"] is False and records[0]["steps"] == 30


def test_non_finite_action_is_a_recorded_policy_failure():
    env = MockEnv({0: {"success_at": 5}, 1: {"success_at": 5}, 2: {"success_at": 5}})
    records, _ = run_trials(MockPolicy(nan_on_trial=2, nan_at_call=3), env, states(0, 1, 2), horizon=400, **IDS)
    assert [r["outcome"] for r in records] == ["success", "policy_failure", "success"]
    assert records[1]["success"] is False
    assert records[1]["steps"] == 2  # two finite actions were executed before the bad one
    assert "non-finite" in records[1]["error"]
    summary = summarize_trials(records, planned=3)
    assert summary["completed"] == 3 and summary["policy_failures"] == 1 and summary["successes"] == 2
    assert summary["incomplete"] is False


@pytest.mark.parametrize("defect", [{"raise_at": 4}, {"raise_on_reset": True}])
def test_infrastructure_exception_is_retained_and_marks_evaluation_incomplete(defect):
    env = MockEnv({0: {"success_at": 5}, 1: defect, 2: {"success_at": 5}})
    records, _ = run_trials(MockPolicy(), env, states(0, 1, 2), horizon=400, **IDS)
    assert [r["state_id"] for r in records] == ["state_0", "state_1", "state_2"]
    assert [r["outcome"] for r in records] == ["success", "infrastructure_error", "success"]
    assert "RuntimeError" in records[1]["error"]
    assert records[1]["success"] is None
    summary = summarize_trials(records, planned=3)
    assert summary == {
        "planned": 3,
        "attempted": 3,
        "completed": 2,
        "successes": 2,
        "policy_failures": 0,
        "infrastructure_errors": 1,
        "incomplete": True,
        "attempt_records": 3,
    }


def test_unexpected_exception_types_are_also_recorded():
    class OddEnv(MockEnv):
        def step(self, action):
            raise ZeroDivisionError("unexpected")

    records, _ = run_trials(MockPolicy(), OddEnv({0: {}}), states(0), horizon=400, **IDS)
    assert records[0]["outcome"] == "infrastructure_error"
    assert "ZeroDivisionError" in records[0]["error"]


def test_retry_uses_the_same_state_and_keeps_the_failed_attempt():
    broken = MockEnv({0: {"success_at": 5}, 1: {"raise_at": 4}})
    first, _ = run_trials(MockPolicy(), broken, states(0, 1), horizon=400, **IDS)
    fixed = MockEnv({0: {"success_at": 5}, 1: {}})
    retried, _ = run_trials(
        MockPolicy(), fixed, states(0, 1), horizon=400, previous_records=first, retry_state_ids=["state_1"], **IDS
    )
    assert fixed.reset_to_calls == [1]  # only the failed state is re-run, from the same saved state
    assert len(retried) == 3
    assert [(r["state_id"], r["attempt"], r["outcome"]) for r in retried] == [
        ("state_0", 1, "success"),
        ("state_1", 1, "infrastructure_error"),
        ("state_1", 2, "horizon_reached"),
    ]
    summary = summarize_trials(retried, planned=2)
    assert summary["completed"] == 2 and summary["successes"] == 1
    assert summary["infrastructure_errors"] == 0 and summary["incomplete"] is False
    assert summary["attempt_records"] == 3


def test_retry_cannot_introduce_a_new_state():
    env = MockEnv({0: {}})
    with pytest.raises(ValueError, match="state_7"):
        run_trials(MockPolicy(), env, states(0), horizon=400, previous_records=[], retry_state_ids=["state_7"], **IDS)


def test_policy_is_reset_before_every_trial():
    env = MockEnv({0: {"success_at": 2}, 1: {"success_at": 2}, 2: {"success_at": 2}})
    policy = MockPolicy()
    run_trials(policy, env, states(0, 1, 2), horizon=400, **IDS)
    assert policy.starts == 3


def test_interrupted_evaluation_reports_unattempted_trials():
    env = MockEnv({0: {"success_at": 2}, 1: {"success_at": 2}})
    records, _ = run_trials(MockPolicy(), env, states(0, 1), horizon=400, **IDS)
    summary = summarize_trials(records, planned=5)
    assert summary["planned"] == 5 and summary["attempted"] == 2 and summary["completed"] == 2
    assert summary["incomplete"] is True


def test_trajectories_are_recorded_for_later_rendering():
    env = MockEnv({0: {"success_at": 3}})
    _, trajectories = run_trials(MockPolicy(), env, states(0), horizon=400, **IDS)
    trajectory = trajectories[("state_0", 1)]
    assert trajectory["actions"].shape == (3, ACTION_DIM)
    assert trajectory["states"].shape[0] == 4  # initial state plus one per executed step


def write_mock_bank(path, values, role="development"):
    with h5py.File(path, "w") as f:
        f.attrs["role"] = role
        for entry in states(*values):
            group = f.create_group(f"states/{entry['state_id']}")
            group.attrs["seed"] = entry["seed"]
            group.create_dataset("model", data=entry["model"])
            group.create_dataset("states", data=entry["states"])
    return path


def test_evaluate_rollouts_writes_records_matching_the_summary(tmp_path, checkpoint):
    bank = write_mock_bank(tmp_path / "bank.hdf5", [0, 1, 2, 3])
    env = MockEnv({0: {"success_at": 5}, 1: {}, 2: {"raise_at": 2}, 3: {"done_at": 9}})
    out = tmp_path / "out"
    summary = evaluate_rollouts(checkpoint, bank, 400, out, load=lambda path: (MockPolicy(), env))

    with open(out / "rollouts.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert [row["outcome"] for row in rows] == ["success", "horizon_reached", "infrastructure_error", "terminated"]
    assert all(row["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest() for row in rows)
    assert all(row["state_bank_sha256"] == hashlib.sha256(bank.read_bytes()).hexdigest() for row in rows)

    saved = json.loads((out / "results.json").read_text())
    assert saved["summary"] == summary
    assert summary["planned"] == 4 and summary["attempted"] == 4 and summary["completed"] == 3
    assert summary["successes"] == sum(row["outcome"] == "success" for row in rows)
    assert summary["infrastructure_errors"] == 1 and summary["incomplete"] is True
    assert saved["state_bank_role"] == "development"
    assert (out / "trajectories.hdf5").exists()


def test_evaluate_rollouts_refuses_to_overwrite_existing_results(tmp_path, checkpoint):
    bank = write_mock_bank(tmp_path / "bank.hdf5", [0])
    out = tmp_path / "out"
    evaluate_rollouts(checkpoint, bank, 400, out, load=lambda path: (MockPolicy(), MockEnv({0: {}})))
    with pytest.raises(FileExistsError):
        evaluate_rollouts(checkpoint, bank, 400, out, load=lambda path: (MockPolicy(), MockEnv({0: {}})))


def test_evaluate_rollouts_retry_appends_to_existing_records(tmp_path, checkpoint):
    bank = write_mock_bank(tmp_path / "bank.hdf5", [0, 1])
    out = tmp_path / "out"
    broken = MockEnv({0: {"success_at": 5}, 1: {"raise_at": 2}})
    evaluate_rollouts(checkpoint, bank, 400, out, load=lambda path: (MockPolicy(), broken))
    fixed = MockEnv({0: {"success_at": 5}, 1: {"success_at": 8}})
    summary = evaluate_rollouts(
        checkpoint, bank, 400, out, load=lambda path: (MockPolicy(), fixed), retry_state_ids=["state_1"]
    )
    with open(out / "rollouts.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["state_id"], r["attempt"], r["outcome"]) for r in rows] == [
        ("state_0", "1", "success"),
        ("state_1", "1", "infrastructure_error"),
        ("state_1", "2", "success"),
    ]
    assert summary["successes"] == 2 and summary["incomplete"] is False and summary["attempt_records"] == 3


def test_module_exposes_the_planned_interfaces():
    for name in ("score_offline", "create_state_bank", "select_checkpoint", "evaluate_rollouts"):
        assert callable(getattr(evaluate, name))
