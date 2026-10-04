import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = ["bc_lift_10k", "bc_smoke"]
OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]


def load(name):
    return json.loads((ROOT / "configs" / f"{name}.json").read_text())


@pytest.fixture(params=CONFIG_NAMES)
def config(request):
    return load(request.param)


@pytest.fixture
def protocol():
    return load("protocol_first_evaluation")


def test_algorithm_is_feed_forward_deterministic_bc(config):
    assert config["algo_name"] == "bc"
    for head in ("gaussian", "gmm", "vae", "rnn", "transformer"):
        assert config["algo"][head]["enabled"] is False, head
    assert config["train"]["seq_length"] == 1
    assert config["train"]["frame_stack"] == 1


def test_model_and_loss_match_the_plan(config):
    assert config["algo"]["actor_layer_dims"] == [256, 256]
    assert config["algo"]["loss"] == {"l2_weight": 1.0, "l1_weight": 0.0, "cos_weight": 0.0}
    policy = config["algo"]["optim_params"]["policy"]
    assert policy["optimizer_type"] == "adam"
    assert policy["learning_rate"]["initial"] == 1e-4
    assert policy["learning_rate"]["epoch_schedule"] == []
    assert policy["regularization"]["L2"] == 0.0
    assert config["train"]["batch_size"] == 128


def test_observations_are_low_dimensional_only(config):
    obs = config["observation"]["modalities"]["obs"]
    assert sorted(obs["low_dim"]) == sorted(OBS_KEYS)
    for modality in ("rgb", "depth", "scan"):
        assert obs[modality] == [], modality
    for modality, keys in config["observation"]["modalities"]["goal"].items():
        assert keys == [], modality
    assert config["train"]["goal_mode"] is None


def test_training_uses_only_the_training_mask(config):
    assert config["train"]["hdf5_filter_key"] == "d2p_train"
    assert config["train"]["hdf5_validation_filter_key"] == "d2p_val"
    assert config["experiment"]["validate"] is True
    assert config["train"]["data"] == [{"path": "data/processed/lift_clean.hdf5"}]


def test_supervised_only_no_rollout_driven_training_or_selection(config):
    experiment = config["experiment"]
    assert config["train"]["dataset_keys"] == ["actions"]
    assert experiment["rollout"]["enabled"] is False
    assert experiment["render"] is False
    assert experiment["render_video"] is False
    save = experiment["save"]
    assert save["enabled"] is True
    assert save["on_best_validation"] is False
    assert save["on_best_rollout_return"] is False
    assert save["on_best_rollout_success_rate"] is False
    assert save["every_n_seconds"] is None
    assert save["every_n_epochs"] is None


def test_cpu_first_settings(config):
    assert config["train"]["cuda"] is False
    assert config["train"]["num_data_workers"] == 0


def test_identity_preprocessing(config):
    assert config["train"]["hdf5_normalize_obs"] is False
    assert config["train"]["action_keys"] == ["actions"]
    assert config["train"]["action_config"] == {"actions": {"normalization": None}}


def test_no_external_logging_services(config):
    assert config["experiment"]["logging"]["log_wandb"] is False


def test_paths_are_repository_relative(config):
    assert not Path(config["train"]["output_dir"]).is_absolute()
    assert not Path(config["train"]["data"][0]["path"]).is_absolute()
    assert "/Users/" not in json.dumps(config)


def test_config_is_accepted_by_the_installed_robomimic_schema(config):
    config_factory = pytest.importorskip("robomimic.config").config_factory

    base = config_factory(config["algo_name"])
    with base.values_unlocked():
        base.update(config)  # raises on keys that the installed schema does not define
    assert base.algo.actor_layer_dims == [256, 256]


def test_main_budget_and_checkpoint_schedule():
    config = load("bc_lift_10k")
    experiment = config["experiment"]
    assert experiment["epoch_every_n_steps"] == 100
    assert config["train"]["num_epochs"] == 100
    assert experiment["save"]["epochs"] == [25, 50, 75, 100]
    assert config["train"]["seed"] == 1


def test_smoke_config_is_tiny_and_labelled_diagnostic():
    config = load("bc_smoke")
    updates = config["train"]["num_epochs"] * config["experiment"]["epoch_every_n_steps"]
    assert updates <= 500
    assert "smoke" in config["experiment"]["name"]
    assert config["experiment"]["save"]["epochs"] == [config["train"]["num_epochs"]]


def test_smoke_and_main_differ_only_in_budget_and_name():
    main, smoke = load("bc_lift_10k"), load("bc_smoke")
    for cfg in (main, smoke):
        cfg["experiment"].pop("name")
        cfg["experiment"]["save"].pop("epochs")
        cfg["train"].pop("num_epochs")
    assert main == smoke


def test_protocol_matches_main_config(protocol):
    config = load("bc_lift_10k")
    steps = config["experiment"]["epoch_every_n_steps"]
    training = protocol["training"]
    assert training["seed"] == config["train"]["seed"] == 1
    assert training["updates_per_epoch"] == steps
    assert training["update_budget"] == config["train"]["num_epochs"] * steps
    assert training["checkpoint_updates"] == [e * steps for e in config["experiment"]["save"]["epochs"]]
    assert training["train_mask"] == config["train"]["hdf5_filter_key"]
    assert training["validation_mask"] == config["train"]["hdf5_validation_filter_key"]


def test_protocol_evaluation_rules(protocol):
    dev, final = protocol["development"], protocol["final"]
    assert dev["seeds"] == list(range(10000, 10010))
    assert final["seeds"] == list(range(20000, 20050))
    assert dev["num_states"] == 10 and final["num_states"] == 50
    assert not set(dev["seeds"]) & set(final["seeds"])
    assert protocol["rollout"]["horizon"] == 400
    assert protocol["rollout"]["terminate_on_success"] is True
    assert final["test_mask"] == "d2p_test"
    assert final["used_for_selection"] is False
    assert protocol["selection"]["rule"] == [
        "max development_successes",
        "min validation_mean_episode_mse",
        "min update",
    ]


# ---------------------------------------------------------------- follow-up protocol (longer training)


def test_longer_training_config_changes_only_the_budget():
    baseline, longer = load("bc_lift_10k"), load("bc_lift_50k")
    assert longer["experiment"]["name"] == "bc_lift_50k"
    assert longer["train"]["num_epochs"] == 500
    assert longer["experiment"]["save"]["epochs"] == [125, 250, 375, 500]
    for cfg in (baseline, longer):
        cfg["experiment"].pop("name")
        cfg["experiment"]["save"].pop("epochs")
        cfg["train"].pop("num_epochs")
    assert baseline == longer


def test_follow_up_protocol_matches_its_config():
    protocol, config = load("protocol_follow_up"), load("bc_lift_50k")
    steps = config["experiment"]["epoch_every_n_steps"]
    training = protocol["training"]
    assert training["config"] == "configs/bc_lift_50k.json"
    assert training["seed"] == config["train"]["seed"]
    assert training["update_budget"] == config["train"]["num_epochs"] * steps == 50000
    assert training["checkpoint_updates"] == [e * steps for e in config["experiment"]["save"]["epochs"]]
    assert protocol["selection"]["rule"] == load("protocol_first_evaluation")["selection"]["rule"]
    assert protocol["rollout"] == load("protocol_first_evaluation")["rollout"]


def test_follow_up_final_states_have_never_been_used():
    first, follow_up = load("protocol_first_evaluation"), load("protocol_follow_up")
    used = set(first["development"]["seeds"]) | set(first["final"]["seeds"]) | set(range(11000, 11100))
    final = follow_up["final"]
    assert final["seeds"] == list(range(30000, 30050)) and final["num_states"] == 50
    assert not set(final["seeds"]) & used
    assert not set(final["seeds"]) & set(follow_up["development"]["seeds"])
    assert final["used_for_selection"] is False
    assert final["state_bank"] != first["final"]["state_bank"]
    for earlier in (
        first["development"]["state_bank"],
        first["final"]["state_bank"],
        follow_up["development"]["state_bank"],
    ):
        assert earlier in final["distinct_from"]


def test_follow_up_protocol_discloses_what_informed_it():
    protocol = load("protocol_follow_up")
    assert protocol["development"]["seeds"] == list(range(11000, 11100))
    assert "exploration" in protocol["development"]["note"]
    assert protocol["informed_by"]
    assert protocol["artifacts"] == "artifacts/3_follow_up"
    assert protocol["reference"]["selection"] == "artifacts/1_first_evaluation/selection.json"
