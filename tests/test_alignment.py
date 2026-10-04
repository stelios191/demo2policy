import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.alignment import make_alignment_pairs, write_alignment_datasets

ROOT = Path(__file__).resolve().parents[1]
OBS_KEYS = ["robot0_eef_pos", "object"]
MANIFEST = {"train": ["demo_0", "demo_1"], "val": ["demo_2"], "test": ["demo_3"]}
LENGTHS = {"demo_0": 6, "demo_1": 9, "demo_2": 5, "demo_3": 7}


def sentinel(episode_index, length, width):
    """Value at time t encodes both the episode and t, so misplaced labels are detectable."""
    column = episode_index * 1000 + np.arange(length, dtype=float)
    return np.repeat(column[:, None], width, axis=1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "clean.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "Lift"})
        data.attrs["total"] = sum(LENGTHS.values())
        for index, (episode_id, length) in enumerate(LENGTHS.items()):
            ep = data.create_group(episode_id)
            ep.attrs["num_samples"] = length
            ep.attrs["model_file"] = "<xml/>"
            ep.create_dataset("actions", data=sentinel(index, length, 7))
            ep.create_dataset("states", data=sentinel(index, length, 4))
            ep.create_dataset("rewards", data=np.zeros(length))
            for key in OBS_KEYS:
                ep.create_dataset(f"obs/{key}", data=sentinel(index, length, 3) + 0.5)
                ep.create_dataset(f"next_obs/{key}", data=sentinel(index, length, 3) + 0.75)
        for name, mask in (("train", "d2p_train"), ("val", "d2p_val"), ("test", "d2p_test")):
            f.create_dataset(f"mask/{mask}", data=np.array(MANIFEST[name], dtype="S"))
    return path


@pytest.fixture
def derived(tmp_path, source):
    aligned, shifted = tmp_path / "aligned.hdf5", tmp_path / "shifted.hdf5"
    before = sha256(source)
    write_alignment_datasets(source, MANIFEST, aligned, shifted, offset=3)
    return source, aligned, shifted, before


# ---------------------------------------------------------------- pairs


def test_sentinel_pairs_for_six_steps():
    obs = {"a": np.arange(6, dtype=float)[:, None], "b": np.arange(6, dtype=float)[:, None] * 10}
    actions = np.arange(6, dtype=float)[:, None]
    obs_out, aligned, shifted = make_alignment_pairs(obs, actions, offset=3)
    assert obs_out["a"][:, 0].tolist() == [0, 1, 2]
    assert obs_out["b"][:, 0].tolist() == [0, 10, 20]
    assert aligned[:, 0].tolist() == [0, 1, 2]
    assert shifted[:, 0].tolist() == [3, 4, 5]


@pytest.mark.parametrize("length", [1, 2, 3])
def test_episode_no_longer_than_offset_is_rejected(length):
    obs = {"a": np.zeros((length, 1))}
    with pytest.raises(ValueError, match="offset"):
        make_alignment_pairs(obs, np.zeros((length, 1)), offset=3)


def test_pairs_reject_mismatched_lengths_and_bad_offsets():
    with pytest.raises(ValueError, match="length"):
        make_alignment_pairs({"a": np.zeros((5, 1))}, np.zeros((6, 1)), offset=3)
    with pytest.raises(ValueError, match="positive"):
        make_alignment_pairs({"a": np.zeros((6, 1))}, np.zeros((6, 1)), offset=0)


def test_pairs_do_not_modify_inputs():
    obs = {"a": np.arange(6, dtype=float)[:, None]}
    actions = np.arange(6, dtype=float)[:, None]
    obs_copy, actions_copy = obs["a"].copy(), actions.copy()
    make_alignment_pairs(obs, actions, offset=3)
    assert np.array_equal(obs["a"], obs_copy) and np.array_equal(actions, actions_copy)


# ---------------------------------------------------------------- derived datasets


def test_shift_never_crosses_episode_boundaries(derived):
    _, aligned, shifted, _ = derived
    with h5py.File(aligned, "r") as a, h5py.File(shifted, "r") as s:
        for index, episode_id in enumerate(MANIFEST["train"]):
            length = LENGTHS[episode_id] - 3
            expected_time = np.arange(length, dtype=float)
            aligned_labels = a["data"][episode_id]["actions"][:, 0]
            shifted_labels = s["data"][episode_id]["actions"][:, 0]
            assert aligned_labels.tolist() == (index * 1000 + expected_time).tolist()
            assert shifted_labels.tolist() == (index * 1000 + expected_time + 3).tolist()
            assert np.all(shifted_labels // 1000 == index)


def test_conditions_have_identical_observations_and_sample_counts(derived):
    _, aligned, shifted, _ = derived
    with h5py.File(aligned, "r") as a, h5py.File(shifted, "r") as s:
        for episode_id in MANIFEST["train"]:
            ep_a, ep_s = a["data"][episode_id], s["data"][episode_id]
            assert set(ep_a["obs"]) == set(ep_s["obs"]) == set(OBS_KEYS)
            for key in OBS_KEYS:
                assert np.array_equal(ep_a["obs"][key][:], ep_s["obs"][key][:])
                assert ep_a["obs"][key].shape[0] == LENGTHS[episode_id] - 3
            assert ep_a["actions"].shape == ep_s["actions"].shape
            assert ep_a.attrs["num_samples"] == ep_s.attrs["num_samples"] == LENGTHS[episode_id] - 3
        assert a["data"].attrs["total"] == s["data"].attrs["total"]


def test_sample_count_metadata_is_updated(derived):
    _, aligned, _, _ = derived
    with h5py.File(aligned, "r") as f:
        assert f["data"].attrs["total"] == sum(LENGTHS.values()) - 3 * len(MANIFEST["train"])
        assert f["data"].attrs["total"] == sum(f["data"][e]["actions"].shape[0] for e in f["data"])
        assert f.attrs["d2p_offset_steps"] == 3
        assert "not" in f.attrs["d2p_usage"] and "replay" in f.attrs["d2p_usage"]


def test_validation_and_test_episodes_are_unchanged(derived):
    source, aligned, shifted, _ = derived
    with h5py.File(source, "r") as src:
        for path in (aligned, shifted):
            with h5py.File(path, "r") as out:
                for episode_id in MANIFEST["val"] + MANIFEST["test"]:
                    for name in ("actions", "states", "rewards"):
                        assert np.array_equal(src["data"][episode_id][name][:], out["data"][episode_id][name][:])
                    for key in OBS_KEYS:
                        assert np.array_equal(
                            src["data"][episode_id]["obs"][key][:], out["data"][episode_id]["obs"][key][:]
                        )
                    assert out["data"][episode_id].attrs["num_samples"] == LENGTHS[episode_id]


def test_train_episodes_keep_only_supervised_arrays(derived):
    _, aligned, shifted, _ = derived
    for path in (aligned, shifted):
        with h5py.File(path, "r") as f:
            for episode_id in MANIFEST["train"]:
                assert set(f["data"][episode_id].keys()) == {"actions", "obs"}


def test_masks_and_environment_metadata_are_preserved(derived):
    source, aligned, shifted, _ = derived
    with h5py.File(source, "r") as src:
        for path in (aligned, shifted):
            with h5py.File(path, "r") as out:
                for mask in ("d2p_train", "d2p_val", "d2p_test"):
                    assert out["mask"][mask][:].tolist() == src["mask"][mask][:].tolist()
                assert out["data"].attrs["env_args"] == src["data"].attrs["env_args"]
                assert set(out["data"].keys()) == set(src["data"].keys())


def test_source_is_unchanged(derived):
    source, _, _, before = derived
    assert sha256(source) == before


def test_short_training_episode_stops_everything(tmp_path, source):
    with h5py.File(source, "r+") as f:
        episode = f["data/demo_1"]
        del episode["actions"]
        episode.create_dataset("actions", data=np.zeros((3, 7)))
        for key in OBS_KEYS:
            del episode[f"obs/{key}"]
            episode.create_dataset(f"obs/{key}", data=np.zeros((3, 3)))
    aligned, shifted = tmp_path / "aligned.hdf5", tmp_path / "shifted.hdf5"
    with pytest.raises(ValueError, match="demo_1"):
        write_alignment_datasets(source, MANIFEST, aligned, shifted, offset=3)
    assert not aligned.exists() and not shifted.exists()


def test_manifest_must_match_the_masks_in_the_source(tmp_path, source):
    wrong = {"train": ["demo_0", "demo_2"], "val": ["demo_1"], "test": ["demo_3"]}
    with pytest.raises(ValueError, match="manifest"):
        write_alignment_datasets(source, wrong, tmp_path / "a.hdf5", tmp_path / "s.hdf5", offset=3)


def test_outputs_are_not_overwritten_and_cannot_collide(tmp_path, derived):
    source, aligned, shifted, _ = derived
    with pytest.raises(FileExistsError):
        write_alignment_datasets(source, MANIFEST, aligned, shifted, offset=3)
    with pytest.raises(ValueError, match="distinct"):
        write_alignment_datasets(source, MANIFEST, tmp_path / "x.hdf5", tmp_path / "x.hdf5", offset=3)
    with pytest.raises(ValueError, match="distinct"):
        write_alignment_datasets(source, MANIFEST, source, tmp_path / "y.hdf5", offset=3)


# ---------------------------------------------------------------- matched configs


def test_matched_configs_differ_from_baseline_only_in_name_and_data():
    baseline = json.loads((ROOT / "configs" / "bc_lift_10k.json").read_text())
    expected_data = {
        "bc_matched_aligned_10k": "data/processed/lift_matched_aligned.hdf5",
        "bc_shift_plus_3_10k": "data/processed/lift_shift_plus_3.hdf5",
    }
    for name, data_path in expected_data.items():
        config = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        assert config["experiment"]["name"] == name
        assert config["train"]["data"] == [{"path": data_path}]
        config["experiment"]["name"] = baseline["experiment"]["name"]
        config["train"]["data"] = baseline["train"]["data"]
        assert config == baseline
