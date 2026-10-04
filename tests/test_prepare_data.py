import hashlib
import json

import h5py
import numpy as np
import pytest

from scripts.prepare_data import (
    MASK_NAMES,
    REQUIRED_OBS_KEYS,
    AuditError,
    audit_dataset,
    make_split,
    prepare_dataset,
)

OBS_DIMS = {"robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2, "object": 10}


def write_dataset(path, lengths, mutate=None):
    """Write a small robomimic-style file; `mutate(data_group)` injects a defect."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "Lift", "env_version": "1.5.1", "env_kwargs": {}})
        data.attrs["total"] = int(sum(lengths))
        for i, length in enumerate(lengths):
            ep = data.create_group(f"demo_{i}")
            ep.attrs["num_samples"] = length
            ep.create_dataset("actions", data=rng.uniform(-1, 1, size=(length, 7)))
            for key, dim in OBS_DIMS.items():
                ep.create_dataset(f"obs/{key}", data=rng.normal(size=(length, dim)))
        if mutate is not None:
            mutate(data)
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids(n):
    return [f"demo_{i}" for i in range(n)]


# ---------------------------------------------------------------- audit


def test_audit_accepts_clean_dataset(tmp_path):
    path = write_dataset(tmp_path / "clean.hdf5", [5, 8, 3])
    audit = audit_dataset(path)
    assert audit["num_episodes"] == 3
    assert audit["total_samples"] == 16
    assert audit["action_dim"] == 7
    assert audit["obs_shapes"] == {key: [OBS_DIMS[key]] for key in REQUIRED_OBS_KEYS}
    assert audit["source_sha256"] == sha256(path)


def test_audit_reports_missing_key(tmp_path):
    def mutate(data):
        del data["demo_1/obs/object"]

    path = write_dataset(tmp_path / "missing.hdf5", [5, 8, 3], mutate)
    with pytest.raises(AuditError, match=r"demo_1.*obs/object"):
        audit_dataset(path)


def test_audit_reports_missing_actions(tmp_path):
    def mutate(data):
        del data["demo_2/actions"]

    path = write_dataset(tmp_path / "noactions.hdf5", [5, 8, 3], mutate)
    with pytest.raises(AuditError, match=r"demo_2.*actions"):
        audit_dataset(path)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_audit_reports_non_finite_values(tmp_path, bad_value):
    def mutate(data):
        data["demo_0/obs/robot0_eef_pos"][2, 1] = bad_value

    path = write_dataset(tmp_path / "nonfinite.hdf5", [5, 8, 3], mutate)
    with pytest.raises(AuditError, match=r"demo_0.*obs/robot0_eef_pos.*non-finite"):
        audit_dataset(path)


def test_audit_reports_non_finite_actions(tmp_path):
    def mutate(data):
        data["demo_1/actions"][0, 0] = np.nan

    path = write_dataset(tmp_path / "nanaction.hdf5", [5, 8, 3], mutate)
    with pytest.raises(AuditError, match=r"demo_1.*actions.*non-finite"):
        audit_dataset(path)


def test_audit_reports_inconsistent_lengths(tmp_path):
    def mutate(data):
        del data["demo_1/obs/object"]
        data["demo_1"].create_dataset("obs/object", data=np.zeros((7, 10)))

    path = write_dataset(tmp_path / "lengths.hdf5", [5, 8, 3], mutate)
    with pytest.raises(AuditError, match=r"demo_1.*obs/object.*length 7.*8"):
        audit_dataset(path)


def test_audit_reports_zero_length_episode(tmp_path):
    path = write_dataset(tmp_path / "empty.hdf5", [5, 0, 3])
    with pytest.raises(AuditError, match=r"demo_1.*zero-length"):
        audit_dataset(path)


def test_audit_reports_inconsistent_dimensions_across_episodes(tmp_path):
    def mutate(data):
        del data["demo_2/actions"]
        data["demo_2"].create_dataset("actions", data=np.zeros((3, 6)))

    path = write_dataset(tmp_path / "dims.hdf5", [5, 8, 3], mutate)
    with pytest.raises(AuditError, match=r"demo_2.*actions.*shape"):
        audit_dataset(path)


def test_audit_reports_every_defect_not_only_the_first(tmp_path):
    def mutate(data):
        del data["demo_0/obs/object"]
        data["demo_2/actions"][0, 0] = np.inf

    path = write_dataset(tmp_path / "two.hdf5", [5, 8, 3], mutate)
    with pytest.raises(AuditError) as excinfo:
        audit_dataset(path)
    assert "demo_0" in str(excinfo.value) and "demo_2" in str(excinfo.value)


def test_audit_rejects_file_without_episodes(tmp_path):
    path = tmp_path / "nodata.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("data")
    with pytest.raises(AuditError, match="no episodes"):
        audit_dataset(path)


def test_unusual_finite_range_is_a_warning_not_an_error(tmp_path):
    def mutate(data):
        data["demo_0/actions"][0, 0] = 3.5

    path = write_dataset(tmp_path / "range.hdf5", [5, 8, 3], mutate)
    audit = audit_dataset(path)
    assert any("demo_0" in w and "actions" in w for w in audit["warnings"])


def test_audit_does_not_modify_the_file(tmp_path):
    path = write_dataset(tmp_path / "clean.hdf5", [5, 8, 3])
    before = sha256(path)
    audit_dataset(path)
    assert sha256(path) == before


# ---------------------------------------------------------------- split


def test_split_counts_and_partition():
    episode_ids = ids(200)
    split = make_split(episode_ids)
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (160, 20, 20)
    assert (len(train), len(val), len(test)) == (160, 20, 20)
    assert not train & val and not train & test and not val & test
    assert train | val | test == set(episode_ids)


def test_split_is_deterministic_and_independent_of_input_order():
    episode_ids = ids(200)
    assert make_split(episode_ids, seed=42) == make_split(episode_ids, seed=42)
    assert make_split(episode_ids, seed=42) == make_split(list(reversed(episode_ids)), seed=42)


def test_split_depends_on_seed():
    episode_ids = ids(200)
    assert make_split(episode_ids, seed=42) != make_split(episode_ids, seed=43)


def test_split_rejects_duplicates():
    episode_ids = ids(199) + ["demo_0"]
    with pytest.raises(ValueError, match="duplicate"):
        make_split(episode_ids)


@pytest.mark.parametrize("count", [0, 199, 201])
def test_split_rejects_unexpected_counts(count):
    with pytest.raises(ValueError, match="200"):
        make_split(ids(count))


# ---------------------------------------------------------------- prepare


@pytest.fixture
def prepared(tmp_path):
    source = write_dataset(tmp_path / "source.hdf5", [4] * 200)
    paths = {
        "source": source,
        "destination": tmp_path / "processed" / "lift_clean.hdf5",
        "manifest": tmp_path / "artifacts" / "split_manifest.json",
        "audit": tmp_path / "artifacts" / "dataset_audit.json",
    }
    before = sha256(source)
    prepare_dataset(paths["source"], paths["destination"], paths["manifest"], paths["audit"])
    return paths, before


def test_prepare_leaves_source_unchanged(prepared):
    paths, before = prepared
    assert sha256(paths["source"]) == before
    with h5py.File(paths["source"], "r") as f:
        assert "mask" not in f


def test_prepare_writes_disjoint_masks_matching_manifest(prepared):
    paths, before = prepared
    manifest = json.loads(paths["manifest"].read_text())
    assert set(manifest) == {"source_sha256", "split_seed", "train", "val", "test", "version_info"}
    assert manifest["source_sha256"] == before
    assert manifest["split_seed"] == 42
    with h5py.File(paths["destination"], "r") as f:
        masks = {name: [x.decode() for x in f["mask"][mask][:]] for name, mask in MASK_NAMES.items()}
        episodes = set(f["data"].keys())
    for name in ("train", "val", "test"):
        assert masks[name] == manifest[name]
    assert not set(masks["train"]) & set(masks["val"])
    assert not set(masks["train"]) & set(masks["test"])
    assert not set(masks["val"]) & set(masks["test"])
    assert set(masks["train"]) | set(masks["val"]) | set(masks["test"]) == episodes


def test_prepare_copies_data_exactly(prepared):
    paths, _ = prepared
    with h5py.File(paths["source"], "r") as src, h5py.File(paths["destination"], "r") as dst:
        for episode in ("demo_0", "demo_199"):
            assert np.array_equal(src["data"][episode]["actions"][:], dst["data"][episode]["actions"][:])
            for key in REQUIRED_OBS_KEYS:
                assert np.array_equal(src["data"][episode]["obs"][key][:], dst["data"][episode]["obs"][key][:])


def test_prepare_outputs_contain_no_absolute_paths(prepared, tmp_path):
    paths, _ = prepared
    for name in ("manifest", "audit"):
        assert str(tmp_path) not in paths[name].read_text()


def test_prepare_refuses_source_destination_collision(tmp_path):
    source = write_dataset(tmp_path / "source.hdf5", [4] * 200)
    before = sha256(source)
    with pytest.raises(ValueError, match="same file"):
        prepare_dataset(source, tmp_path / "." / "source.hdf5", tmp_path / "m.json", tmp_path / "a.json")
    assert sha256(source) == before


def test_prepare_refuses_collision_through_symlink(tmp_path):
    source = write_dataset(tmp_path / "source.hdf5", [4] * 200)
    link = tmp_path / "link.hdf5"
    link.symlink_to(source)
    before = sha256(source)
    with pytest.raises(ValueError, match="same file"):
        prepare_dataset(source, link, tmp_path / "m.json", tmp_path / "a.json", overwrite=True)
    assert sha256(source) == before


def test_prepare_refuses_unrequested_overwrite(prepared):
    paths, _ = prepared
    before = sha256(paths["destination"])
    with pytest.raises(FileExistsError):
        prepare_dataset(paths["source"], paths["destination"], paths["manifest"], paths["audit"])
    assert sha256(paths["destination"]) == before
    prepare_dataset(paths["source"], paths["destination"], paths["manifest"], paths["audit"], overwrite=True)


def test_prepare_stops_on_audit_failure_without_writing(tmp_path):
    def mutate(data):
        data["demo_3/actions"][0, 0] = np.nan

    source = write_dataset(tmp_path / "source.hdf5", [4] * 200, mutate)
    destination = tmp_path / "out.hdf5"
    with pytest.raises(AuditError):
        prepare_dataset(source, destination, tmp_path / "m.json", tmp_path / "a.json")
    assert not destination.exists()
    assert not (tmp_path / "m.json").exists()


def test_prepare_stops_when_episode_count_is_unexpected(tmp_path):
    source = write_dataset(tmp_path / "source.hdf5", [4] * 150)
    destination = tmp_path / "out.hdf5"
    with pytest.raises(ValueError, match="200"):
        prepare_dataset(source, destination, tmp_path / "m.json", tmp_path / "a.json")
    assert not destination.exists()
