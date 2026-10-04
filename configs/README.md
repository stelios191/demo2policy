# Configurations

| File | Purpose |
| --- | --- |
| `bc_lift_10k.json` | full episodes, 10,000 updates: the first model |
| `bc_lift_50k.json` | identical apart from 50,000 updates: the final model |
| `bc_matched_aligned_{10k,50k}.json` | timing control, at both training lengths |
| `bc_shift_plus_3_{10k,50k}.json` | shifted labels, at both training lengths |
| `bc_smoke.json` | 300-update check that training runs; its outputs are discarded |
| `exploration/` | variants tried while diagnosing the missed grasps |
| `protocol_first_evaluation.json` | evaluation rules for the first evaluation |
| `protocol_follow_up.json` | evaluation rules for the follow-up |
| `protocol_multi_seed.json` | evaluation rules for the multi-seed study |

Each protocol was fixed before its training run; a later file rename is recorded in its
`revisions` (no rule changed).

## Differences from robomimic's `bc.json` template (commit `d309eae`)

| Setting | Upstream template | This project |
| --- | --- | --- |
| `algo.actor_layer_dims` | `[1024, 1024]` | `[256, 256]` |
| `experiment.name` | `"test"` | `"bc_lift_10k"` |
| `experiment.render_video` | `true` | `false` |
| `experiment.rollout.enabled` | `true` | `false` |
| `experiment.save.epochs` | `[]` | `[25, 50, 75, 100]` |
| `experiment.save.every_n_epochs` | `50` | `null` |
| `experiment.save.on_best_rollout_success_rate` | `true` | `false` |
| `experiment.validate` | `false` | `true` |
| `train.batch_size` | `100` | `128` |
| `train.cuda` | `true` | `false` |
| `train.data` | `null` | `[{"path": "data/processed/lift_clean.hdf5"}]` |
| `train.dataset_keys` | `["actions", "rewards", "dones"]` | `["actions"]` |
| `train.hdf5_filter_key` | `null` | `"d2p_train"` |
| `train.hdf5_validation_filter_key` | `null` | `"d2p_val"` |
| `train.num_epochs` | `2000` | `100` |
| `train.output_dir` | `"../bc_trained_models"` | `"runs"` |

- Rollouts during training are off, so simulator outcomes cannot influence checkpoint saving;
  checkpoints are saved on a fixed schedule and selected by `scripts/evaluate.py` under the protocol.
- The logged validation loss uses 10 sampled batches; the reported held-out error uses every sample.
- Paths are relative to the repository root.
