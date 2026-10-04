# Recorded results

Per-trial records (CSV and JSON) behind every number in the docs. Datasets, checkpoints and
full trajectories are not included. Folders are numbered in the order the work was done.

| Folder | What it holds | Protocol | Described in |
| --- | --- | --- | --- |
| top level | data audit, episode split, software environment | | `docs/results.md`, "Setup" |
| `1_first_evaluation/` | the 10,000-update model: checkpoint selection, final results, failure analysis; `timing_experiment/` holds the timing control and shifted-label models | `protocol_first_evaluation.json` | "Main result" (first single-model run) |
| `2_diagnosis/` | why grasps were missed, on 100 exploration states, and the network variants tried | none (diagnostic only) | "Why grasps fail", "What fixed it" |
| `3_follow_up/` | the 50,000-update model confirmed on new initial states, with the first model on the same states (`reference_10k/`) | `protocol_follow_up.json` | "What fixed it" |
| `4_multi_seed/` | six conditions, five training seeds each, one shared set of initial states: the headline result | `protocol_multi_seed.json` | "Main result" |
| `videos/` | the two clips shown in the README, with `index.json` recording which trial each shows | | README |

## Files that recur

| File | Contents |
| --- | --- |
| `rollouts.csv` | One row per simulator trial: initial state, outcome, steps, checkpoint and state-bank hashes |
| `results.json` | Summary of those trials, plus held-out action error where measured |
| `selection.json` | Checkpoint candidates and the one chosen, with the evidence used |
| `validation_scores.json` | Held-out action error for every scheduled checkpoint |
| `run_record.json` | Training run: settings, runtime, versions and checkpoint hashes |
| `training_curves.json` | Loss per epoch, extracted from the training log |
| `state_bank_*.json` | Checks that saved initial states restore exactly and duplicate no earlier state |
| `development/update_*/` | Per-checkpoint trials on development states, the raw records behind `selection.json` |

Records written by `scripts/analyze.py`, one per analysis quoted in the results:

| File | Contents |
| --- | --- |
| `1_first_evaluation/failure_analysis.json` | how each failure of the first evaluation ended |
| `2_diagnosis/grasp_diagnosis.json` | gripper position at the grasp, for the first model and the human |
| `2_diagnosis/validation_vs_success.json` | held-out action error against success, for the long exploration runs |
| `2_diagnosis/state_bank_exploration.json` | checks on the 100 exploration starting positions |
| `3_follow_up/aiming_error.json` | aiming error of both models on the follow-up final states |
| `videos/index.json` | which trial each video shows, and the rule that chose it |

`python -m scripts.report` recomputes every count from `rollouts.csv` and fails on any
mismatch. `make reproduce` regenerates this folder elsewhere and `scripts/compare.py` diffs it.
