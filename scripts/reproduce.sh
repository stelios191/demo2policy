#!/usr/bin/env bash
# Reproduce recorded results into a separate folder, then compare them with the committed results.
#
#   scripts/reproduce.sh [--quick | --full] [OUTPUT_DIR]
#
#   --quick   the first model only, end to end (about 5 minutes)
#   (none)    every result except the long exploration runs (about 47 minutes)
#   --full    everything, including the three long exploration runs
#
# OUTPUT_DIR defaults to ./reproduction and must be empty. The committed results are never
# touched. Independent stages run side by side; each writes its log to OUTPUT_DIR/logs/.
# Set PYTHON to choose the interpreter (default: python).
set -euo pipefail
cd "$(dirname "$0")/.."

SCOPE=core
OUTPUT=reproduction
for arg in "$@"; do
  case "$arg" in
    --quick) SCOPE=quick ;;
    --full) SCOPE=full ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) OUTPUT="$arg" ;;
  esac
done
if [ -d "$OUTPUT" ] && [ -n "$(ls -A "$OUTPUT")" ]; then
  echo "error: $OUTPUT is not empty; choose a new folder or remove it" >&2
  exit 1
fi
mkdir -p "$OUTPUT/logs"
export D2P_OUT="$(cd "$OUTPUT" && pwd)"
PY="${PYTHON:-python}"
RAW=data/raw/lift/ph/low_dim_v15.hdf5
RAW_SHA256=2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540
START=$SECONDS

say() { printf '== [%s min] %s\n' "$(( (SECONDS - START) / 60 ))" "$*"; }
latest() {
  local found; found=$(ls -d "$D2P_OUT/runs/$1"/*/ 2>/dev/null | tail -1)
  [ -n "$found" ] || { echo "error: no run found under runs/$1" >&2; return 1; }
  echo "${found%/}"
}
# Background stages ignore Ctrl-C in a script, so on any exit stop every stage still running,
# together with the processes it started.
STAGE_PIDS=()
kill_tree() {
  local child
  for child in $(pgrep -P "$1"); do kill_tree "$child"; done
  kill "$1" 2>/dev/null || true
}
stop_stages() {
  local pid
  for pid in "${STAGE_PIDS[@]+"${STAGE_PIDS[@]}"}"; do kill_tree "$pid"; done
}
trap stop_stages EXIT
trap 'exit 130' INT TERM
run() { "$PY" "$@"; }

# Run the named stage functions side by side; report the log of any that fails.
in_parallel() {
  local pids=() failed=0
  for stage in "$@"; do
    ( set -e; "$stage" ) > "$D2P_OUT/logs/$stage.log" 2>&1 &
    pids+=($!)
    STAGE_PIDS+=($!)
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "error: stage ${*:$((i + 1)):1} failed; end of $D2P_OUT/logs/${*:$((i + 1)):1}.log:" >&2
      tail -25 "$D2P_OUT/logs/${*:$((i + 1)):1}.log" >&2
      failed=1
    fi
  done
  [ "$failed" = 0 ] || exit 1
}

dataset() {
  if [ ! -f "$RAW" ]; then
    run -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim \
      --download_dir "$PWD/data/raw"
  fi
  run -c "from scripts.common import sha256_file as h; import sys; sys.exit(h('$RAW') != '$RAW_SHA256')" \
    || { echo "$RAW does not match the expected checksum"; exit 1; }
  run -m scripts.prepare_data --source "$RAW"
  PYTHON="$(command -v "$PY")" scripts/check_env.sh --json > "$D2P_OUT/artifacts/environment.json"
}

first_model() {
  run -m scripts.train --config configs/bc_lift_10k.json
  local run_dir; run_dir=$(latest bc_lift_10k)
  run -m scripts.evaluate state-bank --role development --run-dir "$run_dir"
  run -m scripts.evaluate score-validation --run-dir "$run_dir"
  run -m scripts.evaluate develop --run-dir "$run_dir"
  run -m scripts.evaluate state-bank --role final --run-dir "$run_dir"
  run -m scripts.evaluate final
  run -m scripts.analyze failures
}

timing_experiment() {
  run -m scripts.alignment
  for condition in matched_aligned shift_plus_3; do
    run -m scripts.train --config "configs/bc_${condition}_10k.json"
    local run_dir; run_dir=$(latest "bc_${condition}_10k")
    run -m scripts.evaluate --tag "$condition" score-validation --run-dir "$run_dir"
    run -m scripts.evaluate --tag "$condition" develop --run-dir "$run_dir"
    run -m scripts.evaluate --tag "$condition" final
  done
}

diagnosis() {
  run -m scripts.analyze exploration-bank
  run -m scripts.analyze exploration-baseline
  local run_dir; run_dir=$(latest bc_lift_10k)
  run -m scripts.explore --run-dir "$run_dir" --binary-gripper --updates 10000
  run -m scripts.analyze grasp-diagnosis
}

follow_up() {
  run -m scripts.train --config configs/bc_lift_50k.json
  local run_dir protocol=configs/protocol_follow_up.json; run_dir=$(latest bc_lift_50k)
  run -m scripts.evaluate --protocol $protocol state-bank --role development --run-dir "$run_dir"
  run -m scripts.evaluate --protocol $protocol score-validation --run-dir "$run_dir"
  run -m scripts.evaluate --protocol $protocol develop --run-dir "$run_dir"
  run -m scripts.evaluate --protocol $protocol state-bank --role final --run-dir "$run_dir"
  run -m scripts.evaluate --protocol $protocol final
  run -m scripts.evaluate --protocol $protocol final --selection artifacts/1_first_evaluation/selection.json --name reference_10k
  run -m scripts.analyze follow-up-aiming
}

explore_one() {
  run -m scripts.train --config "configs/exploration/$1.json"
  local run_dir; run_dir=$(latest "$1")
  run -m scripts.explore --run-dir "$run_dir"
}
explore_mlp256() { explore_one explore_mlp256_200k; }
explore_mlp1024() { explore_one explore_mlp1024_200k; }
explore_gmm1024() { explore_one explore_gmm1024_200k; }

multi_seed() {
  run -m scripts.multiseed run
  run -m scripts.multiseed summarize
}

finish() {
  run -m scripts.report --collect
  if [ "$SCOPE" != quick ]; then run -m scripts.analyze media; fi
  if [ "$SCOPE" = full ]; then
    run -m scripts.plot_exploration
    run -m scripts.analyze validation-vs-success
  fi
}

say "dataset";                       in_parallel dataset
say "first model";                   in_parallel first_model
if [ "$SCOPE" != quick ]; then
  say "timing experiment, diagnosis and follow-up, side by side"
  in_parallel timing_experiment diagnosis follow_up
  if [ "$SCOPE" = full ]; then
    say "long exploration runs, side by side"
    in_parallel explore_mlp256 explore_mlp1024 explore_gmm1024
  fi
  say "multi-seed study: 30 models";  in_parallel multi_seed
fi
say "charts, videos and GIFs";        in_parallel finish
say "comparison with the committed results"
run -m scripts.compare "$D2P_OUT" --scope "$SCOPE"
