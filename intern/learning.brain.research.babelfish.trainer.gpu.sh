#!/bin/bash
#
# Script for running babelfish jobs on borg using GPUs.
#
# E.g.,
#
# Run in iowa (P100).
# $ gpu.sh --model=<model> --name=<name> --cell=jn --build --cmd=reload

source gbash.sh || exit 1
source module google3/learning/brain/research/babelfish/trainer/launcher_lib.sh

set -e

# GPU-only flags.
DEFINE_bool sync "true" \
    "Uses sync replica trainer. " \
    "The --cfg flag must also be set to a corresponding borg config."
DEFINE_int workers 8 "# workers"
DEFINE_int gpus 2 "# gpus per training worker"
DEFINE_int worker_cpus 4 "# cpus per training worker"
DEFINE_int ps_replicas 1 "# ps"
DEFINE_int ps_cpus 8 "# cpus per PS task"
DEFINE_bool assert "true" "Enable asserts."
DEFINE_bool check_numerics "true" "Enable check numerics."
DEFINE_array gpu_build_opts --delim=" " --default="--copt=-mavx2 --copt=-mfma --config=cuda"  "Extra blaze build opts."
DEFINE_string model_task_names "" \
    "Task names in the format of ['task_1','task_2']. For multi-task training."

# GPU resource flags.
DEFINE_string controller_ram "8G" "Controller RAM."
DEFINE_string trainer_client_ram "8G" "Trainer client RAM."
DEFINE_string gpu_worker_ram "12G" "GPU worker ram"


# Builds the binaries for GPU jobs.
# This function overrides FLAGS_build_opts with FLAGS_gpu_build_opts and calls
# a helper function to build GPU binaries.
# @flags {FLAGS_tpu_build_opts}
function run_gpu_build() {
  [[ "${FLAGS_gpu_build_opts[*]}" =~ "--config=cuda" ]] \
      || gbash::die "Binaries for GPU training must be built with --config=cuda"

  local -a build_opts=("${FLAGS_gpu_build_opts[@]}")
  run_build learning/brain/research/babelfish/trainer:{tf_server,trainer.par}
}


# Building vars specifically for GPU jobs.
# @flag {All flags in this file}
function get_gpu_vars() {
  if (( FLAGS_assert )); then
    ASSERT_ARG="enable_asserts=true"
  else
    ASSERT_ARG="enable_asserts=false"
  fi
  if (( FLAGS_check_numerics )); then
    ASSERT_ARG="$ASSERT_ARG,enable_check_numerics=true"
  else
    ASSERT_ARG="$ASSERT_ARG,enable_check_numerics=false"
  fi

  # Build up the --vars.
  local VARS="${ASSERT_ARG}"
  VARS="${VARS},worker_replicas=${FLAGS_workers},worker_gpus=${FLAGS_gpus}"
  VARS="${VARS},worker_num_gpus_per_split=${FLAGS_split}"
  VARS="${VARS},ps_replicas=${FLAGS_ps_replicas},ps_cpus=${FLAGS_ps_cpus}"
  VARS="${VARS},dp_replicas=0"

  if (( FLAGS_sync )); then
    VARS="${VARS},rpc_layer=rpc2"
  fi

  if [[ -n "${FLAGS_model_task_names}" ]]; then
    VARS="${VARS},model_task_names=${FLAGS_model_task_names}"
  fi

  # Smaller setup.
  VARS="${VARS},worker_cpus=${FLAGS_worker_cpus},worker_ram=${FLAGS_gpu_worker_ram}"
  VARS="${VARS},controller_cpus=4,controller_ram=${FLAGS_controller_ram}"
  if (( FLAGS_sync )); then
    VARS="${VARS},trainer_client_cpus=2,trainer_client_ram=${FLAGS_trainer_client_ram}"
  fi
  echo "$VARS"
}


function main() {
  # GPU-related checks.
  CHECK_GT "${FLAGS_gpus}" 0

  run_gpu_build
  local cfg="$(get_cfg "gpu")"
  echo "Using borg config: ${cfg}"
  local VARS="$(get_common_vars),$(get_gpu_vars)"


  if (( FLAGS_allocs )); then
    BORGCFG=/google/data/ro/teams/traino/borgcfg
  else
    BORGCFG=borgcfg
  fi
  "$BORGCFG" \
    "learning/brain/research/babelfish/${cfg}" \
    --skip_confirmation \
    --strict_cell_confirmation_threshold=2 \
    --vars="${VARS}" \
    ${FLAGS_cmd}
}

gbash::main "$@"
