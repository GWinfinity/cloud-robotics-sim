#!/usr/bin/env bash
# 逐个插件目录实跑测试（每目录独立 pytest 进程，满足 gs.init 进程隔离约束）
# 用法: bash scripts/check_plugin_tests.sh [dir ...]   （默认跑白名单候选）
set -u
cd "$(dirname "$0")/.."

DIRS_DEFAULT="
plugins/predictors/articulation_gen/tests
plugins/scenes/wfc_scenes/tests
plugins/scenes/art_scenes/tests
plugins/solvers/thermal/tests
plugins/solvers/joule_heating/tests
plugins/solvers/acoustics/tests
plugins/solvers/cfd_coupling/tests
plugins/teleop/vr_bridge/tests
plugins/envs/maniskill/tests
plugins/do_as_i_do/tests
plugins/manipulation/dexterous_gnn_qp/tests
plugins/controllers/mpc_wbc/tests
plugins/controllers/wbc_lab/tests
plugins/datasets/dreamdojo/tests
plugins/sim2real/sim2real_dexterous/tests
plugins/examples/costream/tests
"

DIRS="${*:-$DIRS_DEFAULT}"
overall=0
for d in $DIRS; do
  if [ ! -d "$d" ]; then
    echo "MISSING $d"
    continue
  fi
  out=$(uv run --no-sync python -m pytest "$d" -q -m "not slow" -p no:cacheprovider --tb=line 2>&1)
  rc=$?
  echo "=== $d (rc=$rc)"
  echo "$out" | grep -E "[0-9]+ (passed|failed|error)|passed|failed|error" | tail -2
  if [ "$rc" -ne 0 ]; then
    echo "RED $d"
    overall=1
  else
    echo "GREEN $d"
  fi
done
echo "OVERALL=$overall"
exit $overall
