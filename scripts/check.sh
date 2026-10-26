#!/usr/bin/env bash
# Run the full quality gate: ruff, mypy, colcon build, tests.
#   bash scripts/check.sh                 # all packages
#   bash scripts/check.sh rule_learner    # one package
# Exits non-zero on the first failing step.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source env_setup.sh >/dev/null
PKG="${1:-}"
SRC="sorting_ws/src"
if [ -n "$PKG" ]; then TARGETS=("$SRC/$PKG"); SEL=(--packages-select "$PKG"); else
  mapfile -t TARGETS < <(find "$SRC" -maxdepth 2 -name package.xml -printf '%h\n' | sort); SEL=(); fi

echo "== ruff"; python -m ruff check "${TARGETS[@]}"; python -m ruff format --check "${TARGETS[@]}"
echo "== mypy"
for t in "${TARGETS[@]}"; do
  mod="$t/$(basename "$t")"
  if [ -d "$mod" ]; then (cd "$t" && python -m mypy --config-file "$ROOT/mypy.ini" "$(basename "$t")"); fi
done
echo "== colcon build"
mkdir -p sorting_ws/log
if ! (cd sorting_ws && python -m colcon build --symlink-install "${SEL[@]}" > log/check_build.log 2>&1); then
  cat sorting_ws/log/check_build.log; echo "== colcon build FAILED (full log above)"; exit 1
fi
tail -2 sorting_ws/log/check_build.log
# shellcheck disable=SC1091
source sorting_ws/install/setup.bash
echo "== tests"
if [ ! -f "$ROOT/third_party/mujoco_menagerie/franka_emika_panda/panda.xml" ]; then
  echo "!! WARNING: robot model missing: every simulator and motion test will be SKIPPED." >&2
  echo "!!          Run: bash scripts/fetch_robot_models.sh" >&2
fi
for t in "${TARGETS[@]}"; do
  if [ -d "$t/test" ]; then (cd "$t" && python -m pytest test -q -p no:cacheprovider); fi
done
echo "== all checks passed"
