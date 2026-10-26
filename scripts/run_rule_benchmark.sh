#!/usr/bin/env bash
# Regenerate the rule-learning benchmark, its plots, and the README figures.
#   bash scripts/run_rule_benchmark.sh          (~30 s on a laptop CPU)
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source env_setup.sh >/dev/null
ros2 run rule_learner benchmark --out results/rule_learning --rules-per-family 25 --ablations --curves
ros2 run rule_learner plot_results --in results/rule_learning
mkdir -p docs/img
for f in labels_vs_noise accuracy_curves ablation; do cp "results/rule_learning/${f}_light.png" "docs/img/rule_${f}.png"; done
echo "Summary: results/rule_learning/summary.md; README figures: docs/img/rule_*.png"
