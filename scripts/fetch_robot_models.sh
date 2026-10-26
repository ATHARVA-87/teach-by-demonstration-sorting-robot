#!/usr/bin/env bash
# Fetch the Franka Panda MJCF model from Google DeepMind's MuJoCo Menagerie
# (Apache-2.0) into third_party/ (git-ignored). Pinned to one commit so the
# simulated robot is reproducible.
#   bash scripts/fetch_robot_models.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/mujoco_menagerie"
COMMIT="f3b6013629c7b94f8ee93fd977b161fa8350f9bc"   # 2026-10-23
if [ -f "$DEST/franka_emika_panda/panda.xml" ]; then
    HAVE="$(git -C "$DEST" rev-parse HEAD 2>/dev/null || echo unknown)"
    if [ "$HAVE" = "$COMMIT" ]; then
        echo "Panda model already present at the pinned commit: $DEST/franka_emika_panda"
        exit 0
    fi
    echo "third_party/mujoco_menagerie is at $HAVE, expected $COMMIT; checking out the pinned commit."
    git -C "$DEST" fetch -q origin "$COMMIT"
    git -C "$DEST" checkout -q "$COMMIT"
    exit 0
fi
mkdir -p "$ROOT/third_party"
git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie "$DEST"
git -C "$DEST" sparse-checkout set franka_emika_panda
git -C "$DEST" checkout -q "$COMMIT"
echo "Fetched MuJoCo Menagerie @ $COMMIT -> $DEST/franka_emika_panda (license: $DEST/franka_emika_panda/LICENSE)"
