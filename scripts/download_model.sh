#!/usr/bin/env bash
# Download the MediaPipe HandLandmarker model (~7.5 MB) into models/.
# The model is git-ignored, so run this once after cloning:
#     bash scripts/download_model.sh
set -euo pipefail

URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models/hand_landmarker.task"

mkdir -p "$ROOT/models"
if [ -s "$DEST" ]; then
    echo "Model already present: $DEST"
    exit 0
fi

echo "Downloading HandLandmarker model..."
# -f: fail on HTTP errors, -L: follow redirects, --retry: survive flaky networks
curl -fL --retry 3 --connect-timeout 20 -o "$DEST.part" "$URL"
mv "$DEST.part" "$DEST"   # only rename once the download is complete
echo "Saved $DEST ($(du -h "$DEST" | cut -f1))"
