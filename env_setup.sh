# env_setup.sh — set up a terminal for the teach-by-demonstration sorting cell.
#
# Usage (must be *sourced*, not executed, so the changes stay in your shell):
#     source env_setup.sh
#
# Order matters:
#   1. conda activate  -> puts the env's python 3.14 first on PATH.
#   2. source ROS      -> layers ROS's PYTHONPATH / LD_LIBRARY_PATH / AMENT paths
#                         on top, without replacing the env's python.
#   3. source overlay  -> our own built workspace (sorting_ws/install), if built.
# Doing ROS first and conda second can leave a mix of interpreters on PATH.

# Directory this script lives in (works no matter where you source it from).
_SR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SR_ENV="gesture_ros"
_SR_DISTRO="lyrical"

# 1. conda env ---------------------------------------------------------------
# 'conda activate' only works after conda's shell hook is loaded.
if ! command -v conda >/dev/null 2>&1; then
    echo "[env_setup] conda not found. Install Miniconda or add it to PATH." >&2
    return 1
fi
eval "$(conda shell.bash hook)"
conda activate "$_SR_ENV" || {
    echo "[env_setup] Conda env '$_SR_ENV' is missing. Create it with:" >&2
    echo "            conda env create -f $_SR_ROOT/environment.yml" >&2
    return 1
}
# Editors (e.g. the VS Code Python extension) can inject *another* env's bin/
# at the front of PATH behind conda's back, so 'python' would still point
# elsewhere. Force this env's bin/ to the front and verify.
export PATH="$CONDA_PREFIX/bin:$PATH"
# Drop *other* conda envs' bin/ dirs from PATH (same cause). Otherwise tools
# like cmake can resolve to another env's copy and pick up its Python/libs
# during colcon builds.
_SR_CONDA_ENVS="$(dirname "$CONDA_PREFIX")"
PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
  | awk -v envs="$_SR_CONDA_ENVS/" -v mine="$CONDA_PREFIX/bin" \
      'index($0, envs) != 1 || $0 == mine' | paste -sd: -)"
export PATH
# Editors can also export VIRTUAL_ENV for another env. CMake's FindPython3
# checks VIRTUAL_ENV *first*, so message-package builds would use that env's
# Python (seen: "No module named 'em'"). Clear it unless it is this env.
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$CONDA_PREFIX" ]; then
    unset VIRTUAL_ENV
fi
hash -r
if [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
    echo "[env_setup] 'python' resolves to $(command -v python), not $CONDA_PREFIX/bin/python." >&2
    return 1
fi

# 2. ROS 2 -------------------------------------------------------------------
if [ ! -f "/opt/ros/$_SR_DISTRO/setup.bash" ]; then
    echo "[env_setup] ROS 2 $_SR_DISTRO is not installed at /opt/ros/$_SR_DISTRO." >&2
    echo "            See README.md > Setup for the install commands." >&2
    return 1
fi
source "/opt/ros/$_SR_DISTRO/setup.bash"

# 3. Our workspace overlay (only exists after 'colcon build') --------------------
if [ -f "$_SR_ROOT/sorting_ws/install/setup.bash" ]; then
    source "$_SR_ROOT/sorting_ws/install/setup.bash"
fi

# Where ar_teach finds the MediaPipe hand model (scripts/download_model.sh).
export SORTING_HAND_MODEL="$_SR_ROOT/models/hand_landmarker.task"
if [ ! -f "$SORTING_HAND_MODEL" ]; then
    echo "[env_setup] Hand model missing. Download it with: bash $_SR_ROOT/scripts/download_model.sh" >&2
fi

# Simulator robot model (scripts/fetch_robot_models.sh).
export SORTING_PANDA_XML="$_SR_ROOT/third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
if [ ! -f "$SORTING_PANDA_XML" ]; then
    echo "[env_setup] Robot model missing (the simulator and its tests need it). Fetch it with: bash $_SR_ROOT/scripts/fetch_robot_models.sh" >&2
fi

# MuJoCo renders the robot camera off-screen on the GPU through EGL (no window
# needed). Measured: ~1000 fps at 640x480 on the RTX 4050.
export MUJOCO_GL=egl

# Sanity check: ROS's compiled Python modules are built for the system python's
# minor version. The env must match it or 'import rclpy' fails.
_SR_SYS_PY="$(/usr/bin/python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
_SR_ENV_PY="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
if [ "$_SR_SYS_PY" != "$_SR_ENV_PY" ]; then
    echo "[env_setup] WARNING: env python $_SR_ENV_PY != system python $_SR_SYS_PY; rclpy will not import." >&2
fi

echo "[env_setup] Ready: conda env '$_SR_ENV' (python $_SR_ENV_PY) + ROS 2 $ROS_DISTRO"
unset _SR_ROOT _SR_ENV _SR_DISTRO _SR_SYS_PY _SR_ENV_PY _SR_CONDA_ENVS
