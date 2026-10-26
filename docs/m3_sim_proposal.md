# M3: simulated cell + scripted pick-and-place, proposal for decision

## Goal (spec, M3)
- A simulated cell: Franka Panda + parallel gripper, table, boxes A–C, an overhead RGB camera, and randomized items from `sorting_scene`.
- A **scripted** pick-and-place that uses **ground-truth** item positions (perception replaces ground truth in M4).
- MoveIt 2 plans every motion (decision D4).
- Later needs (M5/M7): run 200+ randomized scenes headless, faster than real time, with fast resets.

## Facts checked
- **Installed:**
  - MoveIt 2.15.2 + moveit_py: smoke-tested, Panda plans in 4–14 ms;
  - `mujoco_ros2_control` 0.1.2 (bundles MuJoCo 3.12): the demo ran headless at 50 Hz with camera topics; it has `reset_world`, `set_free_joint_state`, `sim_speed_factor` and `headless`;
  - pip `mujoco==3.12.0` (same version): 65× real-time physics, about 1000 fps EGL rendering.
- **Robot models:**
  - MoveIt uses `moveit_resources_panda_description` (joints `panda_joint1..7`, `panda_finger_joint1/2`).
  - MuJoCo Menagerie's `franka_emika_panda` (Apache-2.0) has joints `joint1..7` and fingers `finger_joint1/2`, coupled by an equality constraint and driven by a tendon actuator (0–255).
  - Both come from Franka's published kinematics. A forward-kinematics consistency test between the URDF and the MJCF is needed either way.
- `mujoco_ros2_control` reads a URDF `<ros2_control>` block plus an MJCF path. ros2_control joint names must match the MJCF joint names. Grippers need a position-actuated tendon. Its URDF→MJCF converter is marked "highly experimental".

## Options

### A. `mujoco_ros2_control` + MoveIt `move_group` (the D4 plan, as stated)
- MJCF cell = Menagerie Panda with joints renamed to MoveIt's names + table/boxes/items generated from `sorting_scene`.
- URDF `<ros2_control>` block with the `MujocoSystemInterface`; `joint_trajectory_controller` + gripper controller; MoveIt executes through the `FollowJointTrajectory` action.
- **Pros:** the same controller stack as a real Panda (ros2_control); strongest "real robot path" story; scene resets via services.
- **Cons / risks:**
  - young package (0.1.2) with few examples;
  - two robot descriptions coupled through name-matching;
  - gripper tendon wiring;
  - camera through a plugin; resets go through ROS services (slower per episode);
  - the most integration time.

### B. Own MuJoCo simulation node (Python, pip `mujoco`) with the same ROS interfaces
- `sorting_sim` node owns the MjModel/MjData. It publishes `/joint_states`, `/clock`, the overhead camera (`sensor_msgs/Image` + `CameraInfo`) and ground-truth item poses on a separate debug topic, which perception never reads.
- It offers `FollowJointTrajectory` (arm) and `GripperCommand` (hand) **action servers**: the same interfaces MoveIt and a real Panda driver use.
- MoveIt (`moveit_py` or `move_group`) plans; the executor sends trajectories to those actions.
- **Pros:**
  - full control over stepping (faster than real time for benchmarks), resets (direct `mj_resetData` + item placement) and domain randomization (lighting, textures, colors, camera noise; needed in M4);
  - one Python process, easy to test headless;
  - the executor talks to standard ROS actions, so swapping in option A or a real robot later is a launch change.
- **Cons:** we write the trajectory-following control (PD position actuators + interpolation, about 150 lines), and the "ros2_control in the loop" story moves to "future work / option A as a launch alternative".

### C. B now, A as a later launch option
B for M3–M7 (benchmarks, randomization). Keep the action interfaces identical, so A can be added later if time permits.

## Builder's recommendation
**C.** The spec's hard requirements (200+ randomized headless episodes, domain-randomized synthetic datasets, ground-truth-free perception, reliable grasps) depend on fast, fully controllable stepping and resets. Option B delivers that with standard ROS action interfaces, so the robot-facing code is identical to what would drive a real Panda. Option A's main benefit, the real ros2_control stack, can be added later without touching the executor.

## Sub-decisions to confirm
1. **Robot model:** Menagerie Panda MJCF with joints renamed to MoveIt names (`panda_joint1..7`, `panda_finger_joint1/2`), fetched by a script into `third_party/` (not committed), plus a URDF↔MJCF forward-kinematics consistency test.
2. **Grasps:**
   - top-down along the item's minimum-width axis (cube: an edge pair; cylinder: any; triangle prism: the flat side vs apex);
   - pre-grasp 10 cm above → descend → close → lift → move above the box → descend → open;
   - success = the item is inside the box footprint at rest (from ground truth, for evaluation only).
3. **Planning:**
   - MoveIt OMPL (RRTConnect) for free-space moves;
   - Pilz LIN (Cartesian straight lines) for approach/retreat;
   - table, box walls and other items as collision objects.
4. **Unreachable or infeasible items** (IK or plan failure, or outside the workspace) are reported and skipped, never attempted blindly.
