# DECISIONS

Every project decision, newest last. Each entry has a date, milestone, decision, who made it and how to reverse it.

## Owner overrides
*(Write overrides here. They replace any agent decision below.)*

## Needs owner review
1. **FYI (non-blocking): D8 amends your D4.** The cell runs as our own MuJoCo node with ros2_control-compatible action interfaces, instead of through `mujoco_ros2_control`. Reason: its model is fixed at startup, and benchmarks and datasets need a new randomized scene every episode. `mujoco_ros2_control` stays installed as a later launch option. To keep D4 as written, add an owner override.

**Resolved 2026-10-26 09:55:** the owner ran the apt commands below. Verified:
- MoveIt 2.15.2, moveit_py, mujoco_ros2_control 0.1.2 and ros2_controllers 6.9.0 are installed;
- turtlesim is removed; no ROS packages are orphaned;
- the `apt autoremove` candidates are unrelated old NVIDIA-570 / GTK packages;
- M0 smoke tests pass (see OVERNIGHT_REPORT).

1. ~~**apt install of the simulation/planning stack (needs sudo, blocked overnight).**~~ Done. Commands, for reference:
   ```bash
   # keep the ROS packages that ros-lyrical-desktop pulled in, then remove turtlesim (old project)
   sudo apt-mark manual $(apt-mark showauto | grep '^ros-lyrical-')
   sudo apt remove -y ros-lyrical-turtlesim ros-lyrical-turtlesim-msgs
   # MoveIt 2 + MuJoCo ros2_control (the chosen stack)
   sudo apt install -y ros-lyrical-moveit ros-lyrical-moveit-py ros-lyrical-moveit-resources-panda-moveit-config \
     ros-lyrical-mujoco-ros2-control ros-lyrical-mujoco-ros2-control-demos \
     ros-lyrical-mujoco-ros2-control-plugins ros-lyrical-ros2-controllers
   ```
   - The first command is needed because removing turtlesim also removes the `ros-lyrical-desktop` metapackage. Without it, 290 ROS packages would become "auto-removable" (verified with `apt-get -s`).
   - The install replaces `libomp-20-dev` with `libomp-21-dev`. Check this doesn't affect your HPC/GROMACS builds.
   - Until this runs, the MoveIt/ros2_control smoke test (end of M0) and the MoveIt-based parts of M3/M5 can't be done.

---

## D1. 2026-10-25, M0: Conda env. **Owner decision.**
- **Decision:** use `gesture_ros` (Python 3.14.9), the env from the turtlesim project. No new envs.
- **Evidence:** `conda env list` → base, cmc, gesture_ros, torch. `env_setup.sh` recorded `gesture_ros`. System Python is 3.14.4, so apt-built `rclpy` loads.
- **Reversibility:** n/a (constraint from the owner).

## D2. 2026-10-25, M0: Cleanup scope. **Owner decision.**
- **Decision:** delete everything turtlesim-only. Keep and move reusable code: webcam capture, MediaPipe setup, image utilities, node runner, `Ema`/`Debouncer`, the text renderer, the dashboard server skeleton and fonts. Uninstall packages that the old project installed and the new one doesn't need.
- **Result:**
  - **pip:** nothing removable. Every package is used here or required by mediapipe/colcon/fastapi/uvicorn (`pip show` "Required-by").
  - **apt:** `ros-lyrical-turtlesim` and `turtlesim-msgs` need sudo → moved to "Needs owner review".
  - **Removed:** `~/.ros/log` contents (old run logs, outside the workspace) at 23:30 IST on 2026-10-25, *before* the overnight rules, under the owner's Step 0a answer "Delete everything related to turtle sim project"; also the scratch test images.
- **Reversibility:** workspace files from the owner's zip backup. The `~/.ros/log` deletion is **not reversible** (old run logs only, no project data).

## D3. 2026-10-25, M0: Git history. **Owner decision.**
- **Decision:** fresh repository (`rm -rf .git && git init`). The turtlesim history is only in the owner's zip backup.
- **Reversibility:** restore `.git` from the zip.

## D4. 2026-10-25, M0: Simulation stack. **Owner decision.**
- **Decision:** MuJoCo + MoveIt 2 through `mujoco_ros2_control` (apt binaries for Lyrical).
- **Options considered:** Gazebo Sim 10 + MoveIt 2 (UR5e + Robotiq); MuJoCo with own IK (no MoveIt); Docker Jazzy + Gazebo.
- **Evidence:**
  - apt candidates exist for `ros-lyrical-moveit` 2.15.2, `mujoco-ros2-control` 0.1.2 (bundles MuJoCo 3.12.0), `ros2-controllers` 6.9.0 and `moveit-resources-panda-moveit-config` 3.2.1.
  - pip `mujoco==3.12.0` installed; smoke test: 65× real-time physics, ~1000 fps EGL rendering at 640×480.
  - Gazebo alternative also available: `ros-lyrical-ros-gz` 3.0.10 and `gz-sim-vendor` (Gazebo Sim 10.5.0) are apt candidates.
  - `mujoco_ros2_control` docs confirm `headless`, `sim_speed_factor`, `reset_world` with free-body overrides, `set_free_joint_state`, and camera and free-joint-state plugins.
- **Reversibility:** high until M3. The scene model (`sorting_scene`) is engine-independent.

## D5. 2026-10-25, M0: Robot model. **Agent decision (within D4).**
- **Decision:** Franka Emika Panda with its parallel hand. MoveIt config from `moveit_resources_panda_moveit_config`; MuJoCo model from Google DeepMind's MuJoCo Menagerie (`franka_emika_panda`, Apache-2.0), fetched by script and not committed.
- **Why:** the spec says "whichever has the best support in the chosen stack". The Panda ships with its gripper in both the MoveIt resources (apt) and Menagerie, so no arm + third-party gripper assembly is needed. UR5e would need UR + Robotiq models combined twice (URDF and MJCF).
- **Reversibility:** medium; the planning and grasp code is robot-agnostic behind MoveIt.

## D6. 2026-10-25 23:50, M1/M6: UI design plan (console + AR overlay). **Decision-maker (delegate), confidence high** (medium on the 1080p fit).
- **Decision:** APPROVE `docs/design_plan.md` WITH REVISIONS. The visual direction stays: RAL greys, signal colors only for state, Barlow / Barlow Semi Condensed, ISO 13850 e-stop, corner brackets, "item color is data". The 13 revisions must be applied before any UI/AR code.
- **Required revisions (summary; all applied in design_plan.md rev 2):**
  1. Enforce the color rule strictly. The AR question marker becomes Signal-white dashed brackets with a "?" tag on a dark scrim, not yellow. Conflict notes aren't red. Item chips show a shape glyph plus a color word. Status hues only in the annunciator, lamps, e-stop and log severity (with "Warning:" / "Error:" prefixes).
  2. Contrast:
     - add ink `#1E2124` for text on green/yellow fills (5.47:1 / 8.9:1); white on yellow is banned;
     - green as text uses `#52C474` (4.91:1 on Anthracite);
     - fault fill `#B8281F` with white text (5.27:1);
     - `#A7A9A8` only on Anthracite or darker.
  3. The e-stop is always on its yellow backplate (red vs Graphite is only 2.70:1); label "Emergency stop".
  4. Wireframe the latched e-stop: annunciator "Fault" plus reason, a "Reset" button, modes disabled with a reason. E-stop and watchdog trips map to Fault.
  5. Node lamps in the frame: Camera, Hand tracking, Rule learner, Simulator, Perception, Executor, Safety, each with a text state.
  6. No font glyphs for symbols (the Barlow cmaps lack → ▸ ✓ ✗ ■ ● ▲ ⌜). Inline SVG on the web; Pillow/OpenCV primitives in AR.
  7. Fit a real 1080p screen: viewport about 1280 × 940; verify with a headless screenshot.
  8. Run: active rule sentence and task file at the top; Start/Pause/Resume; recovery sub-states in the step strip; detection callout style (white with dark halo); fix the counter row.
  9. Review: show conflicting/unexplained demos; wireframe the "Correct rule" flow; confirm the saved path after Approve; button hierarchy (primary white fill + ink, secondary 1 px outline).
  10. Teach: error copy says how to fix it; a "Remove" action per demo; candidate-rule bars white on an Anthracite track.
  11. AR:
      - 1 px dark halo on all white strokes;
      - large ≥ 1.5× small diameter;
      - 2 px item outline;
      - box patterns tied to slot index with a defined continuation (A hatch, B dots, C plain, D cross-hatch, E vertical lines);
      - same letter and pattern on the MuJoCo boxes;
      - copy: "Dropped between boxes. Item returned to the table." and "No webcam found. Plug one in or close other apps using it."
  12. Results: headline numbers above the plots; an empty state; plots use white solid vs `#A7A9A8` dashed with markers and a grey→white confusion matrix, never status colors; also write light versions for the README.
  13. RAL 7021 is "Black grey", not Graphite. Rename the token.
- **Options considered:**
  - Approve as written: breaks its own color rule, contrast failures, missing glyphs, missing Run controls and latched e-stop.
  - **Approve with revisions (chosen).**
  - Reject and redesign: unjustified; nothing reads as a generic default, and the owner approved this language before.
- **Why:** follows the skill's process; grounded in inspection-sensor brackets, ISA-101, rotary mode logic and the ISO e-stop. The "item color is data" rule is domain-specific. The problems are details, fixed by the revisions.
- **Evidence:** fetched SKILL.md; WCAG contrast computed with python3 (values above); fontTools cmap check of the 4 bundled TTFs (`tnum` present, the symbol glyphs absent); spec content per mode checked.
- **Reversibility:** high. Write an override under "Owner overrides"; tokens are CSS variables plus one Python palette module.

## D7. 2026-10-26 00:50, M2: Rule-learning approach. **Decision-maker (delegate), confidence high** (medium on exact parameter values, to be confirmed by the M2 benchmark).
- **Decision:** Option A. Bayesian concept learning over the enumerated, deduplicated, interpretable hypothesis space (1,317 hypotheses), with:
  - a family simplicity prior;
  - a symmetric label-noise likelihood;
  - expected-information-gain queries;
  - every rule compiled to a canonical decision list, which is also the task-file format.

  Version space = the zero-error special case (an empty version space triggers contradiction reporting). The ID3 decision tree is the benchmark baseline only. Proposal: `docs/rule_learning_proposal.md`.
- **Required revisions:**
  1. Remove the "max IG < 0.05 bits" stop rule. It caused premature wrong stops (the needle-in-a-haystack failure of myopic IG). Stop when P(MAP) ≥ threshold or `max_queries` is reached; at the cap, say "still unsure between X and Y" and show the top 2. Never stop silently.
  2. Stop threshold 0.99, not 0.95.
  3. λ = 1.0 (configurable). Prior: P(h) = e^(−λ c_f) / Σ_f' e^(−λ c_f') · 1/|f|, with |f| after deduplication.
  4. ε = 0.05. Benchmark teacher flip rates η ∈ {0, 0.05, 0.1, 0.2} with ε fixed, plus one ε-mismatch ablation (ε = 0.1).
  5. Keep the decision-list family behind a YAML `families:` list; run "without decision lists" as an ablation.
  6. Disjunction wording: when both clauses go to the same box, write "Blue items or cubes to A, everything else to B." (with a test).
  7. Deterministic deduplication ties: (complexity, family order, number of clauses, attribute/value order as in scene.yaml), not text order. Vocabulary always from scene.yaml.
  8. Queries:
     - any of the 36 types (re-querying allowed);
     - ties broken with a seeded RNG;
     - the query item is placed without overlap; sent on `/teach/query`.
  9. Contradictions:
     - hard pair message, e.g. "Drops 3 and 7 put a small red cube in A and in B. Sort it again or remove one."; that item type becomes the next query;
     - when the version space is empty, list the drops the MAP rule disagrees with;
     - always report the version-space size.
  10. Fair active-vs-random benchmark:
      - same learner, ε and threshold; truths stratified by family from the deduplicated space;
      - the same opening scene per trial (one generated scene, sorted by the teacher);
      - common random numbers for teacher flips;
      - only the next-item choice differs (IG vs uniform over 36).

      Metrics:
      - **Primary:** labels until MAP is correct *and stays correct* up to N = 40 (run without stopping); median, IQR, fraction not reached; no sentinel averaging.
      - **Secondary:** labels at stop and P(correct | stopped); accuracy vs labels; generalization on fresh scenes; ID3 on the same random streams.
  11. Task file:
      - fields `schema_version`, `created_at`, attribute vocabulary, box labels (no coordinates), `rule` (decision list), `description`, `family` (informational), `on_unknown_item: "skip_and_report"`;
      - `provenance` (confidence, demos, queries, ε, λ, threshold, competing rules, unexplained demos, `approved: true`);
      - pydantic v2 `extra="forbid"` + cross-field validators (vocabulary, boxes, no duplicate clauses), JSON Schema exported to `docs/task_file_schema.json`;
      - the model must be importable by `sorting_execution` without depending on `rule_learner`;
      - pin `pydantic==2.14.1`.
- **Options considered:** A (chosen); B, decision tree (no posterior, competitors or principled queries; overfits; baseline only); C, pure version space (one mistake empties it; kept as the zero-error view).
- **Evidence (decision-maker's own read-only simulations):**
  - Hypothesis counts: 1,317 (3 / 60 / 66 / 198 / 990); pydantic 2.14.1 installed; no scikit-learn.
  - λ=1, threshold 0.95, with the IG stop: conjunctions only 66–87% correct at stop.
  - λ=1, threshold 0.99, no IG stop: conjunctions 97% correct (no noise), 96% (10% noise), 76% (20% noise); other families ≥ 99%.
  - λ=2: conjunctions 17–23%.
- **Reversibility:** high. All parameters are YAML; the engine sits behind the hypothesis table; the task-file schema is versioned.

### D7-A1. 2026-10-26 13:50, M2: amendment to D7 revision 9 (conflict re-asking). **Builder, based on benchmark evidence.**
- **Change:** a hard conflict's item type is re-asked only while its labels have **no strict majority** (at most `max_conflict_asks` = 4 labels), instead of whenever any conflict exists.
- **Why:** with teacher mistakes, the original policy re-asked the same item for the rest of the session. One wrong drop leaves a permanent conflict, so the question budget was wasted.
- **Evidence:** ablation `conflict_requery_always` vs main, `results/rule_learning/summary.md`, regenerated by `scripts/run_rule_benchmark.sh`. Conjunction rules at 10% teacher mistakes, active learning:
  - never settled within 40 labels: **40% → 0%**;
  - correct when it stops: **60% → 96%**.

  Decision lists: 32% → 0% and 68% → 100%.
- **Also added (from the M2 review):**
  - `ask_after_demos` (default: the scene's minimum size `n_items.min` = 6), so the worker sorts at least that many opening items before the first question (the benchmark waits for the whole 6–10-item opening scene, so live runs ask a little earlier on larger scenes);
  - `query_timeout_s: 60` (re-send the *same* unanswered question; only the answer with that id clears it; no budget refund);
  - unique question ids;
  - `/teach/query` volatile instead of latched.
- **Reversibility:** set `conflict_requery: always` in `rule_learner.yaml`.

## D8. 2026-10-26 14:45, M3: Simulated-cell architecture (**amends D4**). **Decision-maker (delegate), confidence high** (medium on details M3 must measure).
- **Decision:** option C (proposal: `docs/m3_sim_proposal.md`).
  - `sorting_sim` is **our own Python MuJoCo node**, rebuilding the scene per episode with `MjSpec` (Menagerie Panda + `sorting_scene` items).
  - It exposes **exactly the ros2_control interfaces**:
    - `/joint_states` (`panda_joint1..7`, `panda_finger_joint1/2`);
    - `/panda_arm_controller/follow_joint_trajectory` (FollowJointTrajectory);
    - `/panda_hand_controller/gripper_cmd` (**ParallelGripperCommand**);
    - `/clock`, the overhead camera Image + CameraInfo;
    - ground truth only on `/sim/debug/item_poses`; `~/reset_scene`.
  - `mujoco_ros2_control` becomes a later optional backend (`sim_backend:=ros2_control`).
  - D4 now reads: "MuJoCo + MoveIt 2; cell = own MuJoCo node with ros2_control-compatible action interfaces".
- **Why not A:**
  - `mujoco_ros2_control` loads one fixed MJCF at startup. Its services are ResetWorld / SetFreeJointState / SetPause / StepSimulation / ApplyExternalWrench; none change geoms, colors, lights or the number of items.
  - The spec needs 200+ different randomized scenes and domain-randomized datasets.
- **Required (checklist):**
  1. D4 amendment + owner FYI (done above); WALKTHROUGH §8 / README stack updated.
  2. Interface contract in YAML, mirroring ros2_control names; our own `moveit_controllers.yaml` declares ParallelGripperCommand.
  3. Contract test: MoveIt executes a planned trajectory through our action server, final joint error within tolerance.
  4. Robot model:
     - Menagerie loaded via `MjSpec.from_file` and renamed in code (no edits to Menagerie files);
     - fetch script → `third_party/`;
     - FK consistency URDF vs MJCF (`panda_link8` and TCP within 1 mm / 0.5° over ~100 configurations; finger coupling).
  5. Grasps:
     - yaw candidates from shape symmetry (cube 2 face pairs, cylinder 8–12, prism 3 min-width axes);
     - finger-sweep clearance filter (open to item width + 15–20 mm);
     - IK filter; grasp at mid-height;
     - separate pick success / place success / sorting accuracy metrics (ground truth only for scoring);
     - run-time grasp check from the gripper result.
  6. Planning:
     - Pilz PTP for transfers (OMPL RRTConnect fallback); Pilz LIN for 10 cm approach/retreat;
     - collision objects (table, box walls, other items); attach the item to `panda_hand` during transport;
     - log the planner per segment; plan with `moveit_py`;
     - first step: a Pilz LIN smoke test through moveit_py (fallback `/compute_cartesian_path`).
  7. Infeasible items:
     - plan the whole chain before moving; skip and report with reason codes (`outside_workspace`, `no_collision_free_grasp`, `no_ik`, `plan_failed`, `place_unreachable`);
     - report infeasible separately from planning failures;
     - test fixture with an out-of-reach item;
     - reachability sweep (box C at 0.72 m is near the top-down limit; measure it).
  8. Risks: EGL context per thread (render on one thread); MoveIt execution-duration monitoring vs sim time; physics stepping independent of the rclpy executor.
  9. Follow-up after M5: `sim_backend:=ros2_control` demo through `mujoco_ros2_control` + JTC, reusing contract test #3 (or document as future work).
- **Evidence:**
  - read-only checks of `mujoco_ros2_control` services/strings, the Panda MoveIt config (moveit_controllers FollowJointTrajectory; ros2_controllers parallel_gripper_action_controller; gripper_moveit_controllers GripperCommand mismatch; Pilz configs present);
  - imports OK in env (control_msgs actions, MoveItPy, rclpy.action);
  - `MjSpec` attach/compile verified in mujoco 3.12.
- **Reversibility:** high. Owner override → A. The action/topic contract (+ contract test) means only the sim backend changes.

### D8-A1. 2026-10-26, M3 (builder, from implementation + review): robot model names
- **Change to checklist #4:** Menagerie's names are **kept**, not renamed in code. The ROS interface maps them instead (`JOINT_NAME_MAP`: `joint1..7` → `panda_joint1..7`, `finger_joint1/2` → `panda_finger_joint1/2`).
- **Why:** Menagerie's contact exclusions and gripper tendon refer to its own body and joint names. Renaming them made the model fail to compile ("body 'link0' not found in bodypair"; WALKTHROUGH troubleshooting #34).
- **Evidence:** `sorting_sim/test/test_fk_consistency.py`. Over 100 seeded configurations, MoveIt's URDF and Menagerie agree for `panda_link8`, `panda_hand`, the TCP and both fingers, worst 1e-6 mm / 5e-6°; the finger coupling is checked too.
- **Reversibility:** trivial (the mapping is one dictionary).

### D8-A2. 2026-10-26, M3 (builder, from measurements): grasp, place and motion details
- **Release by drop:** items are released with the fingertips 1.5 cm above the tray walls, not lowered inside.
  - Why: an open hand around a 52 mm item is up to 108 mm wide, and the tray opening is 122 mm, so off-centre slots would hit the walls. Dropping also makes any wrist angle valid.
  - Cost: items can bounce and stack. A full bin stacking items counts as sorted.
- **LIN fallback:** a LIN that violates joint limits near the base is retried slower, then replaced by a short PTP to the same pose.
  - This deviates from "LIN for approach/retreat" only as a fallback. The planner of every segment is logged, so the rate is measurable.
- **Gripper strength:** the simulated gripper servo is 3× Menagerie's.
  - Why: Menagerie's gives ~1.4 N squeeze, while a real Franka hand gives up to 70 N; at 10×, prism edges penetrated the pads.
  - It's a `sim.yaml` parameter (`physics.gripper_stiffness_scale`).
- **Executor input:** ground truth re-read before every pick, after the scene settles. This mirrors a camera snapshot per pick, which is what M4's perception will do.
- **Grasp check:** "empty" = finger gap below half the item width. It's a run-time check from the gripper's own state; no ground truth.
- **Evidence:** WALKTHROUGH troubleshooting #59–#68; grasp physics sweeps; batch runs in OVERNIGHT_REPORT.
- **Reversibility:** each is a parameter or one function.
