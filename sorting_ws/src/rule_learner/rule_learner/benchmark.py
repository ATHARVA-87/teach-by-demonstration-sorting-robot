"""Simulated-teacher benchmark for rule learning (no human, no ROS).

Protocol (decision D7, revision 10), per trial:
1. A hidden rule is drawn uniformly from one family of the *deduplicated*
   hypothesis space (each rule counted under its simplest name).
2. **Opening scene:** one ``sorting_scene`` scene (6-10 items, seeded by the
   trial) is sorted by the simulated teacher. Identical for both strategies.
3. **Next items:** ``active`` asks the learner's highest-information-gain
   question; ``random`` takes a uniformly random item type (the scene
   generator's distribution). This is the *only* difference between them.
4. **Teacher mistakes** use common random numbers: whether label number i of
   trial t is flipped (and to which wrong box) depends only on (t, i, η),
   so both strategies see the same mistakes at the same positions.
5. Each run continues to ``budget`` labels without stopping, recording the
   MAP rule after every label.

Metrics (definitions with formulas in WALKTHROUGH section 6):
- **labels_to_correct** (primary): smallest n such that the MAP rule equals
  the hidden rule on all item types at every label from n to the budget.
  Empty (not reached) if it never settles; reported as a fraction, never
  averaged as a sentinel.
- **labels_at_stop / correct_at_stop:** stop at the first n where
  P(MAP) ≥ stop_confidence, or after opening + max_queries labels (same rule
  for both strategies).
- **accuracy_at_stop:** share of items in 20 held-out scenes (seeds disjoint
  from training) that the stopped rule sorts correctly.
- **Decision-tree baseline (ID3):** trained on the *same* label stream as
  the row it sits in (random rows: the random demos; active rows: the
  learner's questions), at the same label count. Compare it on random rows.

Outputs trials.csv and curves.csv (accuracy after every label) and meta.txt.
Every number quoted in the README comes from these files.
"""

from __future__ import annotations

import argparse
import csv
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sorting_task.rules import describe

from rule_learner import decision_tree
from rule_learner.hypotheses import FAMILIES, HypothesisSpace, Rule
from rule_learner.learner import Demo, LearnerConfig, RuleLearner
from sorting_scene import SceneConfig, default_config_path, generate_scene, load_config

HELD_OUT_SEED_OFFSET = 1_000_000  # evaluation scenes never overlap opening scenes
OPENING_SEED_OFFSET = 10_000


@dataclass(frozen=True)
class TrialResult:
    """One row of trials.csv."""

    setting: str
    family: str
    rule: str
    mode: str
    teacher_noise: float
    epsilon: float
    trial: int
    opening_labels: int
    labels_to_correct: int | None
    labels_at_stop: int
    stopped_confident: bool
    correct_at_stop: bool
    accuracy_at_stop: float
    tree_correct_at_stop: bool
    tree_accuracy_at_stop: float


def _key(*parts: object) -> int:
    """Stable 32-bit seed from values (zlib.crc32; hash() is randomized per process)."""
    return zlib.crc32("|".join(map(str, parts)).encode())


def labels_to_correct(correct_after: Sequence[bool]) -> int | None:
    """Smallest n such that the MAP rule is correct at every label n..len (1-based).

    None if it isn't correct at the last label (never settled within the budget).
    """
    n_ok: int | None = None
    for n in range(len(correct_after), 0, -1):
        if not correct_after[n - 1]:
            break
        n_ok = n
    return n_ok


def teacher_label(
    truth: Rule, item: Mapping[str, str], boxes: Sequence[str], noise: float, trial: int, i: int
) -> str:
    """The simulated teacher's drop for label number ``i`` (common random numbers)."""
    correct = truth.apply(item)
    rng = np.random.default_rng(_key("flip", trial, i, noise))
    if rng.random() < noise:
        return str(rng.choice([b for b in boxes if b != correct]))
    return correct


def _accuracy(
    predict: Callable[[Mapping[str, str]], str], truth: Rule, items: Sequence[Mapping[str, str]]
) -> float:
    return sum(predict(it) == truth.apply(it) for it in items) / len(items)


def run_trial(
    space: HypothesisSpace,
    scene_cfg: SceneConfig,
    truth: Rule,
    mode: str,
    noise: float,
    trial: int,
    held_out: Sequence[Mapping[str, str]],
    cfg: LearnerConfig,
    budget: int = 40,
    setting: str = "main",
    curve: list[dict[str, object]] | None = None,
    stream: list[tuple[dict[str, str], str]] | None = None,
) -> TrialResult:
    """One teaching session against a hidden rule."""
    learner = RuleLearner(space, cfg, seed=_key("query", trial))
    pick_rng = np.random.default_rng(_key("pick", trial))
    types = space.items.types
    truth_idx = space.find(truth)
    opening = [it.attributes() for it in generate_scene(scene_cfg, OPENING_SEED_OFFSET + trial).items]
    examples: list[tuple[Mapping[str, str], str]] = []
    correct_after: list[bool] = []
    stop_n: int | None = None
    stop_rule: Rule | None = None
    stop_tree: decision_tree.Node | None = None
    confident = False
    cap = len(opening) + cfg.max_queries

    for i in range(budget):
        if i < len(opening):
            item: Mapping[str, str] = opening[i]
        elif mode == "active":
            item = learner.best_query().attributes
        else:
            item = types[int(pick_rng.integers(len(types)))]
        box = teacher_label(truth, item, space.boxes, noise, trial, i)
        learner.add(Demo(i + 1, dict(item), box, "ar" if i < len(opening) else "query"))
        examples.append((item, box))
        if stream is not None:
            stream.append((dict(item), box))
        best = learner.best()
        correct_after.append(space.find(best.rule) == truth_idx)
        n = i + 1
        if curve is not None and n >= len(opening):
            tree = decision_tree.fit(examples, space.items.attributes, space.boxes)
            curve.append(
                {
                    "setting": setting,
                    "family": truth.family,
                    "mode": mode,
                    "teacher_noise": noise,
                    "trial": trial,
                    "labels": n,
                    "accuracy": _accuracy(best.rule.apply, truth, held_out),
                    "tree_accuracy": _accuracy(tree.predict, truth, held_out),
                }
            )
        if stop_n is None and n >= len(opening) and (learner.converged() or n >= cap):
            stop_n, stop_rule, confident = n, best.rule, learner.converged()
            stop_tree = decision_tree.fit(examples, space.items.attributes, space.boxes)

    l2c = labels_to_correct(correct_after)
    if stop_n is None or stop_rule is None or stop_tree is None:
        raise ValueError(f"budget {budget} is smaller than the opening scene + max_queries ({cap})")
    return TrialResult(
        setting=setting,
        family=truth.family,
        rule=describe(truth),
        mode=mode,
        teacher_noise=noise,
        epsilon=cfg.epsilon,
        trial=trial,
        opening_labels=len(opening),
        labels_to_correct=l2c,
        labels_at_stop=stop_n,
        stopped_confident=confident,
        correct_at_stop=space.find(stop_rule) == truth_idx,
        accuracy_at_stop=_accuracy(stop_rule.apply, truth, held_out),
        tree_correct_at_stop=all(stop_tree.predict(t) == truth.apply(t) for t in types),
        tree_accuracy_at_stop=_accuracy(stop_tree.predict, truth, held_out),
    )


def sample_truths(space: HypothesisSpace, per_family: int, seed: int, families: Sequence[str]) -> list[Rule]:
    """Hidden rules: ``per_family`` distinct rules drawn uniformly from each deduplicated family."""
    rng = np.random.default_rng(seed)
    out: list[Rule] = []
    for fam in families:
        idx = [i for i, r in enumerate(space.rules) if r.family == fam]
        if not idx:
            continue
        pick = rng.choice(idx, size=min(per_family, len(idx)), replace=False)
        out += [space.rules[int(i)] for i in sorted(pick)]
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI: ``ros2 run rule_learner benchmark --out results/rule_learning``."""
    ap = argparse.ArgumentParser(description="Simulated-teacher rule-learning benchmark.")
    ap.add_argument("--out", default="results/rule_learning")
    ap.add_argument("--rules-per-family", type=int, default=25)
    ap.add_argument("--truth-families", nargs="+", default=[f for f in FAMILIES if f != "constant"])
    ap.add_argument("--teacher-noise", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--held-out-scenes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ablations",
        action="store_true",
        help="also run ε=0.1 (mismatch) and 'no decision-list family' settings at 10%% teacher noise",
    )
    ap.add_argument("--curves", action="store_true", help="log accuracy after every label")
    ap.add_argument("--scene-config", default=str(default_config_path()))
    args = ap.parse_args(argv)

    scene_cfg = load_config(args.scene_config)
    full = HypothesisSpace.build(scene_cfg.attribute_values, scene_cfg.box_labels)
    held_out = [
        it.attributes()
        for s in range(args.held_out_scenes)
        for it in generate_scene(scene_cfg, HELD_OUT_SEED_OFFSET + s).items
    ]
    truths = sample_truths(full, args.rules_per_family, args.seed, args.truth_families)

    settings: list[tuple[str, HypothesisSpace, LearnerConfig, list[float]]] = [
        ("main", full, LearnerConfig(), args.teacher_noise),
    ]
    if args.ablations:
        no_dl = HypothesisSpace.build(
            scene_cfg.attribute_values, scene_cfg.box_labels, [f for f in FAMILIES if f != "decision_list"]
        )
        settings += [
            ("conflict_requery_always", full, LearnerConfig(conflict_requery="always"), [0.10]),
            ("epsilon_0.10", full, LearnerConfig(epsilon=0.10), [0.10]),
            ("no_decision_lists", no_dl, LearnerConfig(), [0.10]),
        ]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[TrialResult] = []
    curve: list[dict[str, object]] = []
    t0 = time.perf_counter()
    for name, space, cfg, noises in settings:
        usable = [t for t in truths if name != "no_decision_lists" or t.family != "decision_list"]
        for noise in noises:
            for trial, truth in enumerate(usable):
                for mode in ("random", "active"):
                    rows.append(
                        run_trial(
                            space,
                            scene_cfg,
                            truth,
                            mode,
                            noise,
                            trial,
                            held_out,
                            cfg,
                            args.budget,
                            name,
                            curve if args.curves and name == "main" else None,
                        )
                    )
    with open(out / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        w.writeheader()
        w.writerows(asdict(r) for r in rows)
    if curve:
        with open(out / "curves.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curve[0]))
            w.writeheader()
            w.writerows(curve)
    meta = {
        "settings": {
            name: {**asdict(cfg), "families": [f for f in FAMILIES if sp.family_sizes[f] > 0]}
            for name, sp, cfg, _ in settings
        },
        "hypotheses": len(full),
        "family_sizes": full.family_sizes,
        "learner": asdict(LearnerConfig()),
        "rules_per_family": args.rules_per_family,
        "truth_families": args.truth_families,
        "seed": args.seed,
        "budget": args.budget,
        "held_out_items": len(held_out),
        "teacher_noise": args.teacher_noise,
        "ablations": args.ablations,
        "trials": len(rows),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "meta.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n")
    print(f"{len(rows)} trials in {meta['seconds']} s -> {out}/trials.csv")


if __name__ == "__main__":
    main()
