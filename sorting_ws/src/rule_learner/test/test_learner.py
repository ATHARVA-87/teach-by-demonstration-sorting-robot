"""Tests for the hypothesis space, Bayesian learner, query selection, contradictions and baseline."""

import numpy as np
import pytest

from rule_learner import decision_tree
from rule_learner.benchmark import TrialResult, run_trial, sample_truths, teacher_label
from rule_learner.hypotheses import FAMILIES, HypothesisSpace, Rule
from rule_learner.learner import Demo, LearnerConfig, RuleLearner
from sorting_scene import default_config_path, load_config

VALUES = {
    "color": ("red", "green", "blue", "yellow"),
    "size": ("small", "medium", "large"),
    "shape": ("cube", "cylinder", "prism"),
}
BOXES = ("A", "B", "C")
COLOR_RULE = Rule.from_parts([({"color": "red"}, "A"), ({"color": "blue"}, "B")], "C", "mapping")
LARGE_RULE = Rule.from_parts([({"size": "large"}, "B")], "A", "one_vs_rest")
CONJ_RULE = Rule.from_parts([({"color": "red", "size": "large"}, "C")], "A", "conjunction")


@pytest.fixture(scope="module")
def space() -> HypothesisSpace:
    return HypothesisSpace.build(VALUES, BOXES)


def item(color: str, size: str, shape: str) -> dict[str, str]:
    return {"color": color, "size": size, "shape": shape}


def _teach(learner: RuleLearner, truth: Rule, items: list[dict[str, str]]) -> None:
    for it in items:
        learner.add(Demo(len(learner.demos) + 1, it, truth.apply(it)))


# --- hypothesis space -----------------------------------------------------------
def test_space_size_and_no_duplicate_behaviors(space: HypothesisSpace) -> None:
    assert len(space) == 1317
    assert space.family_sizes == {
        "constant": 3,
        "one_vs_rest": 60,
        "mapping": 66,
        "conjunction": 198,
        "decision_list": 990,
    }
    assert len({row.tobytes() for row in space.labels}) == len(space)


def test_simplest_description_wins(space: HypothesisSpace) -> None:
    # "red to A, everything else to B" is also a mapping and a decision list; one_vs_rest wins.
    r = Rule.from_parts([({"color": "red"}, "A")], "B", "one_vs_rest")
    assert space.rules[space.find(r)].family == "one_vs_rest"


def test_build_is_deterministic() -> None:
    a = HypothesisSpace.build(VALUES, BOXES)
    b = HypothesisSpace.build(VALUES, BOXES)
    assert a.rules == b.rules


def test_family_filter(space: HypothesisSpace) -> None:
    small = HypothesisSpace.build(VALUES, BOXES, [f for f in FAMILIES if f != "decision_list"])
    assert small.family_sizes["decision_list"] == 0 and len(small) == 1317 - 990
    with pytest.raises(ValueError, match="unknown rule families"):
        HypothesisSpace.build(VALUES, BOXES, ["magic"])


def test_spec_rule_types_are_representable(space: HypothesisSpace) -> None:
    for r in (COLOR_RULE, LARGE_RULE, CONJ_RULE):
        assert space.find(r) >= 0


# --- inference --------------------------------------------------------------------
def test_learns_color_rule(space: HypothesisSpace) -> None:
    learner = RuleLearner(space)
    _teach(learner, COLOR_RULE, [item(c, s, "cube") for c in VALUES["color"] for s in VALUES["size"]])
    best = learner.best()
    assert space.find(best.rule) == space.find(COLOR_RULE) and best.probability > 0.9


def test_posterior_normalized_candidates_sorted(space: HypothesisSpace) -> None:
    learner = RuleLearner(space)
    _teach(learner, LARGE_RULE, [item("red", "large", "cube"), item("red", "small", "cube")])
    assert learner.posterior().sum() == pytest.approx(1.0)
    probs = [c.probability for c in learner.candidates(5)]
    assert probs == sorted(probs, reverse=True)


def test_prior_follows_formula(space: HypothesisSpace) -> None:
    learner = RuleLearner(space, LearnerConfig(prior_lambda=1.0))
    p = np.exp(learner.log_prior)
    assert p.sum() == pytest.approx(1.0)
    c = {"constant": 0, "one_vs_rest": 1, "mapping": 2, "conjunction": 2, "decision_list": 3}
    z = sum(np.exp(-v) for v in c.values())
    i = space.find(LARGE_RULE)
    assert p[i] == pytest.approx(np.exp(-1) / z / space.family_sizes["one_vs_rest"])


def test_simplicity_prior_prefers_simpler_consistent_rule(space: HypothesisSpace) -> None:
    learner = RuleLearner(space)
    _teach(
        learner,
        LARGE_RULE,
        [
            item("red", "large", "cube"),
            item("blue", "small", "prism"),
            item("green", "large", "cylinder"),
            item("yellow", "medium", "cube"),
        ],
    )
    assert learner.best().rule.family == "one_vs_rest"


def test_one_mistake_does_not_destroy_the_right_rule(space: HypothesisSpace) -> None:
    learner = RuleLearner(space, LearnerConfig(epsilon=0.05))
    _teach(learner, COLOR_RULE, [item(c, s, "cube") for c in VALUES["color"] for s in VALUES["size"]])
    learner.add(Demo(99, item("red", "small", "prism"), "C"))  # teacher mistake
    assert space.find(learner.best().rule) == space.find(COLOR_RULE)
    assert 99 in learner.unexplained()


def test_converged_only_at_stop_confidence(space: HypothesisSpace) -> None:
    learner = RuleLearner(space, LearnerConfig(stop_confidence=0.99))
    assert not learner.converged()
    _teach(learner, COLOR_RULE, [item("red", "small", "cube")])
    assert not learner.converged()
    _teach(
        learner,
        COLOR_RULE,
        [item(c, s, sh) for c in VALUES["color"] for s in VALUES["size"] for sh in VALUES["shape"]],
    )
    assert learner.converged() and learner.best().probability >= 0.99


def test_remove_demo_recomputes(space: HypothesisSpace) -> None:
    learner = RuleLearner(space)
    _teach(learner, COLOR_RULE, [item("red", "small", "cube"), item("blue", "small", "cube")])
    before = learner.posterior().copy()
    learner.add(Demo(3, item("green", "small", "cube"), "A"))
    learner.remove(3)
    assert np.allclose(learner.posterior(), before)


def test_unknown_box_or_value_rejected(space: HypothesisSpace) -> None:
    learner = RuleLearner(space)
    with pytest.raises(ValueError, match="box"):
        learner.add(Demo(1, item("red", "small", "cube"), "Z"))
    with pytest.raises(ValueError, match="attributes"):
        learner.add(Demo(1, item("purple", "small", "cube"), "A"))


# --- active learning -------------------------------------------------------------
def test_query_maximizes_information_gain(space: HypothesisSpace) -> None:
    learner = RuleLearner(space)
    _teach(learner, COLOR_RULE, [item("red", "small", "cube"), item("blue", "small", "cube")])
    ig = learner.information_gain()
    q = learner.best_query()
    assert q.reason == "information_gain"
    assert ig[space.items.index(q.attributes)] == pytest.approx(ig.max())


def test_settled_items_are_not_asked(space: HypothesisSpace) -> None:
    learner = RuleLearner(space)
    _teach(learner, COLOR_RULE, [item("red", s, sh) for s in VALUES["size"] for sh in VALUES["shape"]])
    ig = learner.information_gain()
    red = [space.items.index(item("red", s, sh)) for s in VALUES["size"] for sh in VALUES["shape"]]
    assert ig[red].max() < ig.max()


def test_ties_are_broken_by_seeded_rng_not_index(space: HypothesisSpace) -> None:
    picks = {tuple(RuleLearner(space, seed=s).best_query().attributes.values()) for s in range(12)}
    assert len(picks) > 1  # with no data every item ties; the seed decides
    a = RuleLearner(space, seed=3).best_query()
    b = RuleLearner(space, seed=3).best_query()
    assert a == b  # reproducible


def test_query_restricted_to_allowed(space: HypothesisSpace) -> None:
    allowed = [item("green", "small", "cube"), item("yellow", "large", "prism")]
    assert RuleLearner(space).best_query(allowed).attributes in allowed


def test_information_gain_bounds(space: HypothesisSpace) -> None:
    ig = RuleLearner(space).information_gain()
    assert (ig >= 0).all() and (ig <= np.log2(3) + 1e-9).all()


# --- contradictions ----------------------------------------------------------------
def test_contradiction_is_reported_and_asked_next(space: HypothesisSpace) -> None:
    learner = RuleLearner(space)
    learner.add(Demo(3, item("red", "small", "cube"), "A"))
    learner.add(Demo(5, item("blue", "small", "cube"), "B"))
    learner.add(Demo(7, item("red", "small", "cube"), "B"))
    [c] = learner.conflicts()
    assert (c.a, c.b) == (3, 7)
    assert c.message == "Drops 3 and 7 put a small red cube in A and in B. Sort it again or remove one."
    q = learner.best_query()
    assert q.reason == "conflict" and q.attributes == item("red", "small", "cube")
    assert learner.version_space_size() == 0


def test_no_conflict_for_different_items(space: HypothesisSpace) -> None:
    learner = RuleLearner(space)
    learner.add(Demo(1, item("red", "small", "cube"), "A"))
    learner.add(Demo(2, item("red", "large", "cube"), "B"))
    assert learner.conflicts() == []


# --- benchmark protocol -------------------------------------------------------------
def test_teacher_flips_are_common_random_numbers() -> None:
    it = item("red", "small", "cube")
    a = [teacher_label(COLOR_RULE, it, BOXES, 0.5, trial=4, i=i) for i in range(30)]
    b = [teacher_label(COLOR_RULE, it, BOXES, 0.5, trial=4, i=i) for i in range(30)]
    assert a == b and any(x != "A" for x in a)
    assert all(teacher_label(COLOR_RULE, it, BOXES, 0.0, 4, i) == "A" for i in range(30))


def test_run_trial_active_and_random(space: HypothesisSpace) -> None:
    scene_cfg = load_config(default_config_path())
    held_out = [item("red", "small", "cube"), item("blue", "large", "prism")]
    results = [
        run_trial(space, scene_cfg, COLOR_RULE, mode, 0.0, 0, held_out, LearnerConfig(), budget=40)
        for mode in ("random", "active")
    ]
    for r in results:
        assert isinstance(r, TrialResult) and r.opening_labels >= scene_cfg.n_items_min
        assert r.labels_at_stop >= r.opening_labels
        if r.labels_to_correct is not None:
            assert r.labels_to_correct <= 40


def test_truth_sampling_is_stratified_and_seeded(space: HypothesisSpace) -> None:
    a = sample_truths(space, 5, 0, ["mapping", "conjunction"])
    assert a == sample_truths(space, 5, 0, ["mapping", "conjunction"])
    assert [r.family for r in a].count("mapping") == 5


# --- decision-tree baseline -------------------------------------------------------
def test_tree_learns_rule_from_complete_data(space: HypothesisSpace) -> None:
    examples = [(t, COLOR_RULE.apply(t)) for t in space.items.types]
    tree = decision_tree.fit(examples, list(VALUES), BOXES)
    assert all(tree.predict(t) == COLOR_RULE.apply(t) for t in space.items.types)


def test_tree_handles_unseen_values() -> None:
    tree = decision_tree.fit(
        [(item("red", "small", "cube"), "A"), (item("blue", "small", "cube"), "B")], list(VALUES), BOXES
    )
    assert tree.predict(item("green", "small", "cube")) in BOXES


def test_resolved_conflict_is_not_asked_forever(space: HypothesisSpace) -> None:
    learner = RuleLearner(space)
    red = item("red", "small", "cube")
    learner.add(Demo(1, red, "A"))
    learner.add(Demo(2, red, "B"))  # conflict: 1 vs 1
    assert learner.best_query().reason == "conflict"
    learner.add(Demo(3, red, "A"))  # majority A: resolved
    assert learner.best_query().reason == "information_gain"
    assert {(c.a, c.b) for c in learner.conflicts()} == {(1, 2), (2, 3)}  # still reported for the operator


def test_labels_to_correct_definition() -> None:
    from rule_learner.benchmark import labels_to_correct

    assert labels_to_correct([False, True, True, True]) == 2
    assert labels_to_correct([True, False, True, True]) == 3  # must *stay* correct
    assert labels_to_correct([True, True, False]) is None  # wrong at the end: never settled
    assert labels_to_correct([True]) == 1


def test_always_policy_keeps_reasking(space: HypothesisSpace) -> None:
    red = item("red", "small", "cube")
    for policy, expected in (("always", "conflict"), ("unresolved", "information_gain")):
        learner = RuleLearner(space, LearnerConfig(conflict_requery=policy))
        learner.add(Demo(1, red, "A"))
        learner.add(Demo(2, red, "B"))
        learner.add(Demo(3, red, "A"))  # majority A
        assert learner.best_query().reason == expected


def test_benchmark_modes_share_opening_scene_and_mistakes(space: HypothesisSpace) -> None:
    from rule_learner.benchmark import OPENING_SEED_OFFSET
    from sorting_scene import generate_scene

    scene_cfg = load_config(default_config_path())
    opening = [it.attributes() for it in generate_scene(scene_cfg, OPENING_SEED_OFFSET + 7).items]
    streams: dict[str, list] = {"random": [], "active": []}
    results = {
        m: run_trial(
            space, scene_cfg, CONJ_RULE, m, 0.3, 7, opening, LearnerConfig(), budget=40, stream=streams[m]
        )
        for m in ("random", "active")
    }
    n = len(opening)
    assert results["random"].opening_labels == results["active"].opening_labels == n
    # The opening labels each strategy actually received are identical, mistakes included.
    assert streams["random"][:n] == streams["active"][:n]
    assert [it for it, _ in streams["random"][:n]] == opening
    expected = [teacher_label(CONJ_RULE, it, BOXES, 0.3, 7, i) for i, it in enumerate(opening)]
    assert [b for _, b in streams["active"][:n]] == expected


def test_max_conflict_asks_caps_requery(space: HypothesisSpace) -> None:
    learner = RuleLearner(space, LearnerConfig(max_conflict_asks=4))
    red = item("red", "small", "cube")
    for i, b in enumerate(["A", "B", "A", "B"], start=1):  # 2-2 tie after 4 labels
        learner.add(Demo(i, red, b))
    assert learner.unresolved_conflicts(4) == []  # capped: left to the posterior
    assert learner.unresolved_conflicts(5) == [red]
