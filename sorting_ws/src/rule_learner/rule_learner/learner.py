"""Bayesian concept learning over the rule hypothesis space (decision D7).

Posterior over hypotheses h given demonstrations D = {(x_i, b_i)}:

    P(h | D)  ∝  P(h) * Π_i P(b_i | x_i, h)

Prior (simplicity / Bayesian Occam's razor), |f| = family size after
deduplication:

    P(h) = exp(-λ c_f) / Σ_f' exp(-λ c_f')  *  1 / |f|

Likelihood with teacher noise ε over k boxes:

    P(b | x, h) = 1 - ε        if h(x) == b
                = ε / (k - 1)  otherwise

so one wrong drop lowers a rule's score instead of eliminating it.

Active learning: ask about the item type maximizing expected information
gain about h,

    IG(x) = H( Σ_h P(h|D) P(y|x,h) ) - E_h[ H(y|x,h) ],

where, with symmetric noise, the second term is the same constant for every
x and every h, so IG is the predictive entropy minus a constant.

Stopping: the learner is confident when P(MAP | D) ≥ ``stop_confidence``.
There is deliberately no "no query is worth asking" stop rule: myopic IG is
near zero for every item when a rare exception (a "needle") hasn't been seen
yet, which made the learner stop confidently wrong (D7, revision 1). The
caller caps the number of questions instead and then reports the top rules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from sorting_task.rules import describe

from rule_learner.hypotheses import COMPLEXITY, FAMILIES, HypothesisSpace, Rule


@dataclass(frozen=True)
class LearnerConfig:
    """Learner hyper-parameters (rule_learner YAML; defaults from decision D7)."""

    prior_lambda: float = 1.0  # λ in P(f) ∝ exp(-λ c_f)
    epsilon: float = 0.05  # ε: assumed probability that a drop is a teacher mistake
    stop_confidence: float = 0.99  # confident when P(MAP) ≥ this
    max_queries: int = 20  # question budget per teaching session
    # "unresolved": re-ask a conflicting item only while its labels have no
    # strict majority (default). "always": re-ask while any conflict exists,
    # the original D7 behavior, kept as a benchmark ablation (it wastes the
    # question budget after a single teacher mistake).
    conflict_requery: str = "unresolved"
    max_conflict_asks: int = 4  # stop re-asking an item type after this many labels

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon < 0.5:
            raise ValueError("epsilon must be in (0, 0.5)")
        if self.prior_lambda < 0:
            raise ValueError("prior_lambda must be >= 0")
        if not 0.0 < self.stop_confidence < 1.0:
            raise ValueError("stop_confidence must be in (0, 1)")
        if self.conflict_requery not in ("unresolved", "always"):
            raise ValueError("conflict_requery must be 'unresolved' or 'always'")


@dataclass(frozen=True)
class Demo:
    """One demonstration: an item's attributes and the box it went into."""

    index: int
    attributes: dict[str, str]
    box: str
    source: str = "ar"


@dataclass(frozen=True)
class Candidate:
    """A rule with its posterior probability."""

    rule: Rule
    probability: float
    description: str
    errors: int  # demonstrations this rule disagrees with


@dataclass(frozen=True)
class Conflict:
    """Two demonstrations that no deterministic rule can both satisfy."""

    a: int
    b: int
    attributes: dict[str, str]
    message: str


@dataclass(frozen=True)
class Query:
    """The item to ask about next."""

    attributes: dict[str, str]
    information_bits: float
    reason: str  # "information_gain" | "conflict"


def item_phrase(attributes: Mapping[str, str]) -> str:
    """'small red cube' (size, color, shape)."""
    return " ".join(attributes[a] for a in ("size", "color", "shape") if a in attributes)


@dataclass
class RuleLearner:
    """Incremental Bayesian rule learner."""

    space: HypothesisSpace
    cfg: LearnerConfig = field(default_factory=LearnerConfig)
    demos: list[Demo] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self) -> None:
        k = len(self.space.boxes)
        self._log_hit = math.log(1.0 - self.cfg.epsilon)
        self._log_miss = math.log(self.cfg.epsilon / (k - 1))
        present = [f for f in FAMILIES if self.space.family_sizes.get(f, 0) > 0]
        fam_logp = {f: -self.cfg.prior_lambda * COMPLEXITY[f] for f in present}
        norm = float(np.logaddexp.reduce(list(fam_logp.values())))
        self.log_prior = np.array(
            [
                fam_logp[r.family] - norm - math.log(self.space.family_sizes[r.family])
                for r in self.space.rules
            ]
        )
        self._rng = np.random.default_rng(self.seed)
        self._recompute()

    # --- updates ----------------------------------------------------------------
    def _recompute(self) -> None:
        self.log_lik = np.zeros(len(self.space))
        self.errors = np.zeros(len(self.space), dtype=np.int32)
        for d in self.demos:
            self._apply(d)

    def _apply(self, d: Demo) -> None:
        x = self.space.items.index(d.attributes)
        b = self.space.boxes.index(d.box)
        hit = self.space.labels[:, x] == b
        self.log_lik += np.where(hit, self._log_hit, self._log_miss)
        self.errors += ~hit

    def add(self, demo: Demo) -> None:
        """Add one demonstration (O(H) update).

        Raises:
            ValueError: unknown box or attribute value.
        """
        if demo.box not in self.space.boxes:
            raise ValueError(f"Unknown box '{demo.box}'")
        try:
            self.space.items.index(demo.attributes)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Unknown item attributes {demo.attributes}") from exc
        self.demos.append(demo)
        self._apply(demo)

    def remove(self, index: int) -> None:
        """Remove a demonstration by its index (e.g. the worker deleted it)."""
        self.demos = [d for d in self.demos if d.index != index]
        self._recompute()

    def reset(self) -> None:
        """Forget all demonstrations."""
        self.demos = []
        self._recompute()

    # --- beliefs ------------------------------------------------------------------
    def posterior(self) -> np.ndarray:
        """P(h | D) for every hypothesis (sums to 1)."""
        logp = self.log_prior + self.log_lik
        return np.exp(logp - np.logaddexp.reduce(logp))

    def candidates(self, k: int = 5) -> list[Candidate]:
        """Top-k rules by posterior probability, most likely first."""
        post = self.posterior()
        top = np.argsort(-post, kind="stable")[:k]
        return [
            Candidate(self.space.rules[i], float(post[i]), describe(self.space.rules[i]), int(self.errors[i]))
            for i in top
        ]

    def best(self) -> Candidate:
        """The maximum a posteriori (MAP) rule."""
        return self.candidates(1)[0]

    def version_space_size(self) -> int:
        """Number of hypotheses consistent with every demonstration (the ε → 0 view)."""
        return int((self.errors == 0).sum())

    def converged(self) -> bool:
        """True when the most likely rule reaches the stop confidence."""
        return bool(self.demos) and self.best().probability >= self.cfg.stop_confidence

    # --- active learning --------------------------------------------------------
    def information_gain(self) -> np.ndarray:
        """Expected information gain (bits) of asking about each item type."""
        post = self.posterior()
        k = len(self.space.boxes)
        eps = self.cfg.epsilon
        miss = eps / (k - 1)
        votes = np.stack([(post[:, None] * (self.space.labels == b)).sum(axis=0) for b in range(k)], axis=1)
        pred = miss + (1.0 - eps - miss) * votes  # (X, k) predictive distribution
        h_pred = -(pred * np.log2(pred)).sum(axis=1)
        h_noise = -((1 - eps) * math.log2(1 - eps) + (k - 1) * miss * math.log2(miss))
        return np.maximum(h_pred - h_noise, 0.0)

    def best_query(self, allowed: Sequence[Mapping[str, str]] | None = None) -> Query:
        """The next question.

        An *unresolved* hard conflict is settled first: that item type is
        asked again until its labels have a strict majority (otherwise a
        single teacher mistake would make the learner re-ask it forever).
        Otherwise the item type with maximum information gain; exact ties
        are broken with the learner's seeded RNG (not by index, which would
        bias questions toward the first color/size/shape).
        """
        if self.cfg.conflict_requery == "always":
            unresolved = [c.attributes for c in self.conflicts()]
        else:
            unresolved = self.unresolved_conflicts(self.cfg.max_conflict_asks)
        if allowed is not None:
            unresolved = [a for a in unresolved if a in [dict(x) for x in allowed]]
        if unresolved:
            attrs = unresolved[-1]
            return Query(
                dict(attrs), float(self.information_gain()[self.space.items.index(attrs)]), "conflict"
            )
        ig = self.information_gain()
        types = self.space.items.types
        idx = (
            np.arange(len(types))
            if allowed is None
            else np.array([self.space.items.index(a) for a in allowed])
        )
        best = ig[idx].max()
        ties = idx[np.isclose(ig[idx], best, rtol=0.0, atol=1e-12)]
        choice = int(self._rng.choice(ties))
        return Query(dict(types[choice]), float(ig[choice]), "information_gain")

    # --- contradictions -------------------------------------------------------------
    def conflicts(self) -> list[Conflict]:
        """Pairs of demos with the same item type but different boxes.

        No deterministic rule can satisfy both: one of them is a mistake, or
        the rule depends on something the cell doesn't measure.
        """
        out = []
        for i, a in enumerate(self.demos):
            for b in self.demos[i + 1 :]:
                if a.attributes == b.attributes and a.box != b.box:
                    out.append(
                        Conflict(
                            a.index,
                            b.index,
                            dict(a.attributes),
                            f"Drops {a.index} and {b.index} put a {item_phrase(a.attributes)} "
                            f"in {a.box} and in {b.box}. Sort it again or remove one.",
                        )
                    )
        return out

    def unresolved_conflicts(self, max_asks: int = 4) -> list[dict[str, str]]:
        """Item types whose labels disagree with no strict majority yet.

        Item types already labeled ``max_asks`` times are left to the
        posterior (the noise model weighs them), so a worker who keeps
        disagreeing with themselves isn't asked forever.
        """
        by_type: dict[int, list[Demo]] = {}
        for d in self.demos:
            by_type.setdefault(self.space.items.index(d.attributes), []).append(d)
        out = []
        for demos in by_type.values():
            if len(demos) < 2 or len(demos) >= max_asks:
                continue
            counts = sorted((sum(x.box == b for x in demos) for b in {x.box for x in demos}), reverse=True)
            if len(counts) > 1 and counts[0] == counts[1]:
                out.append(dict(demos[0].attributes))
        return out

    def unexplained(self) -> list[int]:
        """Demo indices the MAP rule disagrees with (shown in Review as "doesn't fit the rule")."""
        if not self.demos:
            return []
        rule = self.best().rule
        return [d.index for d in self.demos if rule.apply(d.attributes) != d.box]
