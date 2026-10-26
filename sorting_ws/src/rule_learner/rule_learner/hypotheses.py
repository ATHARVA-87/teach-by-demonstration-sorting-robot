"""Hypothesis space: every interpretable sorting rule over the item attributes.

A hypothesis is a total function *item type -> box*, written as a decision
list (``sorting_task.rules.Rule``). Five families of increasing complexity
are enumerated (constant, one value vs rest, single-attribute mapping,
two-attribute conjunction, two-clause decision list), evaluated on every item
type, and rules with identical behavior are merged into one hypothesis that
keeps the simplest description.

For 4 colors x 3 sizes x 3 shapes (36 item types) and 3 boxes this gives
1,317 hypotheses (3 / 60 / 66 / 198 / 990 per family): small enough to
enumerate exactly, so inference is exact Bayesian updating over a table.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sorting_task.rules import COMPLEXITY, FAMILIES, Clause, Rule, describe

__all__ = ["COMPLEXITY", "FAMILIES", "Clause", "HypothesisSpace", "ItemSpace", "Rule", "describe"]


@dataclass(frozen=True)
class ItemSpace:
    """All item types: the Cartesian product of the attribute values."""

    attributes: tuple[str, ...]
    values: tuple[tuple[str, ...], ...]

    @staticmethod
    def from_values(values: Mapping[str, Sequence[str]]) -> ItemSpace:
        """Build from ``{"color": (...), "size": (...), "shape": (...)}`` (order kept)."""
        return ItemSpace(tuple(values), tuple(tuple(v) for v in values.values()))

    @property
    def types(self) -> list[dict[str, str]]:
        """Every item type as an attribute dict, in a fixed order."""
        return [dict(zip(self.attributes, combo, strict=True)) for combo in itertools.product(*self.values)]

    def index(self, item: Mapping[str, str]) -> int:
        """Index of an item type in :attr:`types`.

        Raises:
            KeyError / ValueError: if an attribute or value is unknown.
        """
        idx = 0
        for name, vals in zip(self.attributes, self.values, strict=True):
            idx = idx * len(vals) + vals.index(item[name])
        return idx

    def __len__(self) -> int:
        n = 1
        for v in self.values:
            n *= len(v)
        return n


def _literals(space: ItemSpace) -> list[tuple[str, str]]:
    return [(a, v) for a, vals in zip(space.attributes, space.values, strict=True) for v in vals]


def _enumerate(space: ItemSpace, boxes: Sequence[str]) -> Iterator[Rule]:
    """All rules of all families (with duplicate behaviors)."""
    lits = _literals(space)
    for b in boxes:
        yield Rule((), b, "constant")
    for lit in lits:
        for b1, b2 in itertools.permutations(boxes, 2):
            yield Rule((Clause((lit,), b1),), b2, "one_vs_rest")
    for a, vals in zip(space.attributes, space.values, strict=True):
        for assignment in itertools.product(boxes, repeat=len(vals)):
            if len(set(assignment)) < 2:
                continue
            # Canonical form: the most frequent box is the default ("others");
            # ties go to the first box in slot order.
            default = max(boxes, key=lambda bx: (assignment.count(bx), -boxes.index(bx)))
            clauses = tuple(
                Clause(((a, v),), bx) for v, bx in zip(vals, assignment, strict=True) if bx != default
            )
            yield Rule(clauses, default, "mapping")
    for (a1, v1), (a2, v2) in itertools.combinations(lits, 2):
        if a1 == a2:
            continue
        cond = tuple(sorted(((a1, v1), (a2, v2))))
        for b1, b2 in itertools.permutations(boxes, 2):
            yield Rule((Clause(cond, b1),), b2, "conjunction")
    for l1, l2 in itertools.permutations(lits, 2):
        for b1, b2, b3 in itertools.product(boxes, repeat=3):
            if b1 == b2 == b3:
                continue
            yield Rule((Clause((l1,), b1), Clause((l2,), b2)), b3, "decision_list")


@dataclass(frozen=True)
class HypothesisSpace:
    """Deduplicated hypotheses and their label table.

    ``labels[h, x]`` is the box index hypothesis ``h`` assigns to item type ``x``.
    """

    items: ItemSpace
    boxes: tuple[str, ...]
    rules: tuple[Rule, ...]
    labels: np.ndarray  # (H, X) int8
    family_sizes: dict[str, int]  # after deduplication

    @staticmethod
    def build(
        values: Mapping[str, Sequence[str]],
        boxes: Sequence[str],
        families: Sequence[str] = FAMILIES,
    ) -> HypothesisSpace:
        """Enumerate the chosen families, evaluate on every item type, merge identical behaviors.

        Ties between descriptions of the same behavior are broken
        deterministically (decision D7, revision 7) by complexity, family
        order, number of clauses, then attribute/value/box order as listed in
        scene.yaml. Nothing depends on text or hash order.

        Raises:
            ValueError: for an unknown family name.
        """
        unknown = set(families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown rule families {sorted(unknown)}; choose from {FAMILIES}")
        items = ItemSpace.from_values(values)
        types = items.types
        box_list = tuple(boxes)
        attr_idx = {a: i for i, a in enumerate(items.attributes)}
        val_idx = {
            (a, v): j
            for a, vals in zip(items.attributes, items.values, strict=True)
            for j, v in enumerate(vals)
        }

        def vocab_key(rule: Rule) -> tuple[int, ...]:
            key: list[int] = []
            for c in rule.clauses:
                for a, v in sorted(c.condition, key=lambda av: attr_idx[av[0]]):
                    key += [attr_idx[a], val_idx[(a, v)]]
                key.append(box_list.index(c.box))
            key.append(box_list.index(rule.default))
            return tuple(key)

        best: dict[bytes, tuple[tuple[int, int, int, tuple[int, ...]], Rule, np.ndarray]] = {}
        for rule in _enumerate(items, box_list):
            if rule.family not in families:
                continue
            lab = np.array([box_list.index(rule.apply(t)) for t in types], dtype=np.int8)
            key = lab.tobytes()
            rank = (rule.complexity, FAMILIES.index(rule.family), len(rule.clauses), vocab_key(rule))
            if key not in best or rank < best[key][0]:
                best[key] = (rank, rule, lab)
        ordered = sorted(best.values(), key=lambda t: t[0])
        rules = tuple(r for _, r, _ in ordered)
        labels = np.stack([lab for _, _, lab in ordered])
        sizes = {f: sum(1 for r in rules if r.family == f) for f in FAMILIES}
        return HypothesisSpace(items, box_list, rules, labels, sizes)

    def __len__(self) -> int:
        return len(self.rules)

    def describe(self, index: int) -> str:
        """Plain-language sentence for hypothesis ``index``."""
        return describe(self.rules[index])

    def find(self, rule: Rule) -> int:
        """Index of the hypothesis that behaves exactly like ``rule``.

        Raises:
            KeyError: if no hypothesis behaves like it.
        """
        lab = np.array([self.boxes.index(rule.apply(t)) for t in self.items.types], dtype=np.int8)
        hits = np.flatnonzero((self.labels == lab).all(axis=1))
        if len(hits) == 0:
            raise KeyError("rule is outside the hypothesis space")
        return int(hits[0])
