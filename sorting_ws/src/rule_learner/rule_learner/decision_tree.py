"""Baseline: an ID3 decision tree on categorical attributes.

Used only in the benchmark, to compare against the Bayesian learner. ID3
picks, at each node, the attribute whose split maximizes information gain
on the training demonstrations, with multi-way splits on categorical
values. Unseen values and empty branches fall back to the parent's majority
box. It returns one tree: no confidence, no competing rules, no principled
query selection, which is exactly the contrast the benchmark shows.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class Node:
    """A tree node: either a leaf (``box``) or a split on ``attribute``."""

    box: str
    attribute: str | None = None
    children: dict[str, Node] = field(default_factory=dict)

    def predict(self, item: Mapping[str, str]) -> str:
        """Box for an item (majority box if the value was never seen)."""
        if self.attribute is None:
            return self.box
        child = self.children.get(item[self.attribute])
        return child.predict(item) if child else self.box


def _entropy(labels: Sequence[str]) -> float:
    n = len(labels)
    return -sum(c / n * math.log2(c / n) for c in Counter(labels).values())


def _majority(labels: Sequence[str], tie_order: Sequence[str]) -> str:
    counts = Counter(labels)
    return max(tie_order, key=lambda b: (counts.get(b, 0), -tie_order.index(b)))


def fit(
    examples: Sequence[tuple[Mapping[str, str], str]],
    attributes: Sequence[str],
    boxes: Sequence[str],
    parent_majority: str | None = None,
) -> Node:
    """Train an ID3 tree on (item attributes, box) examples."""
    if not examples:
        return Node(parent_majority or boxes[0])
    labels = [b for _, b in examples]
    majority = _majority(labels, boxes)
    if len(set(labels)) == 1 or not attributes:
        return Node(majority)
    base = _entropy(labels)

    def gain(attr: str) -> float:
        groups: dict[str, list[str]] = {}
        for item, b in examples:
            groups.setdefault(item[attr], []).append(b)
        return base - sum(len(g) / len(labels) * _entropy(g) for g in groups.values())

    best = max(attributes, key=lambda a: (gain(a), -list(attributes).index(a)))
    if gain(best) <= 1e-12:
        return Node(majority)
    node = Node(majority, best)
    rest = [a for a in attributes if a != best]
    for value in sorted({item[best] for item, _ in examples}):
        subset = [(i, b) for i, b in examples if i[best] == value]
        node.children[value] = fit(subset, rest, boxes, majority)
    return node
