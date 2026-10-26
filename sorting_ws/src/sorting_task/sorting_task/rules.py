"""Sorting rules as decision lists, and their plain-language sentences.

A rule is an ordered list of clauses ``if <attribute conditions> then <box>``
plus a default box: the first matching clause wins. This is the format the
rule learner reasons in *and* the format the robot executes, so whatever the
operator approves is exactly what runs.

Families (complexity c, used by the learner's simplicity prior):
    constant            c=0   everything to A
    one_vs_rest         c=1   red to A, everything else to B
    mapping             c=2   red to A, blue to B, others to C
    conjunction         c=2   large red to A, everything else to B
    decision_list       c=3   red to A; otherwise large to B; everything else to C
                              (or a disjunction: red items or large items to A ...)

Pure Python: imported by rule_learner (learning) and sorting_execution
(execution) without either depending on the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

FAMILIES: tuple[str, ...] = ("constant", "one_vs_rest", "mapping", "conjunction", "decision_list")
COMPLEXITY: dict[str, int] = {
    "constant": 0,
    "one_vs_rest": 1,
    "mapping": 2,
    "conjunction": 2,
    "decision_list": 3,
}

Condition = tuple[tuple[str, str], ...]  # sorted ((attribute, value), ...): all must hold


@dataclass(frozen=True)
class Clause:
    """``if all(item[a] == v for a, v in condition) then box``."""

    condition: Condition
    box: str

    def matches(self, item: Mapping[str, str]) -> bool:
        """True if the item satisfies every (attribute, value) in the condition."""
        return all(item[a] == v for a, v in self.condition)


@dataclass(frozen=True)
class Rule:
    """A decision list: first matching clause wins, else ``default``."""

    clauses: tuple[Clause, ...]
    default: str
    family: str

    @property
    def complexity(self) -> int:
        """Description complexity used by the simplicity prior."""
        return COMPLEXITY[self.family]

    def apply(self, item: Mapping[str, str]) -> str:
        """Box for an item with these attributes."""
        for clause in self.clauses:
            if clause.matches(item):
                return clause.box
        return self.default

    def to_dict(self) -> dict[str, object]:
        """Plain dict: ``{"clauses": [{"if": {...}, "box": ...}], "default_box": ...}``."""
        return {
            "clauses": [{"if": dict(c.condition), "box": c.box} for c in self.clauses],
            "default_box": self.default,
        }

    @staticmethod
    def from_parts(clauses: Sequence[tuple[Mapping[str, str], str]], default: str, family: str) -> Rule:
        """Build from ``[(condition dict, box), ...]``."""
        return Rule(tuple(Clause(tuple(sorted(c.items())), b) for c, b in clauses), default, family)


# --- plain language -----------------------------------------------------------
_PHRASE_ORDER = {"size": 0, "color": 1, "shape": 2}


def _phrase(condition: Condition) -> str:
    """Noun phrase such as 'large red items' or 'small cubes'."""
    parts = sorted(condition, key=lambda av: (_PHRASE_ORDER.get(av[0], 9), av[0]))
    shape = next((v for a, v in parts if a == "shape"), None)
    words = [v for a, v in parts if a != "shape"]
    noun = f"{shape}s" if shape else "items"
    return " ".join([*words, noun])


def describe(rule: Rule) -> str:
    """Plain-language sentence for the operator, in sentence case.

    Examples:
        "Put everything in box A."
        "Sort by color: red to A, blue to B, others to C."
        "Sort by size: large to B, everything else to A."
        "Large red items to C, everything else to A."
        "Blue items or cubes to A, everything else to B."
        "Red items to A; otherwise large items to B; everything else to C."
    """
    if rule.family == "constant" or not rule.clauses:
        return f"Put everything in box {rule.default}."
    if rule.family == "mapping":
        attr = rule.clauses[0].condition[0][0]
        by_box: dict[str, list[str]] = {}
        for c in rule.clauses:
            by_box.setdefault(c.box, []).append(c.condition[0][1])
        parts = [f"{' and '.join(vals)} to {box}" for box, vals in by_box.items()]
        return f"Sort by {attr}: {', '.join(parts)}, others to {rule.default}."
    if rule.family == "one_vs_rest" and len(rule.clauses) == 1 and len(rule.clauses[0].condition) == 1:
        ((attr, val),) = rule.clauses[0].condition
        return f"Sort by {attr}: {val} to {rule.clauses[0].box}, everything else to {rule.default}."
    boxes = {c.box for c in rule.clauses}
    if len(rule.clauses) > 1 and len(boxes) == 1 and rule.default not in boxes:
        # Disjunction: every clause sends to the same box.
        text = (
            f"{' or '.join(_phrase(c.condition) for c in rule.clauses)} to {rule.clauses[0].box}, "
            f"everything else to {rule.default}."
        )
    else:
        phrases = [f"{_phrase(c.condition)} to {c.box}" for c in rule.clauses]
        if len(phrases) == 1:
            text = f"{phrases[0]}, everything else to {rule.default}."
        else:
            rest = "; otherwise ".join(phrases[1:])
            text = f"{phrases[0]}; otherwise {rest}; everything else to {rule.default}."
    return text[0].upper() + text[1:]
