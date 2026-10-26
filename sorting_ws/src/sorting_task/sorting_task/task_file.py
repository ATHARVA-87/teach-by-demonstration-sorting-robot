"""Robot-independent task file: the approved sorting rule as JSON.

It says *what* goes where, never *how* to move. Any robot (or a non-robotic
sorter) executes it by evaluating the decision list on each perceived item.
Box labels only, no coordinates: where box "A" is belongs to the cell, not
the task. Schema version 1.0 (decision D7, revision 11); JSON Schema in
``docs/task_file_schema.json`` (``ros2 run sorting_task export_schema``).

Example::

    {
      "schema_version": "1.0",
      "task": "sort",
      "name": "sort_by_color_red_to_a_blue_to_b_others_to_c",
      "created_at": "2026-10-26T10:12:03+05:30",
      "description": "Sort by color: red to A, blue to B, others to C.",
      "family": "mapping",
      "attributes": {"color": ["red", "green", "blue", "yellow"], "size": [...], "shape": [...]},
      "boxes": ["A", "B", "C"],
      "rule": {"clauses": [{"if": {"color": "red"}, "box": "A"},
                           {"if": {"color": "blue"}, "box": "B"}],
               "default_box": "C"},
      "on_unknown_item": "skip_and_report",
      "provenance": {"approved": true, "confidence": 0.995, "demonstrations": 9, "queries": 3,
                     "epsilon": 0.05, "lambda": 1.0, "threshold": 0.99,
                     "competing": [{"description": "...", "probability": 0.003}],
                     "unexplained_demonstrations": []}
    }
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from sorting_task.rules import FAMILIES, Rule

SCHEMA_VERSION: Literal["1.0"] = "1.0"


class ClauseModel(BaseModel):
    """``if`` every listed attribute has the given value, the item goes to ``box``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    when: dict[str, str] = Field(alias="if", min_length=1)
    box: str


class RuleModel(BaseModel):
    """Decision list: the first matching clause wins, otherwise ``default_box``."""

    model_config = ConfigDict(extra="forbid")
    clauses: list[ClauseModel]
    default_box: str


class CompetitorModel(BaseModel):
    """Another rule that was still plausible when the task was approved."""

    model_config = ConfigDict(extra="forbid")
    description: str
    probability: float = Field(ge=0.0, le=1.0)


class ProvenanceModel(BaseModel):
    """How the rule was learned and approved (audit trail and evaluation)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    approved: Literal[True] = True  # only approved rules become task files
    confidence: float = Field(ge=0.0, le=1.0)
    demonstrations: int = Field(ge=0)
    queries: int = Field(ge=0)
    epsilon: float = Field(ge=0.0, lt=0.5)
    prior_lambda: float = Field(alias="lambda", ge=0.0)
    threshold: float = Field(ge=0.0, le=1.0)
    competing: list[CompetitorModel] = []
    unexplained_demonstrations: list[int] = []


class TaskFile(BaseModel):
    """The complete task file."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    task: Literal["sort"] = "sort"
    name: str = Field(pattern=r"^[a-z0-9_\-]+$", max_length=80)
    created_at: AwareDatetime  # with time zone, so files from different machines compare correctly
    description: str = Field(min_length=1)
    family: Literal["constant", "one_vs_rest", "mapping", "conjunction", "decision_list"]
    attributes: dict[str, list[str]] = Field(min_length=1)
    boxes: list[str] = Field(min_length=2)
    rule: RuleModel
    on_unknown_item: Literal["skip_and_report"] = "skip_and_report"
    provenance: ProvenanceModel

    @model_validator(mode="after")
    def _consistent(self) -> TaskFile:
        """Rule references must exist in the declared vocabulary; no duplicates."""
        if len(set(self.boxes)) != len(self.boxes):
            raise ValueError(f"boxes contain duplicates: {self.boxes}")
        for attr, values in self.attributes.items():
            if not values or len(set(values)) != len(values):
                raise ValueError(f"attribute '{attr}' needs unique, non-empty values")
        for box in [c.box for c in self.rule.clauses] + [self.rule.default_box]:
            if box not in self.boxes:
                raise ValueError(f"rule uses box '{box}', which is not in boxes {self.boxes}")
        seen: set[tuple[tuple[str, str], ...]] = set()
        for clause in self.rule.clauses:
            for attr, value in clause.when.items():
                if attr not in self.attributes:
                    raise ValueError(f"rule uses unknown attribute '{attr}'")
                if value not in self.attributes[attr]:
                    raise ValueError(f"rule uses unknown {attr} value '{value}'")
            key = tuple(sorted(clause.when.items()))
            if key in seen:
                raise ValueError(f"duplicate clause condition {dict(key)}")
            seen.add(key)
        return self

    def to_rule(self) -> Rule:
        """The executable rule."""
        return Rule.from_parts(
            [(c.when, c.box) for c in self.rule.clauses], self.rule.default_box, self.family
        )


def build_task_file(
    rule: Rule,
    name: str,
    description: str,
    attributes: dict[str, list[str]],
    boxes: list[str],
    provenance: ProvenanceModel,
    created_at: datetime | None = None,
) -> TaskFile:
    """Assemble and validate a task file from an approved rule."""
    if rule.family not in FAMILIES:
        raise ValueError(f"unknown rule family '{rule.family}'")
    return TaskFile.model_validate(
        {
            "name": name,
            "created_at": created_at or datetime.now().astimezone(),
            "description": description,
            "family": rule.family,
            "attributes": attributes,
            "boxes": boxes,
            "rule": rule.to_dict(),
            "provenance": provenance.model_dump(by_alias=True),
        }
    )


def save(task: TaskFile, path: str | Path) -> Path:
    """Write the task file as indented JSON (parent dirs are created)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(task.model_dump_json(by_alias=True, indent=2) + "\n")
    return p


def load(path: str | Path) -> TaskFile:
    """Read and validate a task file.

    Raises:
        pydantic.ValidationError: naming the exact field that is wrong.
    """
    return TaskFile.model_validate(json.loads(Path(path).read_text()))


def json_schema() -> dict[str, object]:
    """JSON Schema of the task file."""
    return TaskFile.model_json_schema(by_alias=True)


def export_schema_main(argv: list[str] | None = None) -> None:
    """CLI: write the JSON Schema (default docs/task_file_schema.json)."""
    import argparse

    ap = argparse.ArgumentParser(description="Export the task-file JSON Schema.")
    ap.add_argument("--out", default="docs/task_file_schema.json")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_schema(), indent=2) + "\n")
    print(f"Wrote {out}")
