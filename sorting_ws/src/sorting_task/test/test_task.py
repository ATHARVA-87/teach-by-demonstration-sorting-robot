"""Tests for decision-list rules, their sentences, and the task-file schema."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from sorting_task import task_file
from sorting_task.rules import Rule, describe

VOCAB = {
    "color": ["red", "green", "blue", "yellow"],
    "size": ["small", "medium", "large"],
    "shape": ["cube", "cylinder", "prism"],
}
BOXES = ["A", "B", "C"]
COLOR_RULE = Rule.from_parts([({"color": "red"}, "A"), ({"color": "blue"}, "B")], "C", "mapping")


def item(color: str, size: str, shape: str) -> dict[str, str]:
    return {"color": color, "size": size, "shape": shape}


# --- rules ----------------------------------------------------------------------
def test_first_matching_clause_wins() -> None:
    r = Rule.from_parts([({"color": "red"}, "A"), ({"size": "large"}, "B")], "C", "decision_list")
    assert r.apply(item("red", "large", "cube")) == "A"
    assert r.apply(item("blue", "large", "cube")) == "B"
    assert r.apply(item("blue", "small", "cube")) == "C"


def test_conjunction_needs_all_conditions() -> None:
    r = Rule.from_parts([({"color": "red", "size": "large"}, "C")], "A", "conjunction")
    assert r.apply(item("red", "large", "prism")) == "C"
    assert r.apply(item("red", "small", "prism")) == "A"


@pytest.mark.parametrize(
    "rule, sentence",
    [
        (Rule((), "B", "constant"), "Put everything in box B."),
        (COLOR_RULE, "Sort by color: red to A, blue to B, others to C."),
        (
            Rule.from_parts([({"size": "large"}, "B")], "A", "one_vs_rest"),
            "Sort by size: large to B, everything else to A.",
        ),
        (
            Rule.from_parts([({"color": "red", "size": "large"}, "C")], "A", "conjunction"),
            "Large red items to C, everything else to A.",
        ),
        (
            Rule.from_parts([({"color": "blue"}, "A"), ({"shape": "cube"}, "A")], "B", "decision_list"),
            "Blue items or cubes to A, everything else to B.",
        ),
        (
            Rule.from_parts([({"color": "red"}, "A"), ({"size": "large"}, "B")], "C", "decision_list"),
            "Red items to A; otherwise large items to B; everything else to C.",
        ),
    ],
)
def test_plain_language(rule: Rule, sentence: str) -> None:
    assert describe(rule) == sentence


# --- task file ------------------------------------------------------------------
def _task() -> task_file.TaskFile:
    prov = task_file.ProvenanceModel.model_validate(
        {
            "confidence": 0.995,
            "demonstrations": 9,
            "queries": 3,
            "epsilon": 0.05,
            "lambda": 1.0,
            "threshold": 0.99,
            "competing": [{"description": "Sort by color: red to A, others to C.", "probability": 0.003}],
        }
    )
    return task_file.build_task_file(
        COLOR_RULE,
        "sort_by_color",
        describe(COLOR_RULE),
        VOCAB,
        BOXES,
        prov,
        created_at=datetime(2026, 10, 26, tzinfo=UTC),
    )


def test_round_trip(tmp_path: Path) -> None:
    p = task_file.save(_task(), tmp_path / "tasks" / "t.json")
    loaded = task_file.load(p)
    assert loaded.to_rule() == COLOR_RULE
    raw = json.loads(p.read_text())
    assert raw["rule"]["clauses"][0] == {"if": {"color": "red"}, "box": "A"}
    assert raw["provenance"]["lambda"] == 1.0 and raw["provenance"]["approved"] is True
    assert raw["on_unknown_item"] == "skip_and_report"
    rule_text = json.dumps(raw["rule"])
    assert '"x"' not in rule_text and '"y"' not in rule_text  # robot-independent: no coordinates


def _bad(mutate) -> dict:  # noqa: ANN001
    raw = json.loads(_task().model_dump_json(by_alias=True))
    mutate(raw)
    return raw


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["rule"].__setitem__("default_box", "Z"), "box 'Z'"),
        (lambda d: d["rule"]["clauses"][0]["if"].__setitem__("color", "purple"), "purple"),
        (lambda d: d["rule"]["clauses"][0]["if"].__setitem__("weight", "heavy"), "weight"),
        (lambda d: d["rule"]["clauses"].append({"if": {"color": "red"}, "box": "B"}), "duplicate clause"),
        (lambda d: d["rule"]["clauses"][0].__setitem__("if", {}), "if"),
        (lambda d: d.__setitem__("boxes", ["A", "A", "B"]), "duplicates"),
        (lambda d: d.__setitem__("schema_version", "2.0"), "schema_version"),
        (lambda d: d.__setitem__("name", "Bad Name!"), "name"),
        (lambda d: d.__setitem__("robot_speed", 1.0), "robot_speed"),
        (lambda d: d.pop("rule"), "rule"),
        (lambda d: d["provenance"].__setitem__("confidence", 1.5), "confidence"),
    ],
)
def test_schema_rejects_bad_files(tmp_path: Path, mutate, message: str) -> None:  # noqa: ANN001
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(_bad(mutate)))
    with pytest.raises(ValidationError, match=message):
        task_file.load(p)


def test_json_schema_export(tmp_path: Path) -> None:
    out = tmp_path / "schema.json"
    task_file.export_schema_main(["--out", str(out)])
    schema = json.loads(out.read_text())
    assert schema["title"] == "TaskFile" and "rule" in schema["properties"]
