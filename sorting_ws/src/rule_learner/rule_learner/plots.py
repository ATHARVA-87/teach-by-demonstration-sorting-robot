"""Summary tables and plots for the rule-learning benchmark.

Reads trials.csv / curves.csv written by ``benchmark`` and writes
``summary.md`` plus PNG figures. Styling follows docs/design_plan.md (D6,
rev. 12): series are Signal white solid vs. grey dashed with distinct markers
(never status colors) on the console's dark background, plus a light
variant for the README and the technical report.

    ros2 run rule_learner plot_results --in results/rule_learning
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "dark": {
        "bg": "#2E3236",
        "fg": "#ECECE7",
        "muted": "#A7A9A8",
        "grid": "#474D52",
        "a": "#ECECE7",
        "b": "#A7A9A8",
    },
    "light": {
        "bg": "#FFFFFF",
        "fg": "#1E2124",
        "muted": "#5A6066",
        "grid": "#D5D7D8",
        "a": "#1E2124",
        "b": "#7A8086",
    },
}
FAMILY_LABELS = {
    "one_vs_rest": "One value vs rest",
    "mapping": "Single-attribute mapping",
    "conjunction": "Two-attribute conjunction",
    "decision_list": "Two-clause decision list",
}
SETTING_LABELS = {
    "main": "Final learner (ε = 0.05)",
    "conflict_requery_always": "Before fix: re-ask any conflict",
    "epsilon_0.10": "ε = 0.10 (matches the teacher)",
    "no_decision_lists": "Without the decision-list family",
}


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _iqr(values: list[int]) -> tuple[float, float]:
    if len(values) < 2:
        return (float(values[0]), float(values[0])) if values else (float("nan"), float("nan"))
    q = quantiles(values, n=4)
    return q[0], q[2]


def summarize(trials: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Aggregates per (setting, family, teacher noise, mode)."""
    groups: dict[tuple[str, str, float, str], list[dict[str, str]]] = defaultdict(list)
    for r in trials:
        groups[(r["setting"], r["family"], float(r["teacher_noise"]), r["mode"])].append(r)
    rows = []
    for (setting, fam, noise, mode), rs in sorted(groups.items()):
        reached = [int(r["labels_to_correct"]) for r in rs if r["labels_to_correct"]]
        lo, hi = _iqr(reached)
        rows.append(
            {
                "setting": setting,
                "family": fam,
                "noise": noise,
                "mode": mode,
                "trials": len(rs),
                "l2c_median": median(reached) if reached else float("nan"),
                "l2c_q1": lo,
                "l2c_q3": hi,
                "not_reached": 1 - len(reached) / len(rs),
                "stop_median": median(int(r["labels_at_stop"]) for r in rs),
                "correct_at_stop": mean(r["correct_at_stop"] == "True" for r in rs),
                "accuracy_at_stop": mean(float(r["accuracy_at_stop"]) for r in rs),
                "tree_accuracy_at_stop": mean(float(r["tree_accuracy_at_stop"]) for r in rs),
            }
        )
    return rows


def _row(r: dict[str, Any]) -> str:
    l2c = (
        "–"
        if r["l2c_median"] != r["l2c_median"]
        else f"{r['l2c_median']:g} ({r['l2c_q1']:g}–{r['l2c_q3']:g})"
    )
    return (
        f"| {FAMILY_LABELS.get(r['family'], r['family'])} | {r['noise']:.0%} | {r['mode']} | {l2c} | "
        f"{r['not_reached']:.0%} | {r['stop_median']:g} | {r['correct_at_stop']:.0%} | "
        f"{r['accuracy_at_stop']:.1%} | {r['tree_accuracy_at_stop']:.1%} |"
    )


HEADER = (
    "| Rule type | Teacher mistakes | Mode | Labels to correct, median (IQR) | Never settled (≤40) | "
    "Labels at stop | Correct at stop | Accuracy on new scenes | ID3 tree accuracy |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    """Main table + ablation table (copied into README / technical report)."""
    main = [r for r in rows if r["setting"] == "main"]
    lines = [
        "### Main results (ε = 0.05, stop at P ≥ 0.99, ≤ 20 questions)",
        "",
        HEADER,
        *map(_row, main),
        "",
    ]
    abl = [r for r in rows if r["noise"] == 0.10 and r["mode"] == "active"]
    if any(r["setting"] != "main" for r in abl):
        lines += [
            "### Ablations (active learning, 10% teacher mistakes)",
            "",
            "| Setting | Rule type | Labels to correct, median | Never settled | Correct at stop |",
            "|---|---|---|---|---|",
        ]
        for r in sorted(abl, key=lambda r: (list(SETTING_LABELS).index(r["setting"]), r["family"])):
            l2c = "–" if r["l2c_median"] != r["l2c_median"] else f"{r['l2c_median']:g}"
            lines.append(
                f"| {SETTING_LABELS.get(r['setting'], r['setting'])} | "
                f"{FAMILY_LABELS.get(r['family'], r['family'])} | {l2c} | {r['not_reached']:.0%} | "
                f"{r['correct_at_stop']:.0%} |"
            )
    path.write_text("\n".join(lines) + "\n")


def _style(ax: Any, t: dict[str, str]) -> None:
    ax.set_facecolor(t["bg"])
    for s in ax.spines.values():
        s.set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=7)
    ax.grid(color=t["grid"], linewidth=0.6)
    ax.xaxis.label.set_color(t["fg"])
    ax.yaxis.label.set_color(t["fg"])
    ax.title.set_color(t["fg"])


def _legend(ax: Any, t: dict[str, str]) -> None:
    leg = ax.legend(fontsize=7, frameon=False)
    for txt in leg.get_texts():
        txt.set_color(t["fg"])


def plot_labels_vs_noise(rows: list[dict[str, Any]], out: Path) -> None:
    """Median labels to the correct rule vs teacher mistake rate, per rule type."""
    main = [r for r in rows if r["setting"] == "main"]
    fams = [f for f in FAMILY_LABELS if any(r["family"] == f for r in main)]
    for theme, t in THEMES.items():
        fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.0), sharey=True, facecolor=t["bg"])
        for ax, fam in zip(axes, fams, strict=True):
            for mode, ls, color, marker in (("active", "-", t["a"], "o"), ("random", "--", t["b"], "s")):
                rs = sorted(
                    (r for r in main if r["family"] == fam and r["mode"] == mode), key=lambda r: r["noise"]
                )
                xs = [r["noise"] * 100 for r in rs]
                label = "active questions" if mode == "active" else "random demos"
                ax.plot(
                    xs,
                    [r["l2c_median"] for r in rs],
                    ls,
                    color=color,
                    marker=marker,
                    markersize=4,
                    linewidth=1.6,
                    label=label,
                )
                ax.fill_between(
                    xs, [r["l2c_q1"] for r in rs], [r["l2c_q3"] for r in rs], color=color, alpha=0.12
                )
            ax.set_title(FAMILY_LABELS[fam], fontsize=8)
            ax.set_xlabel("Teacher mistakes (%)", fontsize=8)
            _style(ax, t)
        axes[0].set_ylabel("Labels until correct (median, IQR)", fontsize=8)
        _legend(axes[0], t)
        fig.tight_layout()
        fig.savefig(out / f"labels_vs_noise_{theme}.png", dpi=150, facecolor=t["bg"])
        plt.close(fig)


def plot_curves(curves: list[dict[str, str]], out: Path, noise: float = 0.10) -> None:
    """Accuracy on new scenes vs number of labels (one teacher-mistake level)."""
    fams = [f for f in FAMILY_LABELS if any(c["family"] == f for c in curves)]
    series = (
        ("active", "accuracy", "-", "a", "o", "Bayesian, active questions"),
        ("random", "accuracy", "--", "b", "s", "Bayesian, random demos"),
        ("random", "tree_accuracy", ":", "muted", "^", "ID3 tree, random demos"),
    )
    for theme, t in THEMES.items():
        fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.0), sharey=True, facecolor=t["bg"])
        for ax, fam in zip(axes, fams, strict=True):
            for mode, col, ls, color_key, marker, label in series:
                by_n: dict[int, list[float]] = defaultdict(list)
                for c in curves:
                    if (
                        c["setting"] == "main"
                        and c["family"] == fam
                        and c["mode"] == mode
                        and float(c["teacher_noise"]) == noise
                    ):
                        by_n[int(c["labels"])].append(float(c[col]))
                ns = sorted(by_n)
                ax.plot(
                    ns,
                    [mean(by_n[n]) for n in ns],
                    ls,
                    color=t[color_key],
                    marker=marker,
                    markersize=3,
                    markevery=4,
                    linewidth=1.4,
                    label=label,
                )
            ax.set_title(FAMILY_LABELS[fam], fontsize=8)
            ax.set_xlabel("Labels (opening scene + questions/demos)", fontsize=8)
            _style(ax, t)
        axes[0].set_ylabel(f"Accuracy on new scenes ({noise:.0%} mistakes)", fontsize=8)
        _legend(axes[-1], t)
        fig.tight_layout()
        fig.savefig(out / f"accuracy_curves_{theme}.png", dpi=150, facecolor=t["bg"])
        plt.close(fig)


def plot_ablation(rows: list[dict[str, Any]], out: Path) -> None:
    """P(correct at stop) per setting at 10% teacher mistakes (active learning)."""
    abl = [r for r in rows if r["noise"] == 0.10 and r["mode"] == "active"]
    settings = [s for s in SETTING_LABELS if any(r["setting"] == s for r in abl)]
    fams = [f for f in FAMILY_LABELS if any(r["family"] == f for r in abl)]
    hatches = ["", "//", "..", "xx"]
    for theme, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.0, 3.0), facecolor=t["bg"])
        width = 0.8 / len(settings)
        for j, s in enumerate(settings):
            for i, fam in enumerate(fams):
                r = next((r for r in abl if r["setting"] == s and r["family"] == fam), None)
                if r is None:
                    continue
                ax.bar(
                    i - 0.4 + width * (j + 0.5),
                    r["correct_at_stop"] * 100,
                    width * 0.9,
                    color=t["bg"],
                    edgecolor=t["a"] if j == 0 else t["b"],
                    hatch=hatches[j % 4],
                    linewidth=1.1,
                    label=SETTING_LABELS[s] if i == 0 else None,
                )
        ax.set_xticks(range(len(fams)), [FAMILY_LABELS[f] for f in fams], fontsize=7)
        ax.set_ylabel("Correct when it stops (%)", fontsize=8)
        ax.set_ylim(0, 105)
        _style(ax, t)
        _legend(ax, t)
        fig.tight_layout()
        fig.savefig(out / f"ablation_{theme}.png", dpi=150, facecolor=t["bg"])
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Plot rule-learning benchmark results.")
    ap.add_argument("--in", dest="indir", default="results/rule_learning")
    args = ap.parse_args(argv)
    d = Path(args.indir)
    rows = summarize(_read(d / "trials.csv"))
    write_markdown(rows, d / "summary.md")
    plot_labels_vs_noise(rows, d)
    plot_ablation(rows, d)
    if (d / "curves.csv").exists():
        plot_curves(_read(d / "curves.csv"), d)
    print(f"Wrote {d}/summary.md and figures")


if __name__ == "__main__":
    main()
