# Rule learning: proposal for decision (M2)

> **Superseded in parts by decision D7 (DECISIONS.md):** stop threshold 0.99 (not 0.95), **no** "max IG < 0.05 bits" stop rule, a question budget instead, and the other D7 revisions. This file is kept as the original proposal for the record.

## Problem
From a few (item attributes → box) demonstrations, infer the sorting rule:
- report its confidence and the competing rules;
- ask for the most informative item when the rule is ambiguous;
- detect contradictions;
- output a robot-independent task file.

Item space: color (4) × size (3) × shape (3) = **36 item types**; boxes A, B, C.

## Option A (proposed): Bayesian concept learning over an enumerated, interpretable hypothesis space
- **Hypotheses:** total functions *item type → box*, written in five families of increasing complexity:
  | Family | Example | Complexity c |
  |---|---|---|
  | constant | everything to A | 0 |
  | one value vs rest | red to A, else B | 1 |
  | single-attribute mapping | red to A, blue to B, others to C | 2 |
  | 2-attribute conjunction vs rest | large red to A, else B | 2 |
  | 2-clause decision list | red to A, then large to B, else C | 3 |
- **Deduplication:** hypotheses with the same behavior on all 36 item types are merged, keeping the simplest description. That leaves roughly 1,000 hypotheses, so exact enumeration is cheap (vectorized, < 1 ms per update).
- **Prior (simplicity / Bayesian Occam's razor):** $P(h) = P(f)/|f|$ with $P(f) \propto e^{-\lambda c_f}$. Simple families get more mass, and large families spread theirs thinner.
- **Likelihood with teacher noise ε:** $P(b \mid x, h) = 1-\varepsilon$ if $h(x) = b$, else $\varepsilon/(k-1)$. One wrong drop lowers a hypothesis's score but doesn't eliminate it.
- **Version space** = hypotheses with zero errors (the ε → 0 limit), reported alongside.
- **Active learning:** query the item type $x^* = \arg\max_x \mathrm{IG}(x)$, where $\mathrm{IG}(x) = H\big(\sum_h P(h \mid D)\, P(y \mid x,h)\big) - \mathbb{E}_h[H(y \mid x,h)]$. With symmetric noise the second term is constant, so this is maximum predictive entropy (maximum disagreement weighted by the posterior). Stop when $P(h_{MAP} \mid D) \ge 0.95$ or $\max_x \mathrm{IG} < 0.05$ bits.
- **Contradictions:**
  - *Hard:* same item type, different boxes. No deterministic rule fits both; reported as a pair.
  - *Soft:* demos the MAP rule disagrees with, reported as "doesn't fit the learned rule".
- **Output:** every hypothesis compiles to a canonical **decision list** (`[{if: {attr: value, ...}, box}]` + `default_box`). That's the task-file format, and it's trivial for any robot to execute.

## Option B: decision tree (ID3/CART on the demonstrations)
- **Pros:** standard and familiar; handles any rule.
- **Cons:**
  - no calibrated confidence and no competing hypotheses;
  - no principled query selection (only heuristics such as tree-disagreement ensembles);
  - with 3–8 examples it overfits or picks arbitrary splits;
  - a contradiction just becomes an impure leaf.
- **Use here:** baseline in the benchmark (demos to the correct rule, accuracy on new scenes).

## Option C: pure version-space learning (candidate elimination)
- **Pros:** exact and simple.
- **Cons:** a single teacher mistake empties the version space (no noise robustness); no preference between consistent rules without an added bias.
- **Use here:** reported as the ε → 0 special case of A.

## Implementation notes
- Pure Python + numpy (no new dependencies).
- The decision-tree baseline is a small ID3 implemented in the repo, so no scikit-learn dependency.
- Task-file schema validated with pydantic (already installed as a FastAPI dependency; will be pinned), with JSON Schema exported to `docs/task_file_schema.json`.
