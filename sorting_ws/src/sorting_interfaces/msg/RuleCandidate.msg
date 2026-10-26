# One candidate sorting rule and its posterior probability.

string description   # plain-language sentence, e.g. "Sort by color: red to A, blue to B, others to C."
string rule_json     # decision list: {"clauses": [{"if": {...}, "box": "A"}], "default_box": "C"}
string family        # constant | one_vs_rest | mapping | conjunction | decision_list
float64 probability  # P(rule | demonstrations)
uint32 errors        # demonstrations this rule disagrees with
