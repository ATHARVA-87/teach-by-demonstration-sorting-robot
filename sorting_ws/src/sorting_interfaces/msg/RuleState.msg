# What the rule learner currently believes (latched; published after every change).

std_msgs/Header header
string session_id                  # teaching session the demonstrations came from
uint32 n_demonstrations
RuleCandidate[] candidates         # most likely first
Conflict[] conflicts               # hard contradictions
uint32[] unexplained               # demo indices the most likely rule disagrees with
uint32 version_space_size          # rules consistent with every demonstration
bool converged                     # P(most likely rule) >= stop_confidence
bool asking                        # a question item is waiting in the AR view
Item query                         # the question item (valid when asking)
uint32 queries_asked
bool budget_exhausted              # max_queries reached without converging: operator decides
