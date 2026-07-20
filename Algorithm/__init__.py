"""
Algorithm/ — self-contained implementation of the R-MCTS planners
(UCB variant now; RUDB variant later).

Deliberately duplicated from src/robust_pomdp (not imported) so the two
codebases can be compared side by side. See docs / project_progress for the
design rationale (scalar-c tighter certificate, S_next next-state support).
"""
