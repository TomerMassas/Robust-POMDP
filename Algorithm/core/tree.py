"""
Search-tree data structures for the R-MCTS planners (option-B: deterministic
projected-Bayes beliefs, no particles).

The tree alternates HistoryNode -> ActionNode -> HistoryNode -> ...

New vs. the src/robust_pomdp tree:
  - No particle list / rollout caching / dirty flags (that machinery is
    POMCP-specific; beliefs here are deterministic projected-Bayes).
  - ActionNode carries S_next (union of its children's S_in = the sampled
    next-state support) and a per-state continuation-survival `delta` (paper δ).
  - HistoryNode carries a per-state continuation-survival `delta` (greedy-policy value).

`delta` and beliefs are keyed by state index (dict) so cross-node lookups
(child.delta[s']) stay clean regardless of differing supports.
Ids are 0-based; state/action/observation indices are 0-based.
"""

from __future__ import annotations


class NodeIdCounter:
    """Counter producing unique sequential node ids. Reset per planner call."""

    def __init__(self) -> None:
        self.next: int = 0

    def next_id(self) -> int:
        self.next += 1
        return self.next


class ActionNode:
    """An action edge in the search tree.

    Attributes:
        id: unique integer id.
        children: observation -> child HistoryNode.
        Z_in: sampled observation support.
        S_next: sampled next-state support (union of children's S_in).
        N: visit count.
        Q_rob: robust value of this action (projected worst-case backup).
        Q_nom: sample-mean return (nominal-mode UCB exploration only).
        delta: dict state -> continuation-survival δ (leakage certificate, Eq. 50),
            over the parent's S_in.
        r_exact: exact immediate reward Σ_s b0(s)·R(s,a), computed once at ROOT
            action nodes (the root belief is exact); None otherwise. Used in place
            of the projected immediate reward in backup.
    """

    def __init__(self, id: int) -> None:
        self.id = id
        self.children: dict[int, HistoryNode] = {}
        self.Z_in: set[int] = set()
        self.S_next: set[int] = set()
        self.N: int = 0
        self.Q_rob: float = 0.0
        self.Q_nom: float = 0.0
        self.delta: dict[int, float] = {}
        self.r_exact: float | None = None

    def get_or_create_child(self,
                            z: int,
                            child_depth: int,
                            counter: NodeIdCounter
                            ) -> tuple["HistoryNode", bool]:
        """Record observation z, return its child HistoryNode (creating if new).

        Returns (child_node, was_created).
        """
        self.Z_in.add(z)
        if z not in self.children:
            self.children[z] = HistoryNode(counter.next_id(), child_depth)
            return self.children[z], True
        return self.children[z], False


class HistoryNode:
    """A belief node in the search tree.

    Attributes:
        id: unique integer id.
        depth: depth in tree (root = 0).
        S_in: sampled state support at this node.
        children: action -> ActionNode (only expanded actions present).
        N: visit count.
        V_rob: max_a Q_rob(a) over expanded actions (greedy robust value).
        delta: dict state -> continuation-survival δ under the greedy policy.
    """

    def __init__(self, id: int, depth: int) -> None:
        self.id = id
        self.depth = depth
        self.S_in: set[int] = set()
        self.children: dict[int, ActionNode] = {}
        self.N: int = 0
        self.V_rob: float = 0.0
        self.delta: dict[int, float] = {}

    def is_frontier(self) -> bool:
        """A frontier node has no expanded actions yet (a just-created leaf)."""
        return not self.children

    def unexpanded_actions(self, n_actions: int) -> list[int]:
        """Actions not yet expanded at this node (0-based)."""
        return [a for a in range(n_actions) if a not in self.children]

    def expand_action(self, a: int, counter: NodeIdCounter) -> ActionNode:
        """Create and attach an ActionNode for action a."""
        node = ActionNode(counter.next_id())
        self.children[a] = node
        return node
