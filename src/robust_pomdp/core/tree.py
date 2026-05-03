"""
Search tree data structures for online POMDP planning.

The tree alternates between HistoryNode (belief nodes) and ActionNode (action edges):
  HistoryNode -> (pick action) -> ActionNode -> (observe) -> HistoryNode -> ...

Each HistoryNode stores particles defining the belief and S_in (sampled state support).
Each ActionNode stores Q-values and Z_in (sampled observation support).

Every node gets a unique integer id at creation time (via NodeIdCounter.next_id()).
Events emitted during simulation reference nodes by these ids.

Action / state / observation ids are 0-based throughout (Python convention).
"""

from __future__ import annotations


class NodeIdCounter:
    """Counter producing unique sequential node ids. Reset per planner call."""

    def __init__(self) -> None:
        self.next: int = 0

    def next_id(self) -> int:
        """Return a fresh id and advance the counter."""
        self.next += 1
        return self.next


class ActionNode:
    """An action edge in the search tree.

    Attributes:
        id: unique integer id, assigned at creation.
        children: observation -> child HistoryNode.
        Z_in: set of observations seen (sampled observation support).
        N: visit count.
        Q_nominal: average return from simulations (standard POMCP value).
        Q_robust: robust value from worst-case backup.
        dirty: whether this node needs robust value recomputation.
    """

    def __init__(self, id: int) -> None:
        self.id = id
        self.children: dict[int, HistoryNode] = {}
        self.Z_in: set[int] = set()
        self.N: int = 0
        self.Q_nominal: float = 0.0
        self.Q_robust: float = 0.0
        self.dirty: bool = True

    def get_or_create_child(self,
                            z: int,
                            child_depth: int,
                            id_counter: NodeIdCounter
                            ) -> tuple[HistoryNode, bool]:
        """Record observation z and return its child HistoryNode (creating if needed).

        Marks the action node as dirty when a new child is created.
        Returns (child_node, was_created).
        """
        self.Z_in.add(z)
        was_created = False
        if z not in self.children:
            new_id = id_counter.next_id()
            self.children[z] = HistoryNode(new_id, child_depth)
            self.dirty = True
            was_created = True
        return self.children[z], was_created


class HistoryNode:
    """A belief node in the search tree.

    Attributes:
        id: unique integer id.
        depth: depth in tree (root = 0).
        particles: list of states sampled at this node (may contain duplicates).
        S_in: set of unique states (sampled state support).
        children: action -> ActionNode.
        N: visit count.
        dirty: whether this node needs robust value recomputation.
        V_robust: cached max_a Q_robust(a), updated during robust backup.
        V_nominal: cached max_a Q_nominal(a), updated during simulation.
        V_rollout: rollout return saved when this node was first expanded.
        has_rollout: whether V_rollout has been set.
    """

    def __init__(self, id: int, depth: int) -> None:
        self.id = id
        self.depth = depth
        self.particles: list[int] = []
        self.S_in: set[int] = set()
        self.children: dict[int, ActionNode] = {}
        self.N: int = 0
        self.dirty: bool = True
        self.V_robust: float = 0.0
        self.V_nominal: float = 0.0
        self.V_rollout: float = 0.0
        self.has_rollout: bool = False

    def add_particle(self, s: int) -> None:
        """Add a sampled state to this node. Marks node as dirty."""
        self.particles.append(s)
        self.S_in.add(s)
        self.N += 1
        self.dirty = True

    def is_leaf(self) -> bool:
        """Return True if this node has not yet been expanded."""
        return not self.children

    def expand(self, n_actions: int, id_counter: NodeIdCounter) -> tuple[list[int], list[int]]:
        """Create ActionNodes for any actions not yet expanded.

        Returns (created_ids, created_actions): aligned lists where created_ids[i]
        is the new ActionNode's id and created_actions[i] is the action index (0-based).
        """
        created_ids: list[int] = []
        created_actions: list[int] = []
        for a in range(n_actions):
            if a not in self.children:
                new_id = id_counter.next_id()
                self.children[a] = ActionNode(new_id)
                created_ids.append(new_id)
                created_actions.append(a)
        return created_ids, created_actions

    def update_robust_value(self) -> None:
        """Update cached V_robust from children's Q_robust values.

        Only considers actions that have been visited (N > 0). If no action has been
        visited, falls back to V_rollout. If V_rollout hasn't been set either,
        falls back to 0.0. (Bug fix from Julia Session 5: unvisited Q_robust=0 was
        contaminating parent backups.)
        """
        if self.is_leaf():
            self.V_robust = self.V_rollout if self.has_rollout else 0.0
            return

        visited_qs = [an.Q_robust for an in self.children.values() if an.N > 0]

        if not visited_qs:
            self.V_robust = self.V_rollout if self.has_rollout else 0.0
        else:
            self.V_robust = max(visited_qs)

    def update_nominal_value(self) -> None:
        """Update cached V_nominal from children's Q_nominal values.

        Same fallback logic as update_robust_value.
        """
        if self.is_leaf():
            self.V_nominal = self.V_rollout if self.has_rollout else 0.0
            return

        visited_qs = [an.Q_nominal for an in self.children.values() if an.N > 0]

        if not visited_qs:
            self.V_nominal = self.V_rollout if self.has_rollout else 0.0
        else:
            self.V_nominal = max(visited_qs)
