"""
Per-node SAMPLED-projection tree (scheme="sampler") — the third support mechanism
alongside the uniform / exp_decay masks of fixed_policy_tree.

At every node the support is drawn from that node's own belief-predictive:
    S_in(root)     ~ b0
    Z_in(h, a)     ~ P(z | b, a)         (observation marginal)
    S_in(child_z)  ~ P(s' | b, a, z)     (one-step posterior)
sampled WITH replacement then deduped, so |support| <= K and peaked ("promising")
beliefs concentrate the support — the POMCP-style focusing.

Beliefs are option-B projected-Bayes (root = unnormalized restriction of b0;
children = posterior restricted to the sampled child support, renormalized), so
the SAME backup / robust_q / leakage_delta machinery applies unchanged. The only
structural point vs a uniform mask: each action node's next-state support is
S_next = union_z child_z.S_in (paper Eq. 8), set after its children are sampled.

Mirrors A1's sampled_projection.py (same rng.choice draw order, so a shared seed
reproduces A1's supports) but on the NEW core: new terminal convention
(depth == H-1 terminal) and new S_next.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from Algorithm.core.pomdp import TabularPOMDP
from Algorithm.experiments.post_hoc_guarantees.support_picker import sample_pick
from Algorithm.core.tree import HistoryNode, NodeIdCounter
from Algorithm.core.uncertainty import UncertaintySets
from Algorithm.ucb.backup import backup
from Algorithm.ucb.belief import restrict

Policy = Callable[[tuple], int]   # obs_path (z1, ..., zk) -> action


def eval_sampled_policy(model: TabularPOMDP,
                        uncertainty: UncertaintySets,
                        policy: Policy,
                        b0: np.ndarray,
                        H: int,
                        K: int,
                        rng: np.random.Generator,
                        abort_action: int | None = None,
                        lp_cache: dict | None = None,
                        ) -> HistoryNode:
    """Build the per-node sampled tree following `policy` and back it up.

    Returns the root HistoryNode (V_rob + survival c populated). Share one
    `lp_cache` across (K, seed) calls to reuse the persistent LPs.
    """
    if lp_cache is None:
        lp_cache = {}
    b0 = np.asarray(b0, dtype=np.float64)
    N = model.n_states
    ctr = NodeIdCounter()

    def _expand(depth: int, obs_path: tuple,
                S_in: list[int], belief: np.ndarray) -> HistoryNode:
        # `belief` aligned with sorted `S_in` (root: unnormalized restriction of
        # b0; child: normalized over its sampled support).
        node = HistoryNode(ctr.next_id(), depth)
        node.S_in = set(S_in)
        a = policy(obs_path)
        terminal = (depth == H - 1) or (abort_action is not None and a == abort_action)
        an = node.expand_action(a, ctr)
        if depth == 0:                          # root belief exact -> exact immediate reward
            an.r_exact = float(b0 @ model.R[:, a])
        if not terminal:
            b_full = np.zeros(N, dtype=np.float64)
            b_full[S_in] = belief
            next_marginal = b_full @ model.T[:, a, :]      # Σ_s b(s) T(s,a,·)   (len N)
            obs_marginal = next_marginal @ model.O          # Σ_s' next(s') O(s',·) (len M)
            Z_in = sample_pick(obs_marginal, K, rng)
            an.Z_in = set(Z_in)
            for z in Z_in:
                post_full = next_marginal * model.O[:, z]    # Σ_s b(s) T(s,a,s') O(s',z)
                S_in_child = sample_pick(post_full, K, rng)
                if not S_in_child:
                    continue
                child_belief = post_full[S_in_child] / post_full[S_in_child].sum()
                an.children[z] = _expand(depth + 1, obs_path + (z,),
                                         S_in_child, child_belief)
            an.S_next = set().union(*(c.S_in for c in an.children.values())) \
                if an.children else set()
        backup(node, belief, model, uncertainty, H, abort_action, lp_cache)
        return node

    S_in_root = sample_pick(b0, K, rng)
    return _expand(0, (), S_in_root, restrict(b0, S_in_root))


def avg_support_fraction(root: HistoryNode, N: int, M: int) -> tuple[float, float]:
    """(mean |S_in(h)|/N over history nodes, mean |Z_in(h,a)|/M over expanded
    action nodes). The sampler's x-axis; matches A1's _support_fractions."""
    s_fracs: list[float] = []
    z_fracs: list[float] = []

    def _walk(node: HistoryNode) -> None:
        s_fracs.append(len(node.S_in) / N)
        for an in node.children.values():          # 0 or 1 action node (fixed policy)
            if an.Z_in:                            # non-terminal (observations sampled)
                z_fracs.append(len(an.Z_in) / M)
            for child in an.children.values():
                _walk(child)

    _walk(root)
    return (float(np.mean(s_fracs)) if s_fracs else 0.0,
            float(np.mean(z_fracs)) if z_fracs else 0.0)
