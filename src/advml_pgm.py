"""Small finite graphical models for Advanced ML, Fall 2026.

Independent teaching implementation of standard sum-product equations.
Reference: Murphy, Probabilistic Machine Learning: Advanced Topics, chapters 4/9.
States are integers 0..K-1. Pairwise tables are indexed [state_u, state_v].
The exact oracle is deliberately limited to small models. No project solution
(cycle-cutset reconstruction or student-designed correction) is supplied here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def logsumexp(x: np.ndarray, axis=None) -> np.ndarray:
    """Stable log(sum(exp(x))), including an all-negative-infinity slice."""
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    safe_m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide='ignore', invalid='ignore', under='ignore'):
        result = safe_m + np.log(np.exp(x - safe_m).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis) if axis is not None else result.reshape(())


def normalize_log(x: np.ndarray) -> np.ndarray:
    """Return normalized log probabilities; reject impossible evidence."""
    z = logsumexp(x)
    if not np.isfinite(z):
        raise ValueError('Zero total mass: evidence or constraints are impossible.')
    return np.asarray(x, dtype=float) - z


@dataclass
class PairwiseModel:
    """Log potentials for a finite undirected model, same K states per node.

    Edge keys may have either orientation; reversed keys transpose the table.
    Negative infinity encodes a hard zero. NaN and positive infinity are invalid.
    Arrays are copied so that construction does not mutate the caller's inputs.
    """
    log_unary: np.ndarray
    log_edges: dict[tuple[int, int], np.ndarray]

    def __post_init__(self):
        self.log_unary = np.array(self.log_unary, dtype=float, copy=True)
        if self.log_unary.ndim != 2 or min(self.log_unary.shape) < 1:
            raise ValueError('log_unary must have shape (N,K), with N,K >= 1.')
        canonical = {}
        for (u, v), values in self.log_edges.items():
            if not (0 <= u < self.n and 0 <= v < self.n) or u == v:
                raise ValueError('Edges must connect distinct valid nodes.')
            values = np.array(values, dtype=float, copy=True)
            if values.shape != (self.k, self.k):
                raise ValueError('Edge table must have shape (K,K).')
            if u > v:
                u, v, values = v, u, values.T
            if (u, v) in canonical:
                raise ValueError('Duplicate undirected edge.')
            canonical[u, v] = values
        self.log_edges = canonical
        for x in [self.log_unary, *canonical.values()]:
            if np.isnan(x).any() or np.isposinf(x).any():
                raise ValueError('Use finite log potentials or negative infinity.')

    @property
    def n(self) -> int:
        return self.log_unary.shape[0]

    @property
    def k(self) -> int:
        return self.log_unary.shape[1]

    def neighbors(self) -> list[list[int]]:
        """Adjacency lists in deterministic order."""
        result = [[] for _ in range(self.n)]
        for u, v in sorted(self.log_edges):
            result[u].append(v); result[v].append(u)
        return result

    def edge(self, u: int, v: int) -> np.ndarray:
        """Table with rows belonging to u and columns belonging to v."""
        return self.log_edges[u, v] if u < v else self.log_edges[v, u].T

    def condition(self, evidence: dict[int, int]) -> PairwiseModel:
        """Clamp states while retaining the original unnormalized mass scale.

        The graph is retained. This operation alone does not turn a cycle into a
        tree: reducing a clamped graph is part of the project, not this utility.
        """
        unary = self.log_unary.copy()
        for node, state in evidence.items():
            if not (0 <= node < self.n and 0 <= state < self.k):
                raise ValueError('Invalid evidence index or state.')
            keep = unary[node, state]
            unary[node] = -np.inf
            unary[node, state] = keep
        return PairwiseModel(unary, self.log_edges)


def enumerate_exact(model: PairwiseModel, max_states: int = 262144) -> dict:
    """Independent complete-joint oracle; O(K**N (N+E)), small models only.

    Returns marginals, log normalizer, joint probabilities and state assignments.
    For clamped models log_z is log unnormalized evidence mass, not log P(e)
    unless the original model's normalizer is one.
    """
    count = model.k ** model.n
    if count > max_states:
        raise ValueError(f'Oracle limited to {max_states} states; requested {count}.')
    states = np.indices((model.k,) * model.n).reshape(model.n, -1).T
    log_weights = model.log_unary[np.arange(model.n), states].sum(axis=1)
    for (u, v), potential in model.log_edges.items():
        log_weights += potential[states[:, u], states[:, v]]
    z = float(logsumexp(log_weights))
    if not np.isfinite(z):
        raise ValueError('Zero total mass: evidence or constraints are impossible.')
    probs = np.exp(log_weights - z)
    marginals = np.array([np.bincount(states[:, i], weights=probs, minlength=model.k)
                          for i in range(model.n)])
    return dict(marginals=marginals, log_z=z, states=states, probabilities=probs)


def tree_sum_product(model: PairwiseModel, root: int = 0) -> dict:
    """Exact two-pass log-domain BP on a connected tree (single node allowed).

    Messages retain their scale so log_z is available for evidence comparisons.
    Returns all one-node marginals and oriented log messages. With bounded degree,
    runtime is O(N K**2); this clear implementation recomputes neighbor products,
    adding O(K sum(degree**2)) work on high-degree trees.
    """
    if not 0 <= root < model.n:
        raise ValueError('Invalid root.')
    adjacency = model.neighbors()
    parent = {root: -1}; order = [root]
    for u in order:
        for v in adjacency[u]:
            if v == parent[u]:
                continue
            if v in parent:
                raise ValueError('tree_sum_product requires a connected tree.')
            parent[v] = u; order.append(v)
    if len(order) != model.n or len(model.log_edges) != model.n - 1:
        raise ValueError('tree_sum_product requires a connected tree.')
    messages = {}

    def send(u, v):
        local = model.log_unary[u].copy()
        for w in adjacency[u]:
            if w != v:
                local += messages[w, u]
        messages[u, v] = logsumexp(local[:, None] + model.edge(u, v), axis=0)

    for u in reversed(order[1:]):
        send(u, parent[u])
    root_belief = model.log_unary[root].copy()
    for w in adjacency[root]:
        root_belief += messages[w, root]
    log_z = float(logsumexp(root_belief))
    if not np.isfinite(log_z):
        raise ValueError('Zero total mass: evidence or constraints are impossible.')
    for u in order:
        for v in adjacency[u]:
            if parent.get(v) == u:
                send(u, v)
    beliefs = []
    for u in range(model.n):
        local = model.log_unary[u].copy()
        for w in adjacency[u]:
            local += messages[w, u]
        beliefs.append(np.exp(normalize_log(local)))
    return dict(marginals=np.array(beliefs), log_z=log_z, log_messages=messages)


def loopy_sum_product(model: PairwiseModel, *, damping: float = 0.0,
                      max_iter: int = 300, tol: float = 1e-10) -> dict:
    """Synchronous approximate BP baseline with probability-space damping.

    damping is the OLD-message weight (0 = undamped; must be below 1).
    Residual measures consecutive normalized message changes, NOT posterior error.
    A converged result on a cyclic graph need not be exact. No log_z is claimed.
    """
    if not 0 <= damping < 1 or max_iter < 1 or tol <= 0:
        raise ValueError('Require 0 <= damping < 1, max_iter >= 1, tol > 0.')
    adjacency = model.neighbors()
    messages = {(u, v): np.full(model.k, -np.log(model.k))
                for u in range(model.n) for v in adjacency[u]}
    residuals = []
    for _ in range(max_iter):
        updated = {}; residual = 0.0
        for (u, v), old in messages.items():
            local = model.log_unary[u].copy()
            for w in adjacency[u]:
                if w != v:
                    local += messages[w, u]
            new = normalize_log(logsumexp(local[:, None] + model.edge(u, v), axis=0))
            if damping:
                new = normalize_log(np.logaddexp(np.log(damping) + old,
                                                np.log1p(-damping) + new))
            residual = max(residual, float(np.max(np.abs(np.exp(new) - np.exp(old)))))
            updated[u, v] = new
        messages = updated; residuals.append(residual)
        if residual < tol:
            break
    beliefs = []
    for u in range(model.n):
        local = model.log_unary[u].copy()
        for w in adjacency[u]:
            local += messages[w, u]
        beliefs.append(np.exp(normalize_log(local)))
    return dict(marginals=np.array(beliefs), residuals=np.array(residuals),
                converged=residuals[-1] < tol, iterations=len(residuals))


def ising_model(fields: np.ndarray, couplings: dict[tuple[int, int], float]) -> PairwiseModel:
    """Binary Ising model: state 0=-1, state 1=+1; positive J favors agreement."""
    spins = np.array([-1.0, 1.0])
    return PairwiseModel(np.asarray(fields)[:, None] * spins,
                         {edge: j * np.outer(spins, spins) for edge, j in couplings.items()})


def max_marginal_tv(p: np.ndarray, q: np.ndarray) -> float:
    """Maximum single-node total variation distance (not joint TV)."""
    p, q = np.asarray(p), np.asarray(q)
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError('Expected matching (N,K) probability arrays.')
    return float(np.max(0.5 * np.abs(p - q).sum(axis=1)))
