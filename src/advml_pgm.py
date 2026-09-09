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
    """Compute log(sum(exp(x))) stably without first exponentiating large logs.
    Parameters: x is a nonempty numeric array; axis=None reduces all entries,
    an integer axis reduces that dimension. Returns an ndarray with the reduced
    axis removed (a scalar-shaped array for axis=None). All -inf mass returns
    -inf, representing zero mass. Inputs are not changed. Example:
    logsumexp(np.log([2., 3.])) equals np.log(5.)."""
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    safe_m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide='ignore', invalid='ignore', under='ignore'):
        result = safe_m + np.log(np.exp(x - safe_m).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis) if axis is not None else result.reshape(())


def normalize_log(x: np.ndarray) -> np.ndarray:
    """Convert log weights x to normalized log probabilities of the same shape.
    x: numeric array with at least one finite log weight. All entries together
    are normalized, not each row separately. Returns x - logsumexp(x), without
    changing x. Zero total mass raises ValueError: impossible evidence cannot
    be replaced by a uniform posterior. Example: exp(normalize_log(log([2,3])))
    is [0.4,0.6]."""
    z = logsumexp(x)
    if not np.isfinite(z):
        raise ValueError('Zero total mass: evidence or constraints are impossible.')
    return np.asarray(x, dtype=float) - z


@dataclass
class PairwiseModel:
    """Finite undirected model represented by log potentials, not fitted parameters.
    log_unary: numeric (N,K) array, N nodes and K states per node, N,K >= 1.
    log_edges: dict {(u,v): (K,K) numeric array}, indexed [state_u,state_v].
    Node IDs are 0..N-1 and states are 0..K-1. Reverse keys are transposed
    to canonical order; duplicate undirected edges and self edges are invalid.
    The unnormalized log joint sums one unary entry per node and one edge entry
    per edge. Factors need not be normalized probability tables. Negative infinity
    encodes a hard zero; NaN and positive infinity raise ValueError. Arrays are
    copied at construction. Methods perform inference/conditioning; no training
    data or fit method is involved. Example: PairwiseModel(np.zeros((2,2)),
    {(0,1):np.log([[2.,1.],[1.,2.]])})."""
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
        """Return N, the number of nodes (first log_unary dimension); no arguments or mutation."""
        return self.log_unary.shape[0]

    @property
    def k(self) -> int:
        """Return K, the number of states per node; no arguments or mutation."""
        return self.log_unary.shape[1]

    def neighbors(self) -> list[list[int]]:
        """Return a new length-N list of sorted neighbor-ID lists.
        No arguments. Uses the stored undirected edges and does not change the graph.
        Example: a 0--1--2 chain returns [[1],[0,2],[1]]."""
        result = [[] for _ in range(self.n)]
        for u, v in sorted(self.log_edges):
            result[u].append(v); result[v].append(u)
        return result

    def edge(self, u: int, v: int) -> np.ndarray:
        """Return the log edge table oriented from node u to node v.
        u,v: distinct node IDs of an existing edge. Output shape (K,K), with u
        states in rows and v states in columns. Missing edges raise KeyError.
        The returned array may share storage with the model: treat it as read-only.
        edge(1,0) is edge(0,1).T, not a different undirected factor."""
        return self.log_edges[u, v] if u < v else self.log_edges[v, u].T

    def condition(self, evidence: dict[int, int]) -> PairwiseModel:
        """Return a new model clamped to evidence without changing this model.
        evidence: dict {node_id: observed_state}, IDs in 0..N-1, states in 0..K-1.
        Invalid indices raise ValueError. Inconsistent total mass is detected by
        the inference routine. Nonselected unary states receive -inf; the selected
        state keeps its original scale. The graph and N remain unchanged: clamping
        a cycle does not structurally turn it into a tree. Evidence probability is
        exp(log_z_clamped - log_z_original), not exp(log_z_clamped) in general."""
        unary = self.log_unary.copy()
        for node, state in evidence.items():
            if not (0 <= node < self.n and 0 <= state < self.k):
                raise ValueError('Invalid evidence index or state.')
            keep = unary[node, state]
            unary[node] = -np.inf
            unary[node, state] = keep
        return PairwiseModel(unary, self.log_edges)


def enumerate_exact(model: PairwiseModel, max_states: int = 262144) -> dict:
    """Enumerate the complete joint as an independent small-model oracle.
    model: PairwiseModel. max_states: maximum K**N assignments, default 262144.
    Returns dict: marginals (N,K) normalized probabilities; log_z float;
    states (K**N,N) integer assignments; probabilities (K**N,) normalized
    joint weights in corresponding order. No input is mutated. Excess states
    or zero total mass raise ValueError. Runtime O(K**N*(N+E)); joint storage
    is exponential. Use for verification, not large graphs. For clamped models,
    log_z retains unnormalized evidence mass and is not generally log P(e)."""
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
    """Run exact two-pass log-domain sum-product on a connected tree.
    model: PairwiseModel (single node allowed). root: starting node, default 0.
    Returns dict marginals (N,K), log_z float, and log_messages dict whose
    (u,v) entry has K values indexed by the receiver v's state. Messages
    retain scale so log_z is recoverable. Model is not changed. Invalid root,
    a cycle/disconnected graph, or zero total mass raises ValueError.
    Time O(N*K**2) at bounded degree, with O(K*sum(degree**2)) extra work
    from explicitly recomputing neighboring products. Different roots should
    produce the same marginals up to floating-point tolerance."""
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
    """Run synchronous approximate sum-product with probability-space damping.
    model: PairwiseModel. damping: OLD-message weight in [0,1), default 0.
    max_iter: positive iteration limit, default 300. tol: positive stopping
    threshold on maximum elementwise normalized-message change, default 1e-10.
    Returns dict marginals (N,K), residuals (iterations,), converged bool,
    and iterations int. Inputs are unchanged; messages start uniformly.
    Each update uses the previous iteration, not partially updated neighbors.
    No log_z estimate is claimed. A small residual is not a bound on marginal
    error on cycles. Invalid settings/impossible local mass raise ValueError.
    Compare accuracy with enumerate_exact only on tractable models."""
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
    """Construct a binary PairwiseModel from fields and couplings.
    fields: length-N numeric array of local fields h_i. couplings: dict
    {(u,v): J_uv} of scalar edge strengths. State 0 maps to spin -1 and
    state 1 to +1. Unary log potential is h_i*s_i; edge log potential is
    J_uv*s_u*s_v, so positive J favors agreement. Returns a new model;
    no fitting or random sampling occurs. Example: ising_model(np.zeros(2),
    {(0,1):0.8}) favors equal states without a local preference."""
    spins = np.array([-1.0, 1.0])
    return PairwiseModel(np.asarray(fields)[:, None] * spins,
                         {edge: j * np.outer(spins, spins) for edge, j in couplings.items()})


def max_marginal_tv(p: np.ndarray, q: np.ndarray) -> float:
    """Return max_i 0.5*sum_k(abs(p[i,k]-q[i,k])) as a float.
    p,q: matching normalized probability arrays of shape (N,K). The caller
    is responsible for nonnegative, unit-sum rows. Shape mismatch raises
    ValueError. Inputs are not changed. This compares single-node marginals,
    not joint distributions and not consecutive messages. Example: rows
    [0.2,0.8] versus [0.5,0.5] have TV 0.3."""
    p, q = np.asarray(p), np.asarray(q)
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError('Expected matching (N,K) probability arrays.')
    return float(np.max(0.5 * np.abs(p - q).sum(axis=1)))
