# Course API reference

Arguments, defaults, shapes and state are copied from reviewed implementation docstrings. Definition links use the same release revision.

## logsumexp

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L14)

```python
logsumexp(x: np.ndarray, axis=None)
```

Compute log(sum(exp(x))) stably without first exponentiating large logs.
Parameters: x is a nonempty numeric array; axis=None reduces all entries,
an integer axis reduces that dimension. Returns an ndarray with the reduced
axis removed (a scalar-shaped array for axis=None). All -inf mass returns
-inf, representing zero mass. Inputs are not changed. Example:
logsumexp(np.log([2., 3.])) equals np.log(5.).

## normalize_log

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L29)

```python
normalize_log(x: np.ndarray)
```

Convert log weights x to normalized log probabilities of the same shape.
x: numeric array with at least one finite log weight. All entries together
are normalized, not each row separately. Returns x - logsumexp(x), without
changing x. Zero total mass raises ValueError: impossible evidence cannot
be replaced by a uniform posterior. Example: exp(normalize_log(log([2,3])))
is [0.4,0.6].

## PairwiseModel

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L43)

```python
PairwiseModel
```

Finite undirected model represented by log potentials, not fitted parameters.
log_unary: numeric (N,K) array, N nodes and K states per node, N,K >= 1.
log_edges: dict {(u,v): (K,K) numeric array}, indexed [state_u,state_v].
Node IDs are 0..N-1 and states are 0..K-1. Reverse keys are transposed
to canonical order; duplicate undirected edges and self edges are invalid.
The unnormalized log joint sums one unary entry per node and one edge entry
per edge. Factors need not be normalized probability tables. Negative infinity
encodes a hard zero; NaN and positive infinity raise ValueError. Arrays are
copied at construction. Methods perform inference/conditioning; no training
data or fit method is involved. Example: PairwiseModel(np.zeros((2,2)),
{(0,1):np.log([[2.,1.],[1.,2.]])}).

## PairwiseModel.n

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L80)

```python
n(self)
```

Return N, the number of nodes (first log_unary dimension); no arguments or mutation.

## PairwiseModel.k

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L85)

```python
k(self)
```

Return K, the number of states per node; no arguments or mutation.

## PairwiseModel.neighbors

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L89)

```python
neighbors(self)
```

Return a new length-N list of sorted neighbor-ID lists.
No arguments. Uses the stored undirected edges and does not change the graph.
Example: a 0--1--2 chain returns [[1],[0,2],[1]].

## PairwiseModel.edge

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L98)

```python
edge(self, u: int, v: int)
```

Return the log edge table oriented from node u to node v.
u,v: distinct node IDs of an existing edge. Output shape (K,K), with u
states in rows and v states in columns. Missing edges raise KeyError.
The returned array may share storage with the model: treat it as read-only.
edge(1,0) is edge(0,1).T, not a different undirected factor.

## PairwiseModel.condition

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L106)

```python
condition(self, evidence: dict[int, int])
```

Return a new model clamped to evidence without changing this model.
evidence: dict {node_id: observed_state}, IDs in 0..N-1, states in 0..K-1.
Invalid indices raise ValueError. Inconsistent total mass is detected by
the inference routine. Nonselected unary states receive -inf; the selected
state keeps its original scale. The graph and N remain unchanged: clamping
a cycle does not structurally turn it into a tree. Evidence probability is
exp(log_z_clamped - log_z_original), not exp(log_z_clamped) in general.

## enumerate_exact

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L124)

```python
enumerate_exact(model: PairwiseModel, max_states: int=262144)
```

Enumerate the complete joint as an independent small-model oracle.
model: PairwiseModel. max_states: maximum K**N assignments, default 262144.
Returns dict: marginals (N,K) normalized probabilities; log_z float;
states (K**N,N) integer assignments; probabilities (K**N,) normalized
joint weights in corresponding order. No input is mutated. Excess states
or zero total mass raise ValueError. Runtime O(K**N*(N+E)); joint storage
is exponential. Use for verification, not large graphs. For clamped models,
log_z retains unnormalized evidence mass and is not generally log P(e).

## tree_sum_product

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L149)

```python
tree_sum_product(model: PairwiseModel, root: int=0)
```

Run exact two-pass log-domain sum-product on a connected tree.
model: PairwiseModel (single node allowed). root: starting node, default 0.
Returns dict marginals (N,K), log_z float, and log_messages dict whose
(u,v) entry has K values indexed by the receiver v's state. Messages
retain scale so log_z is recoverable. Model is not changed. Invalid root,
a cycle/disconnected graph, or zero total mass raises ValueError.
Time O(N*K**2) at bounded degree, with O(K*sum(degree**2)) extra work
from explicitly recomputing neighboring products. Different roots should
produce the same marginals up to floating-point tolerance.

## loopy_sum_product

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L202)

```python
loopy_sum_product(model: PairwiseModel, *, damping: float=0.0, max_iter: int=300, tol: float=1e-10)
```

Run synchronous approximate sum-product with probability-space damping.
model: PairwiseModel. damping: OLD-message weight in [0,1), default 0.
max_iter: positive iteration limit, default 300. tol: positive stopping
threshold on maximum elementwise normalized-message change, default 1e-10.
Returns dict marginals (N,K), residuals (iterations,), converged bool,
and iterations int. Inputs are unchanged; messages start uniformly.
Each update uses the previous iteration, not partially updated neighbors.
No log_z estimate is claimed. A small residual is not a bound on marginal
error on cycles. Invalid settings/impossible local mass raise ValueError.
Compare accuracy with enumerate_exact only on tractable models.

## ising_model

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L246)

```python
ising_model(fields: np.ndarray, couplings: dict[tuple[int, int], float])
```

Construct a binary PairwiseModel from fields and couplings.
fields: length-N numeric array of local fields h_i. couplings: dict
{(u,v): J_uv} of scalar edge strengths. State 0 maps to spin -1 and
state 1 to +1. Unary log potential is h_i*s_i; edge log potential is
J_uv*s_u*s_v, so positive J favors agreement. Returns a new model;
no fitting or random sampling occurs. Example: ising_model(np.zeros(2),
{(0,1):0.8}) favors equal states without a local preference.

## max_marginal_tv

[src/advml_pgm.py · definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w2-explained/src/advml_pgm.py#L259)

```python
max_marginal_tv(p: np.ndarray, q: np.ndarray)
```

Return max_i 0.5*sum_k(abs(p[i,k]-q[i,k])) as a float.
p,q: matching normalized probability arrays of shape (N,K). The caller
is responsible for nonnegative, unit-sum rows. Shape mismatch raises
ValueError. Inputs are not changed. This compares single-node marginals,
not joint distributions and not consecutive messages. Example: rows
[0.2,0.8] versus [0.5,0.5] have TV 0.3.
