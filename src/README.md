# Reusable course code

`advml_pgm.py` begins the course's accumulated inference code. It is independently authored for W2, following standard equations in Murphy, Chapters 4 and 9. It uses only NumPy. See the module's docstrings for potential orientation, state indexing, log scales and the distinction between exact and approximate inference.

```python
from advml_pgm import ising_model, tree_sum_product
model = ising_model([0.2, -0.1, 0.0], {(0, 1): 0.6, (1, 2): -0.4})
result = tree_sum_product(model)
print(result['marginals'], result['log_z'])
```

The W2 notebooks download this module from the immutable-by-convention `2026-fall-w2-code` Git tag and verify its SHA-256. Later lessons should add compatible functions or modules and pin a new reviewed revision without moving old tags. `enumerate_exact` has a deliberate state-count cap; it is a small-model verification tool. Tree BP rejects cyclic/disconnected graphs. Loopy BP returns convergence information without claiming an exact partition function. The W2 project correction and extension are not part of this public library.

Tested locally with Python 3.12, NumPy 2.2.6 and float64. Notebook figures additionally use Matplotlib 3.10.3. The notebook prints the actual runtime versions; local execution is recorded separately from the Colab user interface.
