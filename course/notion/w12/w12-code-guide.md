# W12 · Lab Code Guide

Save a fresh copy of each notebook. Select a CPU runtime. Run each notebook from its own initial cell.

## A · Exact calculations

- **A01 / S02** Edit the prior vector in the same cell as the joint calculation. Compare the two joint masses before normalization.
- **A02 / S03** Change the mass on state 1 and read the graph and both KL fields. Keep the likelihood and model prior fixed.
- **A03–A04 / S04** Repair the scale line in the notebook body. Keep the same noise in both finite-difference evaluations. Explain the sign before trying the opposite noise.

## B · Training and the decoder

- **B01 / S05** An observation batch is [batch, 256]. The encoder mean and log variance are [batch, 2]. The decoder returns logits of shape [batch, 256].
- **B02 / S05** Look inside all three executable functions. The training loop receives the functions defined in this cell. Sum over pixels and latent coordinates separately. Average over the batch after forming a per-image loss.
- **B03–B04 / S06** Use the comparison images and metrics already printed. Change only the line for TRIAL_BETA and rerun that cell. Inspect per-pixel NLL and per-image KL separately.
- **B05 / S07** Edit the indices of the two endpoints. Three panels use encoder means, prior draws, and points connecting encoder means.
- **B06 / S07** Move one slider, compare the probability and observation panels, and click the export button before moving the other slider. Colab downloads a CSV. In a local kernel without widgets the coordinates are editable in the cell body.

The fitting helper contains seeded minibatches, Adam updates and two separate train/evaluation arrays. The core Gaussian transform and training loss are shown in the notebook before training. A copied module is verified against the fixed release hash during setup. See the code itself for the exact training loop.

[Open Lab A](https://colab.research.google.com/github/lunalab-ai/advML/blob/2026-fall-w12/notebooks/student/w12-a-exact-inference.ipynb) · [Open Lab B](https://colab.research.google.com/github/lunalab-ai/advML/blob/2026-fall-w12/notebooks/student/w12-b-train-a-vae.ipynb)
