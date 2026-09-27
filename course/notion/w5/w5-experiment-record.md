# Sampling Experiment Record

Advanced Machine Learning · Week 5 · September 29, 2026

Use this worksheet with Lab A, Lab B, or Sampling Explorer. These are formative activities; no new graded assignment is introduced.

## 1. State the question before changing a slider

**W1.** Write the target, the function or filtering quantity, and one falsifiable prediction.

| Field | Your record |
|---|---|
| Target / observed data | |
| Quantity to estimate | |
| Exact reference available in this benchmark | |
| One setting to change | |
| Settings and data held fixed | |
| Prediction and reason | |
| Seeds and repetition count | |

Examples of experiment types: increase iid sample count; shift an IS proposal; vary MH step size; compare SIS with adaptive resampling at fixed observations. Changing observation noise also changes the statistical problem, so label it separately.

## 2. Keep cost and accuracy separate

**W2.** Fill at least two rows from controlled runs. Export CSV from the app and keep the file with the notebook.

| Run | Method / changed setting | Estimate and exact reference | Error | Relevant diagnostic | Cost and seed |
|---|---|---|---|---|---|
| Baseline | | | | | |
| Comparison | | | | | |
| Repeated-seed summary | | | | | |

For fixed-target experiments, record both the first moment and second moment. For MCMC, record post-warmup draws per chain, chain count, warmup, proposal scale, initialization, rank R-hat, ESS, and MCSE for the quantity of interest. For IS, record proposal settings and **weight** ESS.

Do not label direct iid oracle draws, target-density evaluations, conditional draws, and wall-clock seconds as identical budgets. State what was counted. Report repeated-seed behavior rather than selecting only the best run.

For filtering, use the same observations when comparing particle counts or thresholds. Record RMSE against the Kalman mean separately from RMSE against latent truth. Record resampling events and surviving initial ancestors as well as weight ESS.

## 3. Explain a failure, not just a score

**W3.** Complete this short evidence paragraph:

> I estimated __________ under target __________. I changed __________ while holding __________ fixed. I predicted __________ because __________. Across __________ seeds, I observed __________. The plot shows __________. The diagnostic __________ is informative about __________ but cannot establish __________. My next test would be __________.

A convincing explanation identifies the mechanism: missed support, unstable weights, slow movement, rejected-state mishandling, missed modes, weight degeneracy, or ancestor loss. A result that contradicts your prediction is useful when you explain it honestly.

## Final app activity

Open [Lab B](https://colab.research.google.com/github/lunalab-ai/advML/blob/2026-fall-w5/notebooks/student/w5-b-sequential-monte-carlo.ipynb), run the final Sampling Explorer cell, and inspect both tabs. Use the [code guide](w5-code-guide.md) to follow control → callback → algorithm → returned plot/CSV.

The generated Gradio URL is temporary. Keep the notebook link, your saved copy, the exact code tag, and the exported CSV. A screenshot alone omits settings and numerical data needed for reproduction.
