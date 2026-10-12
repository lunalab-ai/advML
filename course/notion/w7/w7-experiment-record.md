# W7 Experiment Record — Predict, Run, Explain

**October 13, 2026 · Formative practice · CPU · No upload or API key**

Do not begin by trying to get the “best” number. Begin by identifying what the number would tell you. Keep a short record in your own notebook. A changed observation set, a changed model and a changed computational budget are three different interventions.

## Getting started

1. Open [Lab A](https://colab.research.google.com/github/lunalab-ai/advML/blob/2026-fall-w7/notebooks/student/w7-a-predictive-models.ipynb) or [Lab B](https://colab.research.google.com/github/lunalab-ai/advML/blob/2026-fall-w7/notebooks/student/w7-b-bayesian-functions.ipynb), save a copy in Drive and select a CPU runtime.
2. Run cells from the top. Setup checks fixed source hashes and prepares the synthetic data. If it requests a runtime restart after changing packages, restart once and run all again.
3. Read the Markdown explanation before the code. Definition links lead to the exact version used by this lesson.
4. Unfinished exercise cells do not block the demonstrations. Record predictions first, then make your changes in those cells.
5. Lab B ends with the interactive app. Its share URL is temporary; keep the notebook URL as your permanent entry point.

## Observation sheet

| Field | Your record |
| --- | --- |
| Question that can be checked | |
| Quantity held fixed | |
| One quantity changed | |
| Prediction before execution | |
| Model and prior assumptions | |
| Data seed, sample size, noise SD | |
| Measured result and units | |
| What the plot adds to the number | |
| One limitation or alternative explanation | |

## A. Likelihood is a score for observed evidence

Use Lab A's Gaussian and Bernoulli cells. Calculate at least one likelihood contribution by hand. Then compare your result with the implemented summed NLL. Record both the summed and mean values, and explain why they are different.

Try the Gaussian example with a different assumed variance while keeping the observations and predicted means unchanged. Predict how each of the two NLL terms will change. Does retaining only squared error still compare the full predictive distributions?

In the Bernoulli example, check which event is observed in each case. Explain why probabilities for two different cases need not sum to one. Use a real-valued score and identify the inverse link that converts it to a valid probability.

## B. A distribution needs the right support

In the likelihood tab of the app, choose Gaussian, Bernoulli and Poisson in turn. Keep the score fixed, and read the resulting expected response. Explain what that expectation means for each family.

Try an invalid binary or count response. Record the error's modeling meaning rather than only its programming message. A count response must be a nonnegative integer; its expected value need not be one. Compare a Poisson zero with a structural-zero mixture conceptually.

## C. Same data, different predictive assumptions

Keep seed 19, 32 observations and noise SD 0.2. Compare the four app methods. Use the same held-out outcomes for every method. Record RMSE, mean NLL, coverage and width. Read all four region rows, especially the training gap and extrapolation region.

Describe a case where a narrow interval is misleading. State exactly what is Bayesian in the neural-linear method. For the point MLP, identify what its displayed future-response band includes and what it omits. Do not call repeated deterministic forward passes posterior samples.

## D. More observations under the same GP

Compare 16 and 48 training observations while fixing seed, noise and kernel settings. The generator retains the earlier pairs when sample size grows. Predict what happens to latent variance and to the difference between latent and response variance.

After running, distinguish the mathematical variance statement from a claim about error at every test input. A mean curve may move, and a particular test outcome may remain difficult even when conditional variance falls.

## E. Change the prior through the kernel

At a fixed dataset, compare length scales 0.3, 1.0 and 2.5. Before running, predict which setting permits faster local variation. Record the effect in the gap and outside the observed input range.

Do not choose a “winner” from these held-out outcomes and then report the same outcomes as an untouched final test. This is an exploratory sensitivity exercise. A performance-selection study would need a validation split and a separate final test.

## F. Compare two equivalent calculations

Use Lab B's weight-space and function-space experiment. Record the maximum mean and covariance discrepancies. Then change one prior setting in only one calculation and explain why agreement is no longer expected. Restore the matching setting before continuing.

Explain why a negative off-diagonal covariance can be valid. Use the positive-semidefinite condition rather than inspecting each entry's sign.

## G. Extend the app and export evidence

The notebook's `student_extra(metrics)` function receives a copy of a region's metric dictionary. Add a new diagnostic field with a new name, such as a quantity derived from coverage. Do not replace the existing metrics. Launch the app, change a control, click **Fit and compare**, and export the CSV. Check that your new field appears in the file as well as on screen.

Write a brief interpretation that includes the fixed data settings, the change you made, two numerical results, a visible pattern and one limitation. Describe how the comparison would change if you collected new real observations instead of drawing more posterior samples.

## Exit check

Without reading a formula, explain the difference between an observed response, a latent function value, a parameter draw and a predictive distribution. Then point to the two variance terms in your own equation and identify which plotted interval they create.

If you get stuck, return to Worked Calculation 4 before trying the GP matrix expression again. It contains the same essential distinction in one dimension.
