# W6 Experiment Record

These are formative activities. No new graded assignment is introduced. Work with one partner: one person predicts, the other runs; then swap roles.

## Before every comparison

Record the observations, model parameters, query, method, computation budget, seed and exact reference. If two runs use different targets, explain why before comparing their errors. Export the web app CSV and write an interpretation alongside it.

## E1. Read a table

Using the eight-path table, calculate the probability that the **middle** state is one. List the included paths. Compare this full-data answer with the time-2 filtered probability and explain the difference in information available.

Prediction: ______. Selected paths: ______. Calculation: ______. Explanation: ______.

## E2. What can the variational family represent?

Run VI at stay probabilities 0.5, 0.8 and 0.95. Keep sensor accuracy and observations fixed. At 0.95 repeat initial factor probabilities 0.1, 0.5 and 0.9. Record last-state error, covariance error and final ELBO.

Explain separately: (a) a difference caused by initialization; (b) a restriction shared by every member of this family. Do not infer a global optimum from the highest value observed in a few runs.

| Stay | Initialization | Exact last-state probability | VI estimate | Covariance error | ELBO |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 0.5 | | | | |
| 0.8 | 0.5 | | | | |
| 0.95 | 0.1 / 0.5 / 0.9 | | | | |

## E3. Does more computation mean less uncertainty?

Keep the default model fixed. Repeat iid MC with budgets 25, 100 and 400 over at least 30 seeds. Compare the root mean squared error of the estimated event probability. Also record the exact posterior SD of the last state.

Which quantity changed? Which stayed fixed? Is a single lucky run enough to establish a general accuracy claim? Explain why the iid MCSE formula cannot simply use the raw number of MH draws.

## E4. Extend the web application

In the notebook, edit `student_extra(metrics)` to return a new field named `covariance_error`. Compute the absolute difference between the available exact and estimated covariances. Keep the original fields unchanged.

Launch the app, compare Exact and VI, and download the CSV. Confirm that your field appears in both the displayed record and the downloaded file. Explain what this diagnostic reveals that an event-probability error alone might miss.

## E5. Optional sequential extension

Use 30 and 400 particles over several seeds. At time 2, compare with the **time-2 exact filter**, not the full-data smoother. Describe predict, weight and resample in your own words. Explain why repeated binary state values do not measure the number of distinct ancestors.

## Make a claim that matches your evidence

Complete this template: “For the fixed readings ____, model settings ____, query ____ and seeds ____, method ____ had ____. The experiment does not establish ____.”

## Exit explanation

Without looking at equations, explain to a partner: observations → model → posterior → query → approximation → evidence for the claim. Mark the one arrow you still find unclear and bring that question to the next class.
