# Making Sense of Probabilistic Inference: A Weeks 1-5 Rebuild · Quiz explanations

Original W6 formative checkpoints; probabilistic identities follow the reading map in the lesson.

## 1. Does drawing 1,000 more posterior samples give us 1,000 new sensor observations?

No. The observations stay fixed. Samples are simulated possible hidden states used to calculate an answer.

## 2. What does the normalizing denominator add up?

The joint weights of every possible hidden path with the observed readings held fixed. Their total is the evidence, 0.0872 in this example.

## 3. Why is the probability of the MAP path different from the probability that the last state is one?

The MAP path is one complete event, 111. The last-state event includes 001, 011, 101 and 111; their probabilities must be added.

## 4. At time 2, why must an online particle filter be compared with a filtering reference rather than the full-data smoother?

They must use the same information. The filter uses readings 1 and 2, while the smoother also uses reading 3.

## 5. Can more optimization always remove the error caused by a factorized variational family?

No. Optimization changes parameters within the family. It cannot produce dependence that the family excludes; local optimization can introduce an additional gap.

## 6. Which uncertainty shrinks when we draw more samples while keeping the observed data fixed?

Typical numerical error of the estimator shrinks. Posterior spread remains fixed. The usual inverse-square-root sample-count formula assumes independent draws and finite variance.

## 7. Why must a rejected MH proposal remain as a repeated state in the chain?

The chain stays in its current state on rejection. Its residence times contribute to the intended stationary distribution; deleting repetitions changes the represented distribution.

## 8. What must stay fixed to interpret a comparison as an algorithm comparison?

The observed data, model parameters, target and query. Also record budgets, seeds and the metric; changing the target answers a different comparison question.
