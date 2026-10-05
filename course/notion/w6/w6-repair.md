# W6 Repair: Read the Probability Before the Formula

Use this companion when a symbol stops the explanation. The goal is to translate each operation into a question you can answer with a small table.

## A short dictionary

| Term | Meaning in our sensor story |
| --- | --- |
| Random variable | A quantity whose value can vary across possible outcomes |
| Observed value | A value already recorded: the first reading is 1 |
| Latent / hidden | A quantity not directly observed: the true first state |
| Joint probability | Probability of one complete combination of states and readings |
| Prior | Belief before the specified observations |
| Likelihood | Compatibility of the fixed readings with a proposed state/path |
| Posterior | Probability of explanations after conditioning on the readings |
| Evidence / normalizer | Total probability of the observed readings across explanations |
| Marginal | Distribution of selected variables after summing out the others |
| Expectation | Probability-weighted average of a function |
| Log | A monotone transformation turning products into sums; probabilities become nonpositive logs |
| Approximation | A computational representation of the target, with possible error |

The model describes a data-generating story. The posterior answers an inference question under that story. A sampling routine simulates values to compute an answer; it is not the physical measurement process.

## R1. Begin with one reading

Suppose the machine is initially equally likely to be inactive or active. The sensor reports 1.

| True state | Prior | Likelihood of reading 1 | Joint weight | Posterior |
| --- | --- | --- | --- | --- |
| 0 | 0.5 | 0.2 | 0.1 | 0.2 |
| 1 | 0.5 | 0.8 | 0.4 | 0.8 |

The normalizer is $0.1+0.4=0.5$. Divide the last-but-one column by 0.5. Now the two possibilities sum to one. The sensor is informative, but its reading is not proof of the true state.

If the prior probability of state one were 0.1 instead, the two joint weights would be 0.18 and 0.08. The posterior probability of state one would be $0.08/0.26\approx0.308$, despite the same sensor reading and accuracy. Likelihood and posterior are not interchangeable.

## R2. Normalize, then ask a specific question

For all three readings, add the eight joint weights in the main note to get 0.0872. Divide every row by that value. Then select the rows satisfying your query and add their posterior probabilities.

For the last-state query, select **001, 011, 101, 111**. Selecting only 111 answers a different question: whether all three hidden states were one.

The expectation notation is compact bookkeeping. Write eight rows, write the value of $f$ in each row, multiply by the row probability, and sum. For an indicator, the unwanted rows contribute zero. For a nonbinary function, such as the number of active times, the multipliers can be 0, 1, 2 or 3.

## R3. Work the two-entry message slowly

After the first reading, the unnormalized joint table is $[0.1,0.4]$. To reach state zero at the next time, we can come from zero and stay, or come from one and switch. Add these two routes: $0.1(0.8)+0.4(0.2)=0.16$. The second reading is zero, so multiply by the sensor likelihood 0.8 to obtain 0.128.

For state one, the two routes contribute $0.1(0.2)+0.4(0.8)=0.34$. Multiply by the likelihood of a zero reading when the state is one, 0.2, to obtain 0.068. The two resulting weights sum to 0.196, the probability of the first two readings. Normalize to get the filtered belief.

This message does not discard information needed by later steps of this chain. It collects all earlier paths with the same current state. The model's Markov assumption makes that current state sufficient for the next transition calculation.

## R4. Understand the ELBO without memorizing its acronym

The exact posterior is the target. The distribution $q$ is our adjustable approximation. The log evidence is fixed. The identity says:

**fixed height = achieved ELBO + remaining reverse-KL discrepancy.**

Improving the ELBO reduces the discrepancy. But a family of independent factors cannot produce dependence, regardless of how long we optimize. Also, an optimizer can stop at different local solutions. Do not call a flat optimization curve proof of an exact posterior.

Natural logs of numbers between zero and one are negative. An ELBO of −2.73 is larger than −3.77. “Higher” means less negative here. No claim that probabilities are negative is being made.

## R5. Separate the target from its numerical estimate

The posterior probability of the last state remains approximately 0.734 while we run the computer longer. A 100-draw estimate might be 0.71 or 0.78. Those fluctuations concern the algorithm's estimate. The hidden state still has possible values zero and one with the same posterior probabilities.

For independent draws, quadrupling the budget halves the standard deviation of the sample average. It does not halve posterior SD. Dependent MCMC samples do not automatically obey the independent-draw formula using their raw count.

## Self-check route

Return to Q1–Q4 in the main note first. If you can explain the normalizer, the four selected paths and the two-entry message aloud, continue to Q5–Q8 and the notebook. Use the separate checkpoint explanations only after trying.

This companion uses the original W6 sensor example. Its probabilistic identities follow the source map in the main note.
