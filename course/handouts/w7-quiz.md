# From Predictive Models to Bayesian Functions · Quiz explanations

Original W7 formative checkpoints aligned with the main note; model identities follow its textbook reading map.

## 1. Does drawing more posterior functions give us more observed training examples?

No. The observed data remain fixed. Draws represent plausible functions and are used to calculate posterior quantities.

## 2. Under which condition does Gaussian negative log likelihood select the same mean parameters as squared error?

For a common fixed positive observation variance, with the same mean model and data. Variance-dependent terms cannot be discarded when comparing or learning changing variances.

## 3. In logistic regression, which function is the link and which is the inverse link?

The logit maps the Bernoulli mean probability to a real score and is the link. The sigmoid maps the score back to a probability and is the inverse link.

## 4. Does running the same deterministic fitted network repeatedly produce posterior samples?

No. Fixed inputs and fixed weights return the same result. Posterior sampling requires an inference procedure for a parameter or function distribution.

## 5. If we keep the data and posterior fixed but use more Monte Carlo draws, which uncertainty should decrease?

The numerical error of a suitable Monte Carlo estimate typically decreases. The posterior spread and observation-noise variance are not reduced by additional computation alone.

## 6. Why is the interval for a future noisy observation wider than the interval for its latent function value?

Under independent additive Gaussian noise, future-observation variance equals latent-function variance plus the positive observation-noise variance. Both intervals have the same conditional mean.

## 7. What must match when comparing weight-space and function-space predictions?

The feature map, prior mean and covariance, noise model, observed data and prediction inputs. The kernel must be induced by that same feature prior.

## 8. Can a Bayesian model have narrow intervals and poor coverage? Explain one reason.

Yes. A misspecified function family, likelihood, prior or fixed hyperparameter choice can produce confident but systematically wrong predictions. Integrating weights does not validate those assumptions.
