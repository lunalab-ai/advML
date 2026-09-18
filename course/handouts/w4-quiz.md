# Variational Inference: ELBO, Mean Field, and Optimization · Quiz explanations

Original checkpoint explanations aligned with the W4 lecture; no graded assignment.

## 1. Why can ELBO maximization avoid evaluating the evidence during fixed-model inference?

The evidence is constant with respect to the variational parameters. Expanding reverse KL gives log evidence minus ELBO, so maximizing the computable bound minimizes that KL. An absolute numerical check still requires all normalization constants.

## 2. Does mean field assert posterior independence?

No. It constrains the approximating distribution to a product. A dependent target can have a converged mean-field approximation that misses covariance and misrepresents marginal uncertainty.

## 3. Why is the optimal Gaussian mean-field variance 1/Pjj rather than Cjj?

Holding other factors fixed leaves a quadratic in zj with precision Pjj. Its normalized factor has variance 1/Pjj, a conditional variance for a Gaussian. Cjj is the target marginal variance and generally differs.

## 4. What is guaranteed by an exact sequential CAVI update?

When the coordinate normalizer and expectations exist, replacing one factor by its exact coordinate optimum cannot decrease the exact ELBO. This does not guarantee a global optimum in general or an exact posterior within a restricted family.

## 5. Why does the log-standard-deviation gradient contain a +1?

For a diagonal Gaussian, entropy includes the sum of log standard deviations. Differentiating it with respect to each log standard deviation gives one. The likelihood/prior pathwise term alone is incomplete.

## 6. Can increasing iterations eliminate every ELBO gap?

No. It can reduce optimization error relative to the best member of the chosen family. The approximation gap remains unless the family can represent the posterior; at rho=.8 it is about .212340 nats here.

## 7. What distinguishes the W2 Ising variational objective from a log-evidence bound?

The expectation of the unnormalized log Ising density plus entropy lower-bounds log Z. Evidence requires an explicitly conditioned joint and its normalization; a partition function is not automatically an evidence probability.
