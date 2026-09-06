# Mathematical Foundations for Probabilistic Machine Learning · Quiz explanations

Added checkpoint explanations aligned with the user-provided Class 02 material. Complete teaching scripts remain private.

## 1. Why is a likelihood not generally a probability distribution over parameters?

Likelihood treats the observed data as fixed and varies the parameter. Its integral over parameters need not equal one. A posterior requires a prior and normalization.

## 2. What is the mathematical difference between marginalizing $Y$ and conditioning on $Y=y$?

Marginalization sums or integrates the joint distribution over Y. Conditioning restricts to Y=y and normalizes by its probability or density when defined.

## 3. Give one situation in which $X$ and $Y$ are marginally dependent but conditionally independent given $Z$.

If Z describes a shared cause, X and Y may be correlated marginally yet independent given Z. For example, independent measurement errors around a known shared signal have this structure.

## 4. State the law of total variance and interpret its two terms.

Var(X) = E[Var(X|Z)] + Var(E[X|Z]). The terms describe average within-condition uncertainty and variation of conditional means.

## 5. For a Bernoulli model, why does the MLE equal the observed success frequency?

The Bernoulli log-likelihood is k log(p) + (n-k) log(1-p). Setting its derivative to zero yields p=k/n for an interior optimum; k=0 and k=n yield boundary optima.

## 6. How does a Gaussian prior become an $\ell_2$ regularizer in MAP estimation?

The negative log of a zero-mean isotropic Gaussian prior contributes a constant plus a positive multiple of the squared Euclidean norm of the parameter. MAP minimizes the negative log-likelihood plus this penalty.

## 7. What information does the Hessian provide that the gradient does not?

The Hessian gives second-order local curvature, including direction-dependent curvature and cross-parameter interactions. The gradient gives first-order slope.

## 8. For $L(w)=\frac12\lambda w^2$, what learning-rate range guarantees convergence of gradient descent?

For positive lambda, gradient descent gives w_next=(1-eta*lambda)w. Convergence for arbitrary initial w requires |1-eta*lambda|<1, hence 0<eta<2/lambda.

## 9. Why is a minibatch gradient useful even though it is noisy?

A minibatch gradient is cheaper per step and can estimate the full-data gradient under appropriate sampling. Its noise is traded against computational efficiency and variance.

## 10. In Bayesian linear regression, which part of predictive variance is epistemic?

The term x_star^T S_N x_star is epistemic uncertainty from the posterior covariance of the coefficients. The observation-noise variance is aleatoric.
