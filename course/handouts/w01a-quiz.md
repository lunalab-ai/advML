# Course Orientation and the Probabilistic View of Advanced Machine Learning · Quiz explanations

Questions and compact answers reproduced from the instructor-provided Class 01 Notion archive. Full bilingual speaker scripts remain private.

## 1. What is the difference between $p(y\mid x)$ and $p(x\mid y)$?

They condition in opposite directions and are related by Bayes' rule, not by symmetry.

## 2. Why does independence imply zero covariance, while zero covariance does not generally imply independence?

Covariance captures only linear dependence; nonlinear dependence may remain.

## 3. Why is minimizing cross-entropy often equivalent to maximum likelihood estimation?

Cross-entropy is the NLL for Bernoulli or categorical observation models.

## 4. How does a prior become a regularization term in MAP estimation?

The negative log-prior is added to the NLL.

## 5. What is the difference between aleatoric and epistemic uncertainty?

Aleatoric uncertainty is intrinsic data variability; epistemic uncertainty reflects limited knowledge.

## 6. Why can a large learning rate cause divergence even for a convex objective?

The update can overshoot along directions of high curvature and oscillate or diverge.

## 7. What is lost when $p(\theta\mid\mathcal{D})$ is replaced by a point estimate $\hat\theta$?

Parameter uncertainty and model averaging are discarded, often producing overconfident predictions.
