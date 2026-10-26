# [W9] Midterm Review and Board Notes

Advanced Machine Learning for Artificial Intelligence · October 27, 2026

Use the writing areas to record the board work and your own calculations. The original exam contains five questions worth 20 points each, with three parts per question (100 points in total).

Show working. Under the original marking policy, an entire question with no working earns no credit. With partial working, marks are awarded for the steps actually demonstrated. Consistent follow-through of an earlier arithmetic error is distinguished from a new conceptual error. Equivalent expressions and the original rounding allowances remain valid.

Marking is planned to finish this week. Results will be announced after marking is complete.

## Q1(a) · Exact Inference on a Small Chain · 8 points

Let $A\to B\to C$ be a binary chain, with states $0,1$ and $p(A)=(0.5,0.5)$. The transition tables are

$$
T_{AB}=\begin{pmatrix}0.8&0.2\\0.4&0.6\end{pmatrix},\qquad
T_{BC}=\begin{pmatrix}0.75&0.25\\0.25&0.75\end{pmatrix}.
$$

Rows index the source state; columns index the destination state. A fixed observation $e$ depends only on $C$, with likelihoods $(p(e\mid C=0),p(e\mid C=1))=(0.2,1)$. Keep messages unnormalized.

Compute the left message $m_L(b)=p(B=b)$ and the right message $m_R(b)=p(e\mid B=b)$ for $b=0,1$. Show the state-indexed sums.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Left-message sums | Correct state-indexed setup; partial credit for one correctly set-up state. | 2 |
| Left-message values | One point per state; arithmetic and consistent follow-through assessed separately. | 2 |
| Right-message sums | Correct transition row and evidence likelihood for each B state. | 2 |
| Right-message values | One point per original unnormalized value; original values must remain identifiable. | 2 |

### Board notes and calculations

## Q1(b) · Exact Inference on a Small Chain · 6 points

Use the same setting as Q1(a).

Use the messages to compute the weights $w_b=p(B=b,e)$ and the evidence $p(e)$.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Message combination | Correct same-state combination for both states; partial credit for one state. | 2 |
| Joint weights | One point per state; consistent arithmetic follow-through is recognized. | 2 |
| Evidence | Correct aggregation setup and calculation, one point each. | 2 |

### Board notes and calculations

## Q1(c) · Exact Inference on a Small Chain · 6 points

Use the same setting as Q1(a).

Compute $p(B=0\mid e)$ and $p(B=1\mid e)$, then verify that they sum to one.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Posterior for state 0 | Common normalization and resulting value, one point each. | 2 |
| Posterior for state 1 | Common normalization and resulting value, one point each; equivalent complement calculation accepted. | 2 |
| Normalization check | Explicit addition and valid verification, one point each; stating the property alone earns at most one point. | 2 |

### Board notes and calculations

## Q2(a) · Mean-Field Variational Inference · 8 points

The exact posterior is $p(z\mid x)=\mathcal N(\mu,P^{-1})$. Approximate it by $q(z)=\mathcal N(m,\operatorname{diag}(v_1,v_2))$, where
$$
P=\begin{pmatrix}2&-1\\-1&2\end{pmatrix},\qquad
\mu=\begin{pmatrix}1\\-1\end{pmatrix},\qquad m^{(0)}=\begin{pmatrix}0\\0\end{pmatrix}.
$$
Use the Gaussian CAVI updates
$$
v_j=\frac{1}{P_{jj}},\qquad
m_j\leftarrow\mu_j-\frac{P_{jk}}{P_{jj}}(m_k-\mu_k),\quad k\ne j.
$$
All logarithms are natural. Show the substitutions and intermediate calculations.

Compute $C=P^{-1}$, the two exact marginal variances, the optimal diagonal variances $v_1^*,v_2^*$, and each shortfall $C_{jj}-v_j^*$.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Determinant and inverse setup | Correct determinant or equivalent elimination; method credit retained for a single arithmetic slip. | 2 |
| Covariance and marginal variances | Correct matrix structure and identification of its diagonal; consistent arithmetic follow-through accepted. | 2 |
| Optimal diagonal variances | One point per coordinate: justified substitution and calculation. | 2 |
| Variance shortfalls | One point per coordinate: requested subtraction direction and calculation. | 2 |

### Board notes and calculations

## Q2(b) · Mean-Field Variational Inference · 8 points

Use the same setting as Q2(a).

Starting from $m^{(0)}$, perform one sequential sweep: update coordinate 1, then coordinate 2 using the new coordinate 1. Report both updated means.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| First-coordinate substitution | Use the supplied update with the initial second coordinate; partial method credit where applicable. | 2 |
| First-coordinate calculation | Show the calculation; distinguish a new arithmetic slip from an already assessed error. | 2 |
| Sequential input to coordinate 2 | Use the newly updated first coordinate; consistent follow-through accepted. | 2 |
| Second-coordinate calculation | Assess subsequent arithmetic separately from the choice of sequential input. | 2 |

### Board notes and calculations

## Q2(c) · Mean-Field Variational Inference · 4 points

Use the same setting as Q2(a).

For this model, $\log p(x)=-3.0000$, the exact diagonal-family optimum has $\mathcal L^*=-3.1438$, and after (b), with the variances in (a), $\mathcal L_{\mathrm{current}}=-3.3313$ (rounded to four decimal places). Compute the approximation gap and the remaining optimization gap, in nats.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Approximation gap | Select the correct pair of levels and calculate their difference, one point each. | 2 |
| Optimization gap | Select the correct pair of levels and calculate their difference, one point each. | 2 |

### Board notes and calculations

## Q3(a) · Importance Sampling and Effective Sample Size · 8 points

For $f(z)=z$, the normalized target distribution $p$ and proposal distribution $q$ are:
| $z$ | $0$ | $1$ | $2$ |
|---|---:|---:|---:|
| $p(z)$ | $\tfrac{1}{4}$ | $\tfrac{1}{4}$ | $\tfrac{1}{2}$ |
| $q(z)$ | $\tfrac{1}{2}$ | $\tfrac{1}{4}$ | $\tfrac{1}{4}$ |

Four independent draws from $q$ are $(0,1,1,2)$; count repeated values separately. Use
$$
w_i=\frac{p(z^{(i)})}{q(z^{(i)})},\quad
W_i=\frac{w_i}{\sum_{j=1}^{4}w_j},\quad
N_{\mathrm{eff}}^{\mathrm{weight}}=\frac1{\sum_{i=1}^{4}W_i^2}.
$$

Compute the raw weights $w_i$, their sum, and normalized weights $W_i$, in draw order.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Four raw weights | One point per draw: half for the ratio substitution and half for arithmetic; repeated draws retain separate slots. | 4 |
| Weight sum | Add all four contributions; method credit retained for a new arithmetic slip. | 1 |
| Normalization denominator | Use the sum of all raw draw weights. | 1 |
| Four normalized weights | Half a point per draw in the original order; consistent follow-through accepted. | 2 |

### Board notes and calculations

## Q3(b) · Importance Sampling and Effective Sample Size · 8 points

Use the same setting as Q3(a).

Compute the exact target expectation $\mathbb E_p[z]$, the direct IS estimate, and the self-normalized IS (SNIS) estimate from these four draws. Show the denominator of each estimate.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Exact target expectation | State-space setup and result, one point each. | 2 |
| Direct IS | Weighted numerator, requested denominator, and arithmetic, one point each. | 3 |
| SNIS | Weighted numerator, correct normalization, and arithmetic, one point each; equivalent normalized-weight form accepted. | 3 |

### Board notes and calculations

## Q3(c) · Importance Sampling and Effective Sample Size · 4 points

Use the same setting as Q3(a).

Compute the weight ESS using the normalized weights from (a). Give both an exact fraction and a decimal value.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Squared-weight sum | Correct four-draw setup and calculation, one point each; equivalent raw-weight ESS formula accepted. | 2 |
| ESS | Reciprocal operation and result, one point each; the result mark requires both fraction and decimal, with half for one correct form. | 2 |

### Board notes and calculations

## Q4(a) · Predictive Loss and Paired Comparisons · 6 points

Three responses are conditionally independent, with $y=(0,2,4)$, predicted means $\mu_A=(1,1,3)$ and $\mu_B=(0,1,4)$. Both models use the same known variance $\sigma^2=2$:

$$
p(y_i\mid\mu_i,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left[-\frac{(y_i-\mu_i)^2}{2\sigma^2}\right].
$$

For (c), use these separate experiment records. Match conditions by run ID, not row position. Lower error is better.

| A: run ID | A: error | B: run ID | B: error |
|---|---:|---|---:|
| 1 | 0.4 | 3 | 0.2 |
| 2 | 0.3 | 1 | 0.3 |
| 3 | 0.5 | 2 | 0.4 |

Derive the summed Gaussian NLL in terms of $\mathrm{SSE}=\sum_{i=1}^{3}(y_i-\mu_i)^2$, retaining the normalizing constant. Use natural logarithms.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Independent joint density | Show the product over all observations; partial credit for incomplete product or independence stated alone. | 2 |
| Negative log and quadratic term | Credit the log transformation and the supplied-variance coefficient separately. | 2 |
| Normalizing constant | Show its general structure and substitute the supplied sample size and variance. | 2 |

### Board notes and calculations

## Q4(b) · Predictive Loss and Paired Comparisons · 8 points

Use the same setting as Q4(a).

Calculate SSE and MSE for each model, and $\mathrm{NLL}_A-\mathrm{NLL}_B$. Show residual calculations.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Model A residuals and SSE | Credit the residual/squared-residual calculation and the sum separately; consistent opposite residual sign accepted. | 2 |
| Model A MSE | Divide the SSE by the number of observations; consistent follow-through accepted. | 1 |
| Model B residuals and SSE | Credit the residual/squared-residual calculation and the sum separately; consistent opposite residual sign accepted. | 2 |
| Model B MSE | Divide the SSE by the number of observations; consistent follow-through accepted. | 1 |
| Signed NLL difference | Correct common-term cancellation and calculation in the requested A-minus-B direction. | 2 |

### Board notes and calculations

## Q4(c) · Predictive Loss and Paired Comparisons · 6 points

Use the same setting as Q4(a).

Match by run ID. Calculate $d_i=e_{A,i}-e_{B,i}$ for runs $1,2,3$, their mean, and the number of runs B wins. A tie is not a win.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Matched difference, run 1 | Pair the records with the same run ID and retain the specified subtraction direction. | 1 |
| Matched difference, run 2 | Pair the records with the same run ID and retain the specified subtraction direction. | 1 |
| Matched difference, run 3 | Pair the records with the same run ID and retain the specified subtraction direction. | 1 |
| Paired mean | Correct aggregation and calculation, one point each; an equivalent difference of means is accepted. | 2 |
| B win count | Count differences with the correct sign; ties are not wins. | 1 |

### Board notes and calculations

## Q5(a) · Bayesian Prediction in Weight and Function Space · 8 points

Let $f(x)=xw$, with $w\sim\mathcal N(0,\tfrac12)$ and $y_i=x_iw+\epsilon_i$, where the training noises $\epsilon_i\sim\mathcal N(0,1)$ are independent of each other and of $w$. The second argument of $\mathcal N$ is the **variance**. Observe $D=\{(1,1),(2,3)\}$.

Find the posterior mean and variance of $w\mid D$. Show the prior–likelihood calculation, using either square completion or a precision update.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Prior and both likelihoods | Credit the prior-variance contribution and both data contributions separately. | 2 |
| Quadratic or precision calculation | Correct quadratic and linear coefficients, or equivalent precision and information contributions. | 2 |
| Posterior mean | Identify the center or equivalent precision relation and show the resulting calculation. | 2 |
| Posterior variance | Use the precision-to-variance relation and calculate the variance; distinguish standard deviation. | 2 |

### Board notes and calculations

## Q5(b) · Bayesian Prediction in Weight and Function Space · 8 points

Use the same setting as Q5(a).

For a **new response** at $x_*=2$, let $f_*=2w$ and $y_*=f_*+\epsilon_*$, where $\epsilon_*\sim\mathcal N(0,1)$ is independent of $w$ and the training noises. Calculate $E[f_*\mid D]$, $\operatorname{Var}(f_*\mid D)$ and $\operatorname{Var}(y_*\mid D)$. Show the variance calculations.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Latent-function mean | Correct expectation transformation and calculation, one point each. | 2 |
| Latent-function variance | Credit squared scaling, use of the posterior variance, and calculation separately. | 3 |
| New-response variance | Credit independent-noise addition, correct noise variance, and calculation separately. | 3 |

### Board notes and calculations

## Q5(c) · Bayesian Prediction in Weight and Function Space · 4 points

Use the same setting as Q5(a).

**Before observing $D$**, derive the prior kernel $k(x,x')=\operatorname{Cov}(f(x),f(x'))$ and calculate $k(-1,2)$. Show how the weight prior determines this covariance.

### Original marking criteria

| Assessment item | Credit conditions | Points |
|---|---|---:|
| Prior kernel derivation | Use the shared-weight covariance and its prior variance. | 2 |
| Prior covariance for the input pair | Show substitution of the requested input pair and the resulting calculation. | 2 |

### Board notes and calculations

## After-class review notes

Which step will you revisit? Record the relevant question label and your own explanation. This sheet does not introduce a new submission task.

## References

Original instructor-provided midterm, October 20, 2026. Previous class materials: [Course home](https://github.com/lunalab-ai/advML).
