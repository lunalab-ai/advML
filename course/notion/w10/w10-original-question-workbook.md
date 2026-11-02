# W10 · Original Q3–Q5 Question Workbook

Advanced Machine Learning · November3,2026. Original wording, assumptions and original points are retained. The points are archival labels; this is formative class practice, with no new graded assignment. Write your own derivations in the blank grids.

## Q3 · Importance Sampling and Effective Sample Size

20 points

For $f(z)=z$, the normalized target distribution $p$ and proposal distribution $q$ are:

| $z$ | $0$ | $1$ | $2$ |
| --- | --- | --- | --- |
| $p(z)$ | $\tfrac{1}{4}$ | $\tfrac{1}{4}$ | $\tfrac{1}{2}$ |
| $q(z)$ | $\tfrac{1}{2}$ | $\tfrac{1}{4}$ | $\tfrac{1}{4}$ |

Four independent draws from $q$ are $(0,1,1,2)$; count repeated values separately. Use
$$
w_i=\frac{p(z^{(i)})}{q(z^{(i)})},\quad
W_i=\frac{w_i}{\sum_{j=1}^{4}w_j},\quad
N_{\mathrm{eff}}^{\mathrm{weight}}=\frac1{\sum_{i=1}^{4}W_i^2}.
$$

### Q3(a)

Compute the raw weights $w_i$, their sum, and normalized weights $W_i$, in draw order. **[8 points]**

| Q3(a) · Your working and explanation |
|---|
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |

### Q3(b)

Compute the exact target expectation $\mathbb E_p[z]$, the direct IS estimate, and the self-normalized IS (SNIS) estimate from these four draws. Show the denominator of each estimate. **[8 points]**

| Q3(b) · Your working and explanation |
|---|
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |

### Q3(c)

Compute the weight ESS using the normalized weights from (a). Give both an exact fraction and a decimal value. **[4 points]**

Show all working and final answers for (a)-(c).

| Q3(c) · Your working and explanation |
|---|
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |

## Q4 · Predictive Loss and Paired Comparisons

20 points

Three responses are conditionally independent, with $y=(0,2,4)$, predicted means $\mu_A=(1,1,3)$ and $\mu_B=(0,1,4)$. Both models use the same known variance $\sigma^2=2$:

$$
p(y_i\mid\mu_i,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left[-\frac{(y_i-\mu_i)^2}{2\sigma^2}\right].
$$

For (c), use these separate experiment records. Match conditions by run ID, not row position. Lower error is better.

| A: run ID | A: error | B: run ID | B: error |
| --- | --- | --- | --- |
| 1 | 0.4 | 3 | 0.2 |
| 2 | 0.3 | 1 | 0.3 |
| 3 | 0.5 | 2 | 0.4 |

### Q4(a)

Derive the summed Gaussian NLL in terms of $\mathrm{SSE}=\sum_{i=1}^{3}(y_i-\mu_i)^2$, retaining the normalizing constant. Use natural logarithms. **[6 points]**

| Q4(a) · Your working and explanation |
|---|
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |

### Q4(b)

Calculate SSE and MSE for each model, and $\mathrm{NLL}_A-\mathrm{NLL}_B$. Show residual calculations. **[8 points]**

| Q4(b) · Your working and explanation |
|---|
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |

### Q4(c)

Match by run ID. Calculate $d_i=e_{A,i}-e_{B,i}$ for runs $1,2,3$, their mean, and the number of runs B wins. A tie is not a win. **[6 points]**

Show all working and final answers for (a)-(c).

| Q4(c) · Your working and explanation |
|---|
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |

## Q5 · Bayesian Prediction in Weight and Function Space

20 points

Let $f(x)=xw$, with $w\sim\mathcal N(0,\tfrac12)$ and $y_i=x_iw+\epsilon_i$, where the training noises $\epsilon_i\sim\mathcal N(0,1)$ are independent of each other and of $w$. The second argument of $\mathcal N$ is the **variance**. Observe $D=\{(1,1),(2,3)\}$.

### Q5(a)

Find the posterior mean and variance of $w\mid D$. Show the prior–likelihood calculation, using either square completion or a precision update. **[8 points]**

| Q5(a) · Your working and explanation |
|---|
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |

### Q5(b)

For a **new response** at $x_*=2$, let $f_*=2w$ and $y_*=f_*+\epsilon_*$, where $\epsilon_*\sim\mathcal N(0,1)$ is independent of $w$ and the training noises. Calculate $E[f_*\mid D]$, $\operatorname{Var}(f_*\mid D)$ and $\operatorname{Var}(y_*\mid D)$. Show the variance calculations. **[8 points]**

| Q5(b) · Your working and explanation |
|---|
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |

### Q5(c)

**Before observing $D$**, derive the prior kernel $k(x,x')=\operatorname{Cov}(f(x),f(x'))$ and calculate $k(-1,2)$. Show how the weight prior determines this covariance. **[4 points]**

Show all working and final answers for (a)-(c).

| Q5(c) · Your working and explanation |
|---|
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
|   |
