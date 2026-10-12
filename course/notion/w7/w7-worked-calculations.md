# W7 Worked Calculations — Follow Every Term

**October 13, 2026 · Companion to From Predictive Models to Bayesian Functions**

These are explained teaching examples. The separate lab exercises remain opportunities to reason for yourself. For each calculation, name what is observed and what is unknown, predict the direction of the answer, calculate, and then interpret the result in words.

## A notation bridge

$\mathcal N(\mu,v)$ uses **variance** $v$ as its second argument in our equations. A software call may instead request standard deviation: check its contract. Our code's `noise` argument is a standard deviation, so the covariance diagonal receives `noise**2`. A transpose turns a column into a row; $\phi^\top S\phi$ is a scalar when $\phi$ is a column vector and $S$ is a square covariance matrix. The identity matrix $I$ has ones on the diagonal and zeros elsewhere.

Precision means inverse variance in one dimension. Larger precision means a narrower Gaussian distribution. A likelihood is evaluated at fixed observed data while its parameters vary. For a continuous observation, a density is not the probability of observing exactly that real number; densities can exceed one, and their negative logarithms can be negative.

## 1. Gaussian likelihood becomes squared-error loss

**Given:** two independent responses $y_1=1$, $y_2=2$. A candidate model predicts $\mu_1=\mu_2=1.5$. The known noise SD is 1, so variance is also 1.

**Step 1 — Residuals.** Subtract the candidate mean from each observation:

$$
e_1=1-1.5=-0.5,\qquad e_2=2-1.5=0.5.
$$

Their signs cancel if added, which would conceal error. Squaring gives $e_1^2=e_2^2=0.25$. Thus SSE is $0.5$, while MSE is $0.5/2=0.25$.

**Step 2 — Densities.** For each case,

$$
p(y_i\mid\mu_i,1)=\frac{1}{\sqrt{2\pi}}\exp(-0.25/2)\approx0.352065.
$$

Conditional independence gives joint density approximately $0.352065^2=0.123950$.

**Step 3 — Negative logarithm.** The summed NLL is

$$
-\log p(y_1,y_2\mid\mu)=\log(2\pi)+\frac{0.5}{2}
\approx1.837877+0.25=2.087877.
$$

Mean NLL divides that result by two, giving about 1.043939. When someone reports a “loss,” ask whether it is summed or averaged and whether constants are retained.

**Step 4 — Interpretation.** Holding the variance fixed, the normalization term does not depend on the mean parameters. Minimizing NLL therefore selects the same mean parameters as minimizing SSE or MSE. The numerical loss values are not identical. If variance changes, both the normalization term and the scaled residual term change.

**A useful contrast.** For the same residuals but variance $v$, NLL is $\log(2\pi v)+0.25/v$. A smaller variance increases the penalty for a fixed residual. A larger variance lowers that residual penalty but increases the normalization penalty. This is why probabilistic training must not discard the variance-dependent term.

## 2. Bernoulli likelihood becomes binary cross entropy

**Given:** labels $(1,0)$ and probabilities of label 1 equal to $(0.8,0.25)$.

**Step 1 — Select the probability of what actually occurred.** The first observation contributes 0.8 because its label is 1. The second contributes $1-0.25=0.75$ because its label is 0. Multiplying $0.8\times0.25$ would use the wrong event for the second case.

**Step 2 — Multiply and take logs.** The joint probability is $0.8\times0.75=0.6$. Thus summed NLL is

$$
-\log(0.6)=-\log(0.8)-\log(0.75)
\approx0.223144+0.287682=0.510826.
$$

The mean cross entropy is about 0.255413. The natural logarithm is used throughout.

**Step 3 — Connect to real-valued scores.** The scores are logits: $z_1=\log(0.8/0.2)=\log4$ and $z_2=\log(0.25/0.75)=-\log3$. Applying the sigmoid recovers the two probabilities. A stable code expression for one case is $\log(1+e^z)-yz$; `logaddexp(0,z)` avoids overflow for extreme scores.

**Step 4 — Interpret.** The probabilities refer to two different conditional distributions, so $0.8+0.25$ need not equal one. The Bernoulli label is 0 or 1; the predicted mean can be any probability between them. A threshold decision loses some of this distributional information.

**Poisson connection.** For observed count $y$ and log-rate $\eta$, taking the negative log of $e^{-\lambda}\lambda^y/y!$, with $\lambda=e^\eta$, gives $e^\eta-y\eta+\log(y!)$. The factorial term is constant for fitting a fixed observation but is part of its full NLL.

## 3. Average predictive distributions and decompose their variance

**Given:** three possible component models have weights $(0.2,0.3,0.5)$, means $(0,1,2)$ and conditional variances all equal to 0.25. These are illustrative normalized posterior weights, not three fresh observations.

**Step 1 — Predictive mean.** Weight the means:

$$
\bar m=0.2(0)+0.3(1)+0.5(2)=1.3.
$$

**Step 2 — Within-component variance.** Average the conditional variances:

$$
v_{\mathrm{within}}=0.2(0.25)+0.3(0.25)+0.5(0.25)=0.25.
$$

**Step 3 — Between-component variance.** Compute each squared distance from the overall mean, then weight it:

$$
v_{\mathrm{between}}=0.2(0-1.3)^2+0.3(1-1.3)^2+0.5(2-1.3)^2
=0.338+0.027+0.245=0.61.
$$

**Step 4 — Total variance.** Add the two components: $v_y=0.25+0.61=0.86$. The standard deviation is $\sqrt{0.86}\approx0.927362$, not $\sqrt{0.25}+\sqrt{0.61}$.

**Step 5 — Check through second moments.** Each component has $E[y^2\mid j]=v_j+m_j^2$. Therefore the mixture second moment is $0.25+[0.2(0)+0.3(1)+0.5(4)]=2.55$. Subtracting the squared mean gives $2.55-1.3^2=0.86$, independently confirming the answer.

**Why the identity works.** Write $y-\bar m=(y-m_\theta)+(m_\theta-\bar m)$ and expand the square. The conditional expectation of $y-m_\theta$ is zero, so the cross term vanishes. The remaining two terms are the within and between variances.

**Interpretation.** More computation can estimate these quantities more precisely, but it does not automatically make the quantities themselves smaller. Additional informative observations can change the posterior weights. The mixture need not be Gaussian; use its actual distribution or quantiles for a claimed 95% interval.

## 4. Bayesian slope posterior and a new response

**Given:** $w\sim\mathcal N(0,1)$, observations $(1,1)$ and $(2,2)$, model $y_i=x_iw+\epsilon_i$, independent $\epsilon_i\sim\mathcal N(0,1)$.

**Step 1 — Form the unnormalized posterior.** Combine the prior and both likelihood contributions:

$$
p(w\mid D)\propto\exp\left[-\frac12\{w^2+(1-w)^2+(2-2w)^2\}\right].
$$

**Step 2 — Expand.** The expression in braces is

$$
w^2+(1-2w+w^2)+(4-8w+4w^2)=6w^2-10w+5.
$$

**Step 3 — Complete the square.**

$$
6w^2-10w+5=6\left(w-\frac56\right)^2+\frac56.
$$

The last constant affects normalization but not the location or spread. Comparing the exponential with a Gaussian kernel gives posterior mean $5/6$ and variance $1/6$.

**Step 4 — Push this distribution through the function.** At $x_*=3$, $f_*=3w$. Multiplying a random variable by 3 multiplies its mean by 3 and its variance by 9. Hence

$$
f_*\mid D\sim\mathcal N(2.5,1.5).
$$

**Step 5 — Add a new independent noise draw.** Since $y_*=f_*+\epsilon_*$ and the new noise has variance 1,

$$
y_*\mid D\sim\mathcal N(2.5,2.5).
$$

The pointwise 95% latent interval is approximately $2.5\pm2.4005$, while the response interval is approximately $2.5\pm3.0990$. Both are centered at the same number. They answer different questions.

**General matrix derivation.** For $w\sim\mathcal N(m_0,S_0)$ and $y\mid w\sim\mathcal N(\Phi w,\sigma^2I)$, the negative log posterior, ignoring terms independent of $w$, is

$$
\frac12w^\top Aw-w^\top b,
\quad A=S_0^{-1}+\sigma^{-2}\Phi^\top\Phi,
\quad b=S_0^{-1}m_0+\sigma^{-2}\Phi^\top y.
$$

Completing the square gives $\tfrac12(w-A^{-1}b)^\top A(w-A^{-1}b)$ plus a constant. Thus $S=A^{-1}$ and $m=A^{-1}b$. Dimensions: $A,S$ are $d\times d$, $b,m$ are $d$-vectors, and $\Phi$ is $n\times d$. Implement a solve, even when the compact equation displays an inverse.

**MAP and ridge scaling.** With $m_0=0$ and $S_0=\tau^2I$, minimizing $\|y-\Phi w\|^2/(2\sigma^2)+\|w\|^2/(2\tau^2)$ is equivalent to minimizing $\|y-\Phi w\|^2+(\sigma^2/\tau^2)\|w\|^2$. Keep the factors of two and sum-versus-mean convention consistent. A MAP weight vector alone is not the full posterior predictive.

## 5. A two-observation Gaussian process calculation

**Given:** training inputs $X=(0,1)$, responses $y=(0,1)^\top$, test input $x_*=0.5$, zero prior mean, RBF amplitude and length scale both 1, and noise **SD** 0.5. Therefore noise variance is 0.25.

**Step 1 — Build covariances.** Define $\rho=e^{-1/2}\approx0.606531$ and $r=e^{-1/8}\approx0.882497$. Then

$$
K=\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix},\quad
C=K+0.25I=\begin{pmatrix}1.25&\rho\\\rho&1.25\end{pmatrix},\quad
k_*=\begin{pmatrix}r\\r\end{pmatrix},\quad k_{**}=1.
$$

The diagonal of $C$ includes observation noise because the training responses are noisy. The prior test function variance $k_{**}$ does not include new observation noise.

**Step 2 — Solve the mean system.** Write $a=1.25$. For explanation only, the two-by-two inverse is

$$
C^{-1}=\frac{1}{a^2-\rho^2}\begin{pmatrix}a&-\rho\\-\rho&a\end{pmatrix}.
$$

Multiplying by $y$ yields coefficients $(-\rho,a)^\top/(a^2-\rho^2)$. One is negative. They are linear-system coefficients, not posterior probabilities. Multiplying by $k_*^\top$ gives

$$
m_*=\frac{r(a-\rho)}{(a-\rho)(a+\rho)}=\frac{r}{a+\rho}\approx0.475347.
$$

The result is near, but not exactly, the arithmetic average 0.5. The zero-mean prior and nonzero noise influence it.

**Step 3 — Reduce the latent variance by conditioning.** Because the two entries of $k_*$ match, $C^{-1}k_*=(r/(a+\rho))(1,1)^\top$. Consequently,

$$
v_f=k_{**}-k_*^\top C^{-1}k_*=1-\frac{2r^2}{a+\rho}\approx0.161015.
$$

**Step 4 — Predict an observation.** Add variance 0.25, obtaining $v_y\approx0.411015$. The latent SD is about 0.401266 and response SD about 0.641104. A 95% pointwise interval uses the relevant SD, not the variance itself as its radius.

**Step 5 — Relate the implementation.** The program factors $C=LL^\top$, solves $Lz=y$, then $L^\top\alpha=z$, and computes $k_*^\top\alpha$. It similarly solves for the covariance reduction. This is the same mathematical calculation with a more stable numerical route.

## A bridge to the notebook equivalence experiment

Let $\phi(x)=(1,x)^\top$ and $S_0=0.7I$. Then the induced kernel is $k(x,x')=0.7(1+xx')$. With $x=-1$ and $x'=2$, the covariance is $-0.7$. This does not invalidate the kernel. For any feature matrix $\Phi$ and vector $a$, $a^\top(\Phi S_0\Phi^\top)a=(\Phi^\top a)^\top S_0(\Phi^\top a)\geq0$.

Lab B compares the two posterior calculations with exactly this matched structure. Changing the features or prior in only one calculation breaks the comparison. The residual numerical discrepancy should be at floating-point scale, not a substantive model difference.

## Before moving on

For each example, cover the answer and explain what must stay fixed. Reconstruct the intermediate values before using the helper function. If an answer looks implausible, check the event selected by a label, SD versus variance, sum versus average, and whether you are predicting a function or a noisy observation.

Model identities follow Murphy, *Advanced Topics*, Sections 14.2, 15.1–15.2, 17.3.10 and 18.3. These numerical examples and their step-by-step derivations are original additions.
