# W4 shared inference API

All logarithms are natural; two-dimensional arrays follow the model in the lecture.

## [GaussianModel](https://github.com/lunalab-ai/advML/blob/2026-fall-w3/src/advml_vi.py#L14)

Exact reference for one observed vector.

x and mean have shape (2,); noise, precision and covariance have shape
(2,2); log_evidence is a scalar log density, not a probability.
Construct with gaussian_model; treat stored arrays as read-only.

## [gaussian_model](https://github.com/lunalab-ai/advML/blob/2026-fall-w3/src/advml_vi.py#L36)

Return an exact posterior/evidence reference, without fitting or sampling.

rho: observation-noise correlation, strictly between -1 and 1; it is NOT
posterior correlation. x: observed length-2 vector, default (1,-1).
Returned arrays are new. Raises ValueError for invalid dimensions/numbers.
Example: gaussian_model(.8).mean is approximately [.833333,-.833333].

## [mean_field_optimum](https://github.com/lunalab-ai/advML/blob/2026-fall-w3/src/advml_vi.py#L63)

Return the reverse-KL-optimal diagonal Gaussian (mean, variance).

Both outputs have shape (2,). The mean is the exact posterior mean;
variance[j] = 1/precision[j,j], generally NOT covariance[j,j].
Inputs are unchanged; no iteration or download occurs.

## [elbo](https://github.com/lunalab-ai/advML/blob/2026-fall-w3/src/advml_vi.py#L73)

Compute E_q[log p(z)+log p(x|z)] + H(q), directly.

mean and variance are length-2 vectors for q=N(mean,diag(variance)).
Output is a scalar in nats; variance means sigma squared, not sigma.
Includes every normalization constant; does NOT subtract KL from the
evidence oracle internally. Inputs are not mutated.

## [diagonal_kl](https://github.com/lunalab-ai/advML/blob/2026-fall-w3/src/advml_vi.py#L91)

Return KL(q || exact posterior) in nats using the Gaussian KL formula.

mean/variance: length-2 vectors, positive variance. This independently
checks log_evidence - elbo; it is not a sample estimate or a symmetric
distance. Inputs unchanged. A tiny negative roundoff near zero is possible.

## [elbo_gradient](https://github.com/lunalab-ai/advML/blob/2026-fall-w3/src/advml_vi.py#L106)

Analytic ELBO gradients with respect to mean and log standard deviation.

Inputs are length-2 vectors. variance=exp(2*log_std). Returns two length-2
arrays (dL/dmean, dL/dlog_std), for gradient ASCENT. No state is changed.
Using log standard deviation keeps variances positive.

## [reparameterized_gradient](https://github.com/lunalab-ai/advML/blob/2026-fall-w3/src/advml_vi.py#L119)

Estimate the same gradients by z=mean+exp(log_std)*epsilon.

samples: positive integer; seed initializes a local NumPy generator.
Analytic Gaussian entropy supplies the +1 in the log-std gradient.
Returns two (2,) sample-average gradients. No global random state,
downloads or fitting. Estimates fluctuate; no per-step ascent guarantee.

## [run_vi](https://github.com/lunalab-ai/advML/blob/2026-fall-w3/src/advml_vi.py#L139)

Run sequential Gaussian CAVI or analytic ELBO gradient ascent.

method is 'cavi' or 'gradient'; steps is a nonnegative integer. CAVI
updates coordinate 0 then 1 using the newest values; a recorded step is
one complete sweep. Gradient updates mean and log_std jointly with an
Armijo backtracking line search from learning_rate (>0).
Returns means/variances of shape (steps+1,2), elbos (steps+1,),
step_sizes (steps,); NaN for CAVI, and method. Row 0 is initialization.
Inputs/model are not mutated. Raises on invalid inputs or line-search
failure; never replaces a failed calculation with a fabricated result.

## [diagnostics](https://github.com/lunalab-ai/advML/blob/2026-fall-w3/src/advml_vi.py#L197)

Return scalar/JSON-ready diagnostics for one diagonal approximation.

Separates approximation gap at the optimal diagonal q from optimization
gap above that optimum. Gaps are KL quantities in nats. Only a numerical
optimization gap within 1e-10 of zero is rounded to zero for display.

## [trace_figure](https://github.com/lunalab-ai/advML/blob/2026-fall-w3/src/advml_vi.py#L219)

Return a Matplotlib Figure: covariance geometry and ELBO trace.

Contours have unit Mahalanobis radius (not 68% joint credible regions).
trace is a run_vi result. Creates no files or server; caller may savefig.

## [vi_view](https://github.com/lunalab-ai/advML/blob/2026-fall-w3/src/advml_vi.py#L252)

App callback: settings -> (Figure, JSON-ready diagnostics).

rho controls observation noise; method is cavi/gradient; steps counts full
sweeps/steps. Each call constructs a fresh model and trace; no retained fit.
learning_rate applies only to gradient ascent. Default observed x=(1,-1).

## [build_vi_app](https://github.com/lunalab-ai/advML/blob/2026-fall-w3/src/advml_vi.py#L269)

Return an unlaunched Gradio Blocks app using vi_view.

Requires Gradio 6.27.0. No server/download starts here. Call launch with
share=True in Colab for a temporary link tied to the running runtime.
Students extend the displayed diagnostics in their own exercise callback.
