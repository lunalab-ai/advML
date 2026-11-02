# W10 · Code Guide

CPU teaching cases; no dataset upload or hidden training. Inspect the exact definitions before editing the workbench.

## Installation and conventions

The notebook pins five dependencies and verifies five module hashes. Existing matching local/cache files are reused; public files are fetched from the reviewed fixed tag. A mismatch fails explicitly. No unpinned Git install is used. Gaussian loss takes variance; scalar fitting takes noise SD. Shapes are checked before subtraction or fitting.

## discrete_importance

[Exact definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w10/src/advml_review.py#L30)

Summarize fixed draws on states 0,...,K-1, retaining every draw index.

p and q are normalized probability masses of equal length. q must be
strictly positive on the supplied support. p may contain zeros. Draws are
supplied, not sampled by this function. Direct IS divides by draw count;
SNIS divides by raw-weight mass. SNIS is undefined if all drawn weights
vanish, in which case the combined summary raises a clear ValueError.
Returns a per-draw DataFrame, raw/normalized weights, exact expectation of
state value, both estimates, and WEIGHT ESS (not MCMC ESS). No mutation.
Example: discrete_importance([.3,.7],[.5,.5],[0,1])['exact'] == .7.

## gaussian_loss

[Exact definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w10/src/advml_review.py#L70)

Return residual table, SSE, MSE and SUMMED Gaussian NLL, constants kept.

y and mean must have identical (n,) shapes. A scalar positive variance is
required; a standard deviation must be squared before calling. Unlike
NumPy broadcasting, an (n,1) array is rejected. Example:
gaussian_loss([0,1],[0,0],1)['sse'] == 1. Densities can exceed one, so an NLL
need not be nonnegative. There is no model fitting or array mutation.

## paired_runs

[Exact definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w10/src/advml_review.py#L97)

Join separate lower-is-better error records by unique string run_id.

Both frames require run_id,error columns, exactly the same set of IDs,
and finite errors. Extra columns are ignored. Input order of a is retained;
order of b is irrelevant. Returns per-ID d=A-B, mean d, positive/negative
counts and exact ties. A positive d is a B win. No significance claim or
silent missing-ID drop; errors may be negative for valid losses such as NLL.
Example: paired_runs(pd.DataFrame({'run_id':['r1'],'error':[.4]}),
pd.DataFrame({'run_id':['r1'],'error':[.2]}))['b_wins'] == 1.

## scalar_posterior

[Exact definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w10/src/advml_review.py#L131)

Fit y_i=x_i*w+epsilon_i with w~N(0,tau^2), independent N(0,noise_sd^2).

One coefficient, no intercept, known homoscedastic noise. Returns posterior
mean/variance and coefficients a,b,c of a*w^2-2*b*w+c; square_constant is
c-b^2/a. Cross-checks against existing BayesianLinear using feature shape
(n,1). Example: scalar_posterior([1],[1])['variance'] == .5.

## scalar_prediction

[Exact definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w10/src/advml_review.py#L162)

Predict latent f=x_star*w and a NEW response with independent new noise.

Multipliers square in a variance. New noise variance is added exactly once;
a training input reused as x_star still denotes a new response, not the
already observed y. Returns exact means and the two distinct variances.

## shared_weight_covariance

[Exact definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w10/src/advml_review.py#L181)

Return Var(w)*outer(x,x) for function values sharing the SAME weight.

Use prior weight variance for a prior kernel or posterior weight variance
for conditioned function covariance. No observation noise is included.
Negative cross-covariances are compatible with positive semidefiniteness.
Example: shared_weight_covariance([-1,2],1)[0,1] == -2.

## importance_view

[Exact definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w10/src/advml_review_app.py#L40)

Compute fixed-draw weights and all estimates; optionally export this table.

## loss_view

[Exact definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w10/src/advml_review_app.py#L51)

Compare two means on the SAME observations and common noise variance.

An unfinished diagnostic returning None is explicitly labeled; it is not
replaced by a completed answer. Other core results remain independent.

## paired_view

[Exact definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w10/src/advml_review_app.py#L90)

Align independent CSV records by ID and report every signed difference.

## bayes_view

[Exact definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w10/src/advml_review_app.py#L102)

Exact one-slope posterior, exact prediction and separately labeled iid MC.

Monte Carlo draw count changes numerical estimation, not the exact posterior
or its predictive variance. Each function draw shares one weight over x.

## build_app

[Exact definition](https://github.com/lunalab-ai/advML/blob/2026-fall-w10/src/advml_review_app.py#L152)

Return the three-tab workbench; launch separately in the notebook.

The optional diagnostic receives y,mean_A and returns a scalar or None.
CSV buttons recompute and update visible results from current controls.
This function builds UI only; it starts no server or public sharing.

## Six core tasks

E1 predicts roles and weights; E2 explains two denominators; E3 repairs a shape; E4 checks keyed differences; E5 distinguishes two predictive variances; E6 supplies a diagnostic callable. Empty response fields are independent of demonstrations. The diagnostic returning None is labeled unfinished.

## App and file effects

build_app constructs Blocks without starting a server. Launch explicitly in the notebook. Only CSV export writes a new temporary file; Gradio may cache plots. Button callbacks recompute from current controls, update visible results and export the same table. No personal files, credentials, paid GPU or external dataset are required.

## Reproducibility

Data are synthetic teaching inputs. Random function draws use a local Generator and seed. Exact posterior/prediction values are separate from Monte Carlo mean estimates and their iid standard errors. Shared app URLs expire when the runtime stops.
