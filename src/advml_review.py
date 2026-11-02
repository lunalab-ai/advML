"""Auditable small-model inference and prediction workbench.

Original course helpers. No fitting to hidden data, network access, global RNG
mutation, or exercise-specific answers. Prior mean is zero; the scalar model
has one slope and no intercept. All computations use explicit supplied arrays.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd
from advml_sampling import normalize_log_weights, weight_ess
from advml_predictive import gaussian_nll, BayesianLinear


def _vector(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a nonempty finite one-dimensional array.')
    return result.copy()


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive.')
    return value


def discrete_importance(p: Sequence[float], q: Sequence[float], draws: Sequence[int]) -> dict:
    """Summarize fixed draws on states 0,...,K-1, retaining every draw index.

    p and q are normalized probability masses of equal length. q must be
    strictly positive on the supplied support. p may contain zeros. Draws are
    supplied, not sampled by this function. Direct IS divides by draw count;
    SNIS divides by raw-weight mass. SNIS is undefined if all drawn weights
    vanish, in which case the combined summary raises a clear ValueError.
    Returns a per-draw DataFrame, raw/normalized weights, exact expectation of
    state value, both estimates, and WEIGHT ESS (not MCMC ESS). No mutation.
    Example: discrete_importance([.3,.7],[.5,.5],[0,1])['exact'] == .7.
    """
    p, q = _vector(p, 'Target p'), _vector(q, 'Proposal q')
    if p.shape != q.shape or np.any(p < 0) or np.any(q <= 0):
        raise ValueError('Use equal-length p >= 0 and q > 0 on the supplied support.')
    if not all(np.isclose(v.sum(), 1, atol=1e-10, rtol=0) for v in (p, q)):
        raise ValueError('Both p and q must sum to one; use normalized target masses.')
    z = _vector(draws, 'Draws')
    if np.any(z != np.floor(z)) or np.any(z < 0) or np.any(z >= len(p)):
        raise ValueError('Each draw must be an integer state index on the support.')
    z = z.astype(int)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        raw = p[z] / q[z]
        log_weights = np.log(p[z]) - np.log(q[z])
        numerator = float(np.dot(raw, z))
        mass = float(raw.sum())
    if not np.isfinite(raw).all() or not np.isfinite([numerator, mass]).all():
        raise ValueError('Importance ratios overflow; choose a better supported proposal.')
    if mass <= 0:
        raise ValueError('All drawn weights are zero; SNIS is undefined for these draws.')
    normalized = normalize_log_weights(log_weights)
    table = pd.DataFrame({'draw_id': np.arange(1, len(z)+1), 'state': z,
                          'p': p[z], 'q': q[z], 'raw_weight': raw,
                          'normalized_weight': normalized, 'weighted_state': raw*z})
    return {'table': table, 'raw_weights': raw, 'normalized_weights': normalized,
            'draw_count': len(z), 'numerator': numerator, 'weight_sum': mass,
            'exact': float(p @ np.arange(len(p))), 'direct_is': numerator/len(z),
            'snis': float(normalized @ z), 'weight_ess': weight_ess(normalized)}


def gaussian_loss(y: Sequence[float], mean: Sequence[float], variance: float) -> dict:
    """Return residual table, SSE, MSE and SUMMED Gaussian NLL, constants kept.

    y and mean must have identical (n,) shapes. A scalar positive variance is
    required; a standard deviation must be squared before calling. Unlike
    NumPy broadcasting, an (n,1) array is rejected. Example:
    gaussian_loss([0,1],[0,0],1)['sse'] == 1. Densities can exceed one, so an NLL
    need not be nonnegative. There is no model fitting or array mutation.
    """
    y, mean = _vector(y, 'Observed y'), _vector(mean, 'Prediction mean')
    variance = _positive(variance, 'Variance')
    if y.shape != mean.shape:
        raise ValueError('Observed y and prediction mean must have the same (n,) shape.')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        residual = y-mean
        squared = residual**2
        sse = float(squared.sum())
        nll = float(gaussian_nll(y, mean, variance))
    if not np.isfinite([sse, nll]).all():
        raise ValueError('Loss calculations overflow; use moderate finite inputs and variance.')
    return {'table': pd.DataFrame({'observation_id': np.arange(1, len(y)+1),
                                  'observed': y, 'mean': mean, 'residual': residual,
                                  'squared_error': squared}),
            'sse': sse, 'mse': sse/len(y),
            'sum_nll': nll, 'variance': variance}


def paired_runs(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Join separate lower-is-better error records by unique string run_id.

    Both frames require run_id,error columns, exactly the same set of IDs,
    and finite errors. Extra columns are ignored. Input order of a is retained;
    order of b is irrelevant. Returns per-ID d=A-B, mean d, positive/negative
    counts and exact ties. A positive d is a B win. No significance claim or
    silent missing-ID drop; errors may be negative for valid losses such as NLL.
    Example: paired_runs(pd.DataFrame({'run_id':['r1'],'error':[.4]}),
    pd.DataFrame({'run_id':['r1'],'error':[.2]}))['b_wins'] == 1.
    """
    frames = []
    for frame, label in ((a, 'A'), (b, 'B')):
        if not isinstance(frame, pd.DataFrame) or not {'run_id', 'error'} <= set(frame.columns):
            raise ValueError(f'{label} requires a DataFrame with run_id,error columns.')
        f = frame[['run_id', 'error']].copy()
        if f.empty or not all(isinstance(k, str) and k.strip() for k in f.run_id):
            raise ValueError(f'{label} needs nonempty string run IDs.')
        if f.run_id.duplicated().any():
            raise ValueError(f'{label} has duplicate run IDs; one record per run is required.')
        f['error'] = _vector(f.error, f'{label} errors')
        frames.append(f)
    a, b = frames
    if set(a.run_id) != set(b.run_id):
        raise ValueError('A and B must contain exactly the same run IDs.')
    table = a.merge(b, on='run_id', how='left', sort=False, validate='one_to_one',
                    suffixes=('_A', '_B'))
    table['difference_A_minus_B'] = table.error_A-table.error_B
    d = table.difference_A_minus_B.to_numpy()
    return {'table': table, 'mean_difference': float(d.mean()),
            'b_wins': int((d > 0).sum()), 'a_wins': int((d < 0).sum()),
            'ties': int((d == 0).sum()), 'run_count': len(d)}


def scalar_posterior(x: Sequence[float], y: Sequence[float], *,
                     prior_variance: float = 1., noise_sd: float = 1.) -> dict:
    """Fit y_i=x_i*w+epsilon_i with w~N(0,tau^2), independent N(0,noise_sd^2).

    One coefficient, no intercept, known homoscedastic noise. Returns posterior
    mean/variance and coefficients a,b,c of a*w^2-2*b*w+c; square_constant is
    c-b^2/a. Cross-checks against existing BayesianLinear using feature shape
    (n,1). Example: scalar_posterior([1],[1])['variance'] == .5.
    """
    x, y = _vector(x, 'Inputs x'), _vector(y, 'Observed y')
    tau2, noise_sd = _positive(prior_variance, 'Prior variance'), _positive(noise_sd, 'Noise SD')
    if x.shape != y.shape:
        raise ValueError('Inputs x and observed y must have the same (n,) shape.')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        sigma2 = float(np.square(noise_sd))
        prior_precision, data_precision = 1/tau2, float(x@x/sigma2)
        a, b, c = prior_precision+data_precision, float(x@y/sigma2), float(y@y/sigma2)
    if not np.isfinite([sigma2, a, b, c]).all() or sigma2 <= 0:
        raise ValueError('Precision calculations overflow; use moderate inputs and scales.')
    mean, variance = b/a, 1/a
    model = BayesianLinear(noise=noise_sd, prior_variance=tau2).fit(x[:, None], y)
    # Reuse the established fit API, then verify this transparent scalar path.
    prediction = model.predict(np.array([[1.]]))
    if not (np.isclose(prediction['mean'][0], mean) and np.isclose(prediction['var_f'][0], variance)):
        raise ArithmeticError('Scalar precision update disagrees with BayesianLinear.')
    return {'mean': mean, 'variance': variance, 'noise_variance': sigma2,
            'prior_variance': tau2, 'prior_precision': prior_precision,
            'data_precision': data_precision, 'posterior_precision': a,
            'information': b, 'quadratic_constant': c, 'square_constant': c-b*mean}


def scalar_prediction(posterior: dict, x_star: float) -> dict:
    """Predict latent f=x_star*w and a NEW response with independent new noise.

    Multipliers square in a variance. New noise variance is added exactly once;
    a training input reused as x_star still denotes a new response, not the
    already observed y. Returns exact means and the two distinct variances.
    """
    x_star = float(x_star)
    if not np.isfinite(x_star):
        raise ValueError('Prediction input must be finite.')
    s = _positive(posterior['variance'], 'Posterior variance')
    noise = _positive(posterior['noise_variance'], 'Noise variance')
    mean, latent = x_star*float(posterior['mean']), x_star*x_star*s
    if not np.isfinite([mean, latent, latent+noise]).all():
        raise ValueError('Prediction calculations overflow; use a moderate prediction input.')
    return {'x_star': x_star, 'mean_f': mean, 'mean_new_y': mean,
            'variance_f': latent, 'variance_new_y': latent+noise}


def shared_weight_covariance(inputs: Sequence[float], weight_variance: float) -> np.ndarray:
    """Return Var(w)*outer(x,x) for function values sharing the SAME weight.

    Use prior weight variance for a prior kernel or posterior weight variance
    for conditioned function covariance. No observation noise is included.
    Negative cross-covariances are compatible with positive semidefiniteness.
    Example: shared_weight_covariance([-1,2],1)[0,1] == -2.
    """
    x = _vector(inputs, 'Inputs')
    with np.errstate(over='ignore', invalid='ignore'):
        covariance = _positive(weight_variance, 'Weight variance')*np.outer(x, x)
    if not np.isfinite(covariance).all():
        raise ValueError('Covariance calculation overflow; use moderate inputs and variance.')
    return covariance
