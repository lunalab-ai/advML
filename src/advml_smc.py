"""Linear-Gaussian filtering oracles and bootstrap particle filtering.

State z[0] ~ N(0,1); z[t]=a*z[t-1]+process_sd*epsilon[t] for t>=1;
y[t]=z[t]+observation_sd*eta[t]. Both noise parameters are STANDARD DEVIATIONS.
Original course code following Murphy Advanced Topics 13.1-13.2.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from advml_sampling import normalize_log_weights, weight_ess, _count, _positive


def _parameters(a, process_sd, observation_sd):
    if not np.isfinite(a):
        raise ValueError('a must be finite')
    return float(a), _positive(process_sd, 'process_sd'), _positive(observation_sd, 'observation_sd')


def simulate_tracking(steps: int = 40, a: float = .9, process_sd: float = .35,
                      observation_sd: float = .5, seed: int = 20260929) -> dict:
    """Generate fixed synthetic truth and observations, both shape (steps,).

    steps>=1, finite a, positive noise SDs. Initial state has variance one;
    no transition is applied before observation y[0]. Local RNG, no file/network
    access. Keep this seed fixed while varying particle-filter seeds.
    """
    steps = _count(steps)
    a, process_sd, observation_sd = _parameters(a, process_sd, observation_sd)
    rng = np.random.default_rng(seed)
    z = np.empty(steps)
    z[0] = rng.normal()
    for t in range(1, steps):
        z[t] = a*z[t-1] + process_sd*rng.normal()
    y = z + observation_sd*rng.normal(size=steps)
    return {'truth': z, 'observations': y, 'a': a, 'process_sd': process_sd,
            'observation_sd': observation_sd, 'seed': int(seed)}


def kalman_filter(observations, a: float = .9, process_sd: float = .35,
                  observation_sd: float = .5) -> dict:
    """Exact scalar Gaussian filtering reference for the stated initial model.

    observations: finite 1D y of length T>=1. Return mean/variance/predicted_mean/
    predicted_variance/gain arrays (T,), and log_evidence scalar. Variance is
    posterior uncertainty, not the Monte Carlo variance of an estimator.
    Uses data up through t only (filtering); no future-data smoothing.
    """
    a, process_sd, observation_sd = _parameters(a, process_sd, observation_sd)
    y = np.asarray(observations, float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('observations must be a finite nonempty 1D array')
    m, v = 0., 1.
    result = {k: np.empty(len(y)) for k in ('mean','variance','predicted_mean','predicted_variance','gain')}
    log_evidence = 0.
    for t, observation in enumerate(y):
        if t:
            m, v = a*m, a*a*v + process_sd**2
        result['predicted_mean'][t], result['predicted_variance'][t] = m, v
        innovation_variance = v + observation_sd**2
        gain = v / innovation_variance
        log_evidence += -.5*(np.log(2*np.pi*innovation_variance) + (observation-m)**2/innovation_variance)
        m = m + gain*(observation-m)
        v = (1-gain)*v
        result['mean'][t], result['variance'][t], result['gain'][t] = m, v, gain
    result['log_evidence'] = float(log_evidence)
    return result


def systematic_resample(weights, rng: np.random.Generator) -> np.ndarray:
    """Return N ancestor indices by systematic resampling of N weights.

    weights: finite nonnegative 1D, positive sum; rng is a local NumPy Generator.
    Draw one uniform offset in [0,1/N) and use N equally spaced positions.
    Output shape (N,), integer indices in [0,N). Consumes rng but does not
    mutate weights. Resampled particles are dependent copies, not new evidence.
    """
    w = np.asarray(weights, float)
    weight_ess(w)
    w = w / w.sum()
    positions = (rng.random() + np.arange(len(w))) / len(w)
    cdf = np.cumsum(w)
    cdf[-1] = 1.  # Protect the final boundary against floating-point rounding.
    return np.searchsorted(cdf, positions, side='right')


def particle_filter(observations, particles: int = 500, a: float = .9,
                    process_sd: float = .35, observation_sd: float = .5,
                    threshold: float = .5, seed: int = 0) -> dict:
    """Bootstrap SIS/PF with optional resampling AFTER each measurement update.

    y: finite 1D length T. particles=N>=2. threshold in [0,1]; zero means SIS.
    Resample when pre-resampling weight ESS < threshold*N. Propagation uses the
    state transition, so incremental weights are observation likelihoods.
    Initial particles are N(0,1), weighted by y[0]. All means/variances and
    particle/weight arrays are recorded BEFORE resampling. Return arrays mean,
    variance, ess_before, ess_after, unique_roots_after, resampled of shape (T,);
    samples, weights, root_labels_before, resampling_indices of shape (T,N).
    Roots refer to initial particle IDs; distinct positions do not imply distinct
    ancestry. log_evidence estimates log likelihood and is not unbiased in log.
    A collapsed or invalid computation raises; no uniform-weight fallback.
    """
    a, process_sd, observation_sd = _parameters(a, process_sd, observation_sd)
    n = _count(particles, 2)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0,1]')
    y = np.asarray(observations, float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('observations must be a finite nonempty 1D array')
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    logw = np.full(n, -np.log(n))
    roots = np.arange(n)
    out = {k: np.empty(len(y)) for k in ('mean','variance','ess_before','ess_after','unique_roots_after')}
    out.update(samples=np.empty((len(y),n)), weights=np.empty((len(y),n)),
               root_labels_before=np.empty((len(y),n),dtype=int),
               resampling_indices=np.empty((len(y),n),dtype=int),
               resampled=np.zeros(len(y), dtype=bool))
    log_evidence = 0.
    for t, observation in enumerate(y):
        if t:
            z = a*z + process_sd*rng.normal(size=n)
        log_likelihood = -.5*((observation-z)/observation_sd)**2 - np.log(observation_sd*np.sqrt(2*np.pi))
        updated_logw = logw + log_likelihood
        w = normalize_log_weights(updated_logw)
        log_evidence += float(logsumexp(updated_logw))
        out['samples'][t], out['weights'][t] = z, w
        out['root_labels_before'][t] = roots
        mean = float(w @ z)
        out['mean'][t], out['variance'][t] = mean, w @ (z-mean)**2
        ess = weight_ess(w)
        out['ess_before'][t] = ess
        if ess < threshold*n:
            indices = systematic_resample(w, rng)
            z, roots = z[indices], roots[indices]
            logw = np.full(n, -np.log(n))
            out['resampled'][t], out['ess_after'][t] = True, float(n)
        else:
            indices = np.arange(n)
            logw = updated_logw - logsumexp(updated_logw)
            out['ess_after'][t] = ess
        out['resampling_indices'][t] = indices
        out['unique_roots_after'][t] = len(np.unique(roots))
    out['log_evidence'] = log_evidence
    return out
