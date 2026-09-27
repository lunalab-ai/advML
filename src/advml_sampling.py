"""Transparent CPU Monte Carlo algorithms for Advanced ML W5.

Original course implementations; Murphy Advanced Topics Chapters 11-13 guide
the algorithms. All seeds use local generators. Arrays are new; no fitting,
downloads, global RNG mutation or hidden persistent model state occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.special import logsumexp
from advml_vi import GaussianModel, gaussian_model


def _count(n, minimum=1):
    if not np.isfinite(n) or int(n) != n or n < minimum:
        raise ValueError(f"count must be an integer >= {minimum}")
    return int(n)


def _positive(x, name):
    x = float(x)
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return x


def _log_normal(z, mean, covariance):
    delta = np.asarray(z, dtype=float) - mean
    precision = np.linalg.inv(covariance)
    return (-.5 * (len(mean) * np.log(2 * np.pi)
                   + np.linalg.slogdet(covariance)[1]
                   + np.einsum('...i,ij,...j->...', delta, precision, delta)))


@dataclass(frozen=True)
class Target:
    """A two-dimensional benchmark with exact moments and an iid oracle.

    kind is 'gaussian' or 'bimodal'; mean shape (2,), covariance (2,2).
    Construct with make_target. Bimodal means are (-3,0) and (3,0), equal
    weights, within-component covariance .25 I. Treat arrays as read-only.
    The iid oracle is available for this benchmark, not for general inference.
    """
    kind: str
    mean: np.ndarray
    covariance: np.ndarray
    model: GaussianModel | None = None

    def log_density(self, z: np.ndarray) -> np.ndarray:
        """Normalized log density at z of shape (...,2); return shape (...,).

        A single length-2 vector returns a NumPy scalar. Natural logarithms.
        This benchmark knows the normalizer; MH needs only a proportional
        density. No samples or model state are changed by evaluation.
        """
        z = np.asarray(z, dtype=float)
        if z.ndim < 1 or z.shape[-1] != 2 or not np.isfinite(z).all():
            raise ValueError('z must have finite coordinates and last dimension 2')
        if self.kind == 'gaussian':
            return _log_normal(z, self.mean, self.covariance)
        left = _log_normal(z, np.array([-3., 0.]), .25 * np.eye(2))
        right = _log_normal(z, np.array([3., 0.]), .25 * np.eye(2))
        return np.logaddexp(left, right) - np.log(2.)

    def sample(self, n: int = 1000, seed: int = 0) -> np.ndarray:
        """Return n independent oracle draws, shape (n,2), using local seed.

        For the mixture each row chooses its own component. This is an exact
        benchmark sampler, not an MCMC transition or a weighted sample.
        """
        n = _count(n)
        rng = np.random.default_rng(seed)
        if self.kind == 'gaussian':
            return rng.multivariate_normal(self.mean, self.covariance, n)
        z = rng.normal(0., .5, size=(n, 2))
        z[:, 0] += rng.choice([-3., 3.], size=n)
        return z


def make_target(kind: str = 'gaussian', rho: float = .8) -> Target:
    """Build the W4 Gaussian posterior or a deliberately difficult mixture.

    rho is the W4 OBSERVATION-noise correlation, not posterior correlation.
    gaussian_model validates -1 < rho < 1. The mixture ignores rho and has
    exact mean [0,0], covariance diag(9.25,.25). No fitting or random draws.
    Example: make_target().mean is [5/6,-5/6].
    """
    if kind == 'gaussian':
        m = gaussian_model(rho)
        return Target(kind, m.mean.copy(), m.covariance.copy(), m)
    if kind == 'bimodal':
        return Target(kind, np.zeros(2), np.diag([9.25, .25]))
    raise ValueError("kind must be 'gaussian' or 'bimodal'")


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Normalize a nonempty 1D log-weight array stably; return sum-one weights.

    Minus infinity is allowed (zero weight); NaN, plus infinity and all-zero
    mass raise ValueError. A common finite offset changes no normalized weight.
    Example: log([1,2,7]) gives [.1,.2,.7]. No in-place mutation.
    """
    lw = np.asarray(log_weights, dtype=float)
    if (lw.ndim != 1 or len(lw) == 0 or np.isnan(lw).any()
            or np.isposinf(lw).any() or not np.isfinite(lw).any()):
        raise ValueError('log weights must be 1D with some finite mass, no NaN/+inf')
    return np.exp(lw - logsumexp(lw))


def weight_ess(weights: np.ndarray) -> float:
    """Return 1/sum(normalized_weights**2), a WEIGHT-concentration diagnostic.

    Accept nonnegative finite 1D weights with positive sum; normalize internally.
    This is not MCMC autocorrelation ESS and cannot detect unsampled regions.
    Uniform N weights yield N; a single nonzero weight yields 1.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('weights must be finite, nonnegative and have positive mass')
    w = w / w.sum()
    return float(1. / (w @ w))


def importance_sampling(target: Target, n: int = 2000, proposal_scale: float = 1.5,
                        proposal_shift: float = 0., seed: int = 0) -> dict:
    """Draw iid Gaussian proposals and return direct-IS/SNIS building blocks.

    q has mean target.mean + [proposal_shift,0] and covariance
    proposal_scale**2 * target.covariance. Scale is a SD multiplier (>0),
    shift is in coordinate units. Return samples (n,2), normalized weights
    (n,), log_weights (n,), weight_ess, log_normalizer_estimate, and evaluation
    counts. Both benchmark targets have support R^2; this does not guarantee
    finite variance for a narrow q. No support-coverage test is inferred from ESS.
    """
    n = _count(n, 2)
    scale = _positive(proposal_scale, 'proposal_scale')
    if not np.isfinite(proposal_shift):
        raise ValueError('proposal_shift must be finite')
    mean = target.mean + [float(proposal_shift), 0.]
    covariance = scale ** 2 * target.covariance
    z = np.random.default_rng(seed).multivariate_normal(mean, covariance, n)
    lw = target.log_density(z) - _log_normal(z, mean, covariance)
    w = normalize_log_weights(lw)
    return {'samples': z, 'weights': w, 'log_weights': lw,
            'weight_ess': weight_ess(w),
            'log_normalizer_estimate': float(logsumexp(lw) - np.log(n)),
            'target_evaluations': n, 'proposal_evaluations': n}


def snis_summary(values: np.ndarray, weights: np.ndarray) -> dict:
    """Summarize a scalar function on iid proposals with normalized IS weights.

    values/weights are length n>=2. Return estimate, weighted posterior SD,
    and plug-in delta-method MCSE sqrt(n/(n-1)*sum(W^2*(f-estimate)^2)).
    MCSE is asymptotic, requires suitable finite moments and adequate coverage;
    a small reported value is not a certificate. Not for correlated MCMC draws.
    """
    f, w = np.asarray(values, float), np.asarray(weights, float)
    if f.ndim != 1 or f.shape != w.shape or len(f) < 2 or not np.isfinite(f).all():
        raise ValueError('matching finite 1D function values and weights required')
    weight_ess(w)  # Validate the weight contract before normalization.
    w = w / w.sum()
    estimate = float(w @ f)
    centered = f - estimate
    return {'estimate': estimate,
            'posterior_sd': float(np.sqrt(w @ centered**2)),
            'mcse': float(np.sqrt(len(f)/(len(f)-1) * ((w*centered) @ (w*centered))))}


def random_walk_mh(log_density: Callable, initial, draws: int = 2000,
                   scale: float = .5, seed: int = 0) -> dict:
    """Run Gaussian random-walk Metropolis; retain EVERY rejected state.

    log_density maps a finite vector (d,) to a scalar log target up to a
    constant. initial is a finite vector with finite log density; scale is
    proposal SD in each coordinate. draws counts transitions, excludes the
    initial state, and includes warmup if requested by the caller. Return
    samples (draws,d), accepted bool (draws,), initial copy, target_evaluations
    draws+1. No adaptation, thinning or automatic convergence claim.
    """
    draws, scale = _count(draws), _positive(scale, 'scale')
    current = np.asarray(initial, float).copy()
    if current.ndim != 1 or not current.size or not np.isfinite(current).all():
        raise ValueError('initial must be a finite nonempty vector')
    current_log = float(log_density(current))
    if not np.isfinite(current_log):
        raise ValueError('initial log density must be finite')
    initial_copy = current.copy()
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, len(current)))
    accepted = np.zeros(draws, dtype=bool)
    for i in range(draws):
        proposal = current + rng.normal(0., scale, len(current))
        proposed_log = float(log_density(proposal))
        if np.isnan(proposed_log) or np.isposinf(proposed_log):
            raise ValueError('log target returned NaN/+inf')
        log_alpha = min(0., proposed_log - current_log)
        if np.log(rng.uniform(np.nextafter(0., 1.), 1.)) < log_alpha:
            current, current_log = proposal, proposed_log
            accepted[i] = True
        samples[i] = current  # Rejection is a transition back to the same state.
    return {'samples': samples, 'accepted': accepted, 'initial': initial_copy,
            'target_evaluations': draws + 1}


def gaussian_gibbs(model: GaussianModel, initial=(0., 0.), draws: int = 2000,
                   seed: int = 0) -> dict:
    """Sample exact Gaussian conditionals, coordinate 0 then coordinate 1.

    draws counts complete sweeps. Conditional variance is 1/precision[j,j];
    the mean uses the latest other coordinate. Return samples (draws,2),
    substeps (2*draws+1,2), including initial, and conditional_draws=2*draws.
    This is stochastic Gibbs, not deterministic CAVI or marginal iid sampling.
    """
    draws = _count(draws)
    z = np.asarray(initial, float).copy()
    if z.shape != (2,) or not np.isfinite(z).all():
        raise ValueError('initial must be a finite length-2 vector')
    rng = np.random.default_rng(seed)
    samples, path = np.empty((draws, 2)), [z.copy()]
    for i in range(draws):
        for j in range(2):
            k = 1-j
            mean = model.mean[j] - model.precision[j, k] / model.precision[j, j] * (z[k]-model.mean[k])
            z[j] = rng.normal(mean, np.sqrt(1. / model.precision[j, j]))
            path.append(z.copy())
        samples[i] = z
    return {'samples': samples, 'substeps': np.asarray(path), 'conditional_draws': 2*draws}


def chain_diagnostics(values: np.ndarray) -> dict:
    """ArviZ 0.22 diagnostics for one scalar observable, shape (chains,draws).

    Supply >=2 chains, >=20 POST-warmup draws each, without flattening chains.
    Return rank-normalized folded/split R-hat, bulk/tail/mean ESS, mean MCSE,
    pooled mean/SD and retained count. Diagnostics concern the supplied f(z),
    not every possible posterior quantity. Nonfinite diagnostic outputs become
    None and an explicit warning; they are never replaced with a passing value.
    """
    import arviz as az
    x = np.asarray(values, float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 20 or not np.isfinite(x).all():
        raise ValueError('use finite (chains>=2, post-warmup draws>=20) values')
    metrics = {'rhat_rank': float(az.rhat(x, method='rank')),
               'ess_bulk': float(az.ess(x, method='bulk')),
               'ess_tail': float(az.ess(x, method='tail')),
               'ess_mean': float(az.ess(x, method='mean')),
               'mcse_mean': float(az.mcse(x, method='mean'))}
    result = {k: v if np.isfinite(v) else None for k, v in metrics.items()}
    unreliable = any(v is None for v in result.values())
    unreliable = unreliable or result['rhat_rank'] > 1.01 or result['ess_bulk'] < 100
    result.update(mean=float(x.mean()), posterior_sd=float(x.std(ddof=1)), retained_draws=int(x.size),
                  diagnostic_warning=('Investigate mixing; numerical error estimates may be unreliable.'
                                      if unreliable else 'No flag here is a proof of convergence; inspect modes and traces.'))
    return result


def autocorrelation(values: np.ndarray, max_lag: int = 60) -> np.ndarray:
    """Return the biased sample ACF at lags 0..max_lag for one 1D trace.

    Used for visualization, not for computing ESS. Constant traces return NaN
    beyond lag zero to avoid pretending that a stuck chain mixes perfectly.
    """
    x = np.asarray(values, float)
    max_lag = _count(max_lag, 0)
    if x.ndim != 1 or len(x) <= max_lag or not np.isfinite(x).all():
        raise ValueError('finite 1D trace must be longer than max_lag')
    x = x-x.mean()
    denom = x @ x
    if denom == 0:
        return np.r_[1., np.full(max_lag, np.nan)]
    return np.array([x @ x / denom] + [x[:-k] @ x[k:] / denom for k in range(1, max_lag+1)])


def mh_transition_matrix(probabilities, proposal) -> np.ndarray:
    """Exact finite-state MH transition matrix, including reject/self mass.

    probabilities is a positive length-K target (normalized internally).
    proposal is a KxK row-stochastic matrix q[j|i]. Return P[i,j]. Reverse
    proposal probabilities enter the ratio; q[i,j]=0 permits no such move.
    This small exact oracle supports the detailed-balance hand calculation.
    """
    p, q = np.asarray(probabilities, float), np.asarray(proposal, float)
    if p.ndim != 1 or np.any(p <= 0) or not np.isfinite(p).all():
        raise ValueError('target probabilities must be positive and finite')
    if q.shape != (len(p), len(p)) or not np.isfinite(q).all() or np.any(q < 0) or not np.allclose(q.sum(1), 1):
        raise ValueError('proposal must be a row-stochastic KxK matrix')
    transition = np.zeros_like(q)
    for i in range(len(p)):
        for j in range(len(p)):
            if i != j and q[i,j] > 0:
                transition[i,j] = q[i,j] * min(1., p[j]*q[j,i]/(p[i]*q[i,j]))
        transition[i,i] = 1.-transition[i].sum()
    return transition
