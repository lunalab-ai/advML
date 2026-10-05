"""W6 finite-state inference laboratory. Original teaching model, fixed parameters.

The exponential oracle is intentionally limited to ten binary states. It makes
approximation error visible; it is not a production-scale HMM implementation.
All random routines use a local numpy Generator and leave inputs unchanged.
"""
from dataclasses import dataclass
import numpy as np
from advml_pgm import PairwiseModel, enumerate_exact, logsumexp


@dataclass(frozen=True)
class SensorModel:
    """Fixed binary HMM: observations tuple, stay/accuracy/prior_one in (0,1).

    Defaults are y=(1,0,1), stay=.8, accuracy=.8, prior_one=.5. No fitting
    occurs. A state is a length-T binary path; T must be between 1 and 10.
    Invalid/nonfinite probabilities or nonbinary observations raise ValueError.
    """
    observations: tuple = (1, 0, 1)
    stay: float = .8
    accuracy: float = .8
    prior_one: float = .5

    def __post_init__(self):
        obs = tuple(self.observations)
        if not 1 <= len(obs) <= 10 or any(v not in (0, 1) for v in obs):
            raise ValueError('Use 1 to 10 binary observations.')
        for p in (self.stay, self.accuracy, self.prior_one):
            if not np.isfinite(p) or not 0 < p < 1:
                raise ValueError('Probabilities must be finite and strictly between 0 and 1.')
        object.__setattr__(self, 'observations', obs)

    def pairwise(self) -> PairwiseModel:
        """Return W2 PairwiseModel with unary (T,2), transition edges (2,2).

        First unary includes the prior; each unary includes its emission.
        Therefore the sum of joint weights is P(y), not an arbitrary scale.
        """
        y = np.asarray(self.observations)
        unary = np.log(np.where(y[:, None] == np.arange(2), self.accuracy, 1-self.accuracy))
        unary[0] += np.log([1-self.prior_one, self.prior_one])
        transition = np.log([[self.stay, 1-self.stay], [1-self.stay, self.stay]])
        return PairwiseModel(unary, {(t, t+1): transition.copy() for t in range(len(y)-1)})

    def log_joint(self, states: np.ndarray) -> np.ndarray:
        """Return log P(z,y) for binary paths of shape (N,T); output (N,).

        Uses the directed HMM product, independently of W2 enumeration.
        Rejects invalid shape/values; no normalization and no mutation.
        """
        z = np.asarray(states)
        if z.ndim != 2 or z.shape[1] != len(self.observations) or not np.isin(z, [0, 1]).all():
            raise ValueError('states must have shape (N,T) and binary entries.')
        result = np.log(np.where(z[:, 0] == 1, self.prior_one, 1-self.prior_one))
        result += np.log(np.where(z == self.observations, self.accuracy, 1-self.accuracy)).sum(axis=1)
        result += np.log(np.where(z[:, 1:] == z[:, :-1], self.stay, 1-self.stay)).sum(axis=1)
        return result


def exact_posterior(model: SensorModel) -> dict:
    """Return W2 oracle states, probabilities, marginals and log_z plus log_joint.

    States (2**T,T) are lexicographic. probabilities (2**T,) sum to one.
    marginals (T,2) use columns zero/one. Exponential reference only.
    """
    result = enumerate_exact(model.pairwise())
    return {**result, 'log_joint': model.log_joint(result['states'])}


def exact_filter(model: SensorModel) -> dict:
    """Independent forward recursion, O(T): P(z_t=1 | y_1:t), shape (T,).

    Also returns predicted probabilities (T,2), normalizers (T,), and log_z.
    A prefix never uses a future observation. This is filtering, not smoothing.
    """
    belief = np.array([1-model.prior_one, model.prior_one])
    transition = np.array([[model.stay, 1-model.stay], [1-model.stay, model.stay]])
    predicted, filtered, normalizers = [], [], []
    for t, y in enumerate(model.observations):
        pred = belief if t == 0 else belief @ transition
        weighted = pred * np.where(np.arange(2) == y, model.accuracy, 1-model.accuracy)
        normalizers.append(float(weighted.sum()))
        belief = weighted / weighted.sum()
        predicted.append(pred.copy()); filtered.append(float(belief[1]))
    return dict(prob_one=np.array(filtered), predicted=np.array(predicted),
                normalizers=np.array(normalizers), log_z=float(np.log(normalizers).sum()))


def product_probability(states: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Evaluate independent Bernoulli q on paths (N,T), parameters (T,).

    Parameters must lie strictly inside (0,1). Returns (N,), no normalization
    over a subset. On all binary paths these values sum to one.
    """
    states, phi = np.asarray(states), np.asarray(phi, dtype=float)
    if states.ndim != 2 or phi.shape != (states.shape[1],) or not np.isin(states, [0, 1]).all():
        raise ValueError('Incompatible binary states / phi shapes.')
    if not np.isfinite(phi).all() or not ((phi > 0) & (phi < 1)).all():
        raise ValueError('phi must lie strictly inside (0,1).')
    return np.prod(np.where(states == 1, phi, 1-phi), axis=1)


def mean_field(model: SensorModel, initial: float = .5, sweeps: int = 200, tol: float = 1e-11) -> dict:
    """Sequential coordinate-ascent VI for q(z)=product Bernoulli(phi_t).

    initial in (0,1) initializes every factor. sweeps is a positive integer.
    Returns phi (T,), probabilities (2**T,), elbo (initial + each sweep),
    reverse_kl, converged, sweeps. Each update averages log joint over OTHER
    factors then normalizes. This finite oracle implementation is educational.
    Convergence of coordinates does not certify a global family optimum.
    """
    if not 0 < initial < 1 or not np.isfinite(initial) or int(sweeps) != sweeps or sweeps < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError('Invalid initial, sweeps or tolerance.')
    ref = exact_posterior(model); z, lj = ref['states'], ref['log_joint']
    phi = np.full(z.shape[1], initial, dtype=float)
    def objective():
        q = product_probability(z, phi)
        return float(np.sum(q * (lj - np.log(q))))
    trace = [objective()]; converged = False
    for _ in range(int(sweeps)):
        before = phi.copy()
        for j in range(len(phi)):
            other = np.arange(len(phi)) != j
            qo = np.prod(np.where(z[:, other] == 1, phi[other], 1-phi[other]), axis=1)
            expected = np.array([np.sum(qo[z[:, j] == v] * lj[z[:, j] == v]) for v in (0, 1)])
            phi[j] = np.clip(np.exp(expected[1]-logsumexp(expected)), 1e-12, 1-1e-12)
        trace.append(objective())
        if np.max(np.abs(phi-before)) < tol:
            converged = True; break
    q = product_probability(z, phi)
    return dict(phi=phi, probabilities=q, elbo=np.array(trace),
                reverse_kl=float(np.sum(q*(np.log(q)-np.log(ref['probabilities'])))),
                converged=converged, sweeps=len(trace)-1)


def distribution_summary(states: np.ndarray, probabilities: np.ndarray) -> dict:
    """Summarize a normalized path distribution: last-state event and covariance.

    states (N,T), probabilities (N,). Outputs probability_last_one,
    covariance_first_last, posterior_sd_last (spread of Bernoulli state).
    This SD is NOT the Monte Carlo standard error. T=1 yields variance.
    """
    z, p = np.asarray(states), np.asarray(probabilities, dtype=float)
    if z.ndim != 2 or z.shape[1] < 1 or p.shape != (len(z),) or not np.isin(z, [0, 1]).all():
        raise ValueError('Invalid states/probabilities shape.')
    if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError('Use a normalized nonnegative distribution.')
    # Protect Bernoulli variance from roundoff just outside [0,1].
    first, last = (float(np.clip(p @ z[:, j], 0., 1.)) for j in (0, -1))
    return dict(probability_last_one=float(last),
                covariance_first_last=float(p @ (z[:, 0]*z[:, -1])-first*last),
                posterior_sd_last=float(np.sqrt(last*(1-last))))


def mh_transition(model: SensorModel) -> np.ndarray:
    """Exact lazy one-bit MH transition matrix (2**T,2**T), row stochastic.

    Choose uniformly among T bit flips and one self proposal. Symmetric
    proposal, acceptance min(1,p_new/p_old); all rejection mass stays on diagonal.
    Positive model probabilities ensure connected paths and self transitions.
    """
    ref = exact_posterior(model); z, p = ref['states'], ref['probabilities']
    matrix = np.zeros((len(z), len(z))); t = z.shape[1]
    for i in range(len(z)):
        matrix[i, i] += 1/(t+1)
        for bit in range(t):
            j = i ^ (1 << (t-1-bit))
            move = min(1., p[j]/p[i])/(t+1)
            matrix[i, j] += move; matrix[i, i] += 1/(t+1)-move
    return matrix


def sample_paths(model: SensorModel, method: str = 'MC', n: int = 400, seed: int = 19,
                 proposal_one: float = .5, burnin: int = 100) -> dict:
    """MC oracle, self-normalized IS, lazy MH, or full-sweep Gibbs on fixed y.

    n positive draws; burnin nonnegative sweeps/steps for chains only.
    IS uses independent Bernoulli proposal_one in (0,1). Returns paths (n,T),
    weights (n,), histogram probabilities (2**T,), weight_ess for IS only,
    and move_fraction for chains. Rejected MH proposals remain as repeated rows.
    MC needs the exact table: it is a benchmark, not a scalable inference method.
    weight_ess is a weight-concentration measure, not MCMC ESS or unique states.
    """
    if method not in ('MC', 'IS', 'MH', 'Gibbs') or int(n) != n or n < 1 or int(burnin) != burnin or burnin < 0:
        raise ValueError('Invalid method, n or burnin.')
    if not np.isfinite(proposal_one) or not 0 < proposal_one < 1:
        raise ValueError('proposal_one must be strictly inside (0,1).')
    n, burnin = int(n), int(burnin)
    rng = np.random.default_rng(seed); ref = exact_posterior(model)
    z, p = ref['states'], ref['probabilities']; t = z.shape[1]
    weights = np.full(n, 1/n); ess = None; moved = None
    if method == 'MC':
        paths = z[rng.choice(len(z), size=n, p=p)].copy()
    elif method == 'IS':
        paths = (rng.random((n, t)) < proposal_one).astype(int)
        logproposal = np.where(paths, np.log(proposal_one), np.log1p(-proposal_one)).sum(axis=1)
        lw = model.log_joint(paths)-logproposal
        weights = np.exp(lw-logsumexp(lw)); ess = float(1/np.sum(weights**2))
    else:
        current = np.zeros(t, dtype=int); kept = []; moves = []
        for k in range(burnin+n):
            before = current.copy()
            if method == 'MH':
                bit = int(rng.integers(t+1)); proposed = current.copy()
                if bit < t: proposed[bit] = 1-proposed[bit]
                logalpha = model.log_joint(proposed[None, :])[0]-model.log_joint(current[None, :])[0]
                if np.log(rng.random()) < min(0., logalpha): current = proposed
            else:
                for bit in range(t):
                    candidates = np.tile(current, (2, 1)); candidates[:, bit] = [0, 1]
                    lj = model.log_joint(candidates)
                    current[bit] = int(rng.random() < np.exp(lj[1]-logsumexp(lj)))
            if k >= burnin:
                kept.append(current.copy()); moves.append(not np.array_equal(before, current))
        paths = np.array(kept); moved = float(np.mean(moves))
    ids = paths @ (2**np.arange(t-1, -1, -1))
    histogram = np.bincount(ids, weights=weights, minlength=len(z))
    return dict(paths=paths, weights=weights, probabilities=histogram,
                weight_ess=ess, move_fraction=moved)


def particle_filter(model: SensorModel, n: int = 400, seed: int = 19) -> dict:
    """Bootstrap filter with resampling before each next transition, O(N*T).

    Return prob_one (T,), weight_ess (T,), particles/weights (T,N), ancestors
    (T,N; -1 at t=0). Each reported estimate is weighted BEFORE resampling.
    The target at t is P(z_t | y_1:t). Distinct ancestor IDs differ from state
    values: many distinct particles can all be in state one. No future data.
    """
    if int(n) != n or n < 1: raise ValueError('n must be a positive integer.')
    n = int(n); rng = np.random.default_rng(seed)
    particles = (rng.random(n) < model.prior_one).astype(int)
    weights = np.full(n, 1/n); probs = []; ess = []; history = []; wh = []; ah = []
    for t, y in enumerate(model.observations):
        ancestors = np.full(n, -1, dtype=int)
        if t:
            ancestors = rng.choice(n, size=n, p=weights)
            particles = particles[ancestors].copy()
            flip = rng.random(n) >= model.stay
            particles[flip] = 1-particles[flip]
        weights = np.where(particles == y, model.accuracy, 1-model.accuracy)
        weights = weights/weights.sum()
        probs.append(float(weights @ particles)); ess.append(float(1/np.sum(weights**2)))
        history.append(particles.copy()); wh.append(weights.copy()); ah.append(ancestors)
    return dict(prob_one=np.array(probs), weight_ess=np.array(ess),
                particles=np.array(history), weights=np.array(wh), ancestors=np.array(ah))
