"""Auditable variational inference for an original two-dimensional Gaussian model.

Model: z ~ N(0,I), x|z ~ N(z,S), S=[[1,rho],[rho,1]]. All logarithms
are natural. Model parameters and observations stay fixed during inference.
The public algorithms are shared teaching infrastructure, not exercise keys.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GaussianModel:
    """Exact reference for one observed vector.

    x and mean have shape (2,); noise, precision and covariance have shape
    (2,2); log_evidence is a scalar log density, not a probability.
    Construct with gaussian_model; treat stored arrays as read-only.
    """
    x: np.ndarray
    noise: np.ndarray
    precision: np.ndarray
    covariance: np.ndarray
    mean: np.ndarray
    log_evidence: float


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain two finite numbers")
    return result.copy()


def gaussian_model(rho: float = 0.8, x=(1.0, -1.0)) -> GaussianModel:
    """Return an exact posterior/evidence reference, without fitting or sampling.

    rho: observation-noise correlation, strictly between -1 and 1; it is NOT
    posterior correlation. x: observed length-2 vector, default (1,-1).
    Returned arrays are new. Raises ValueError for invalid dimensions/numbers.
    Example: gaussian_model(.8).mean is approximately [.833333,-.833333].
    """
    rho = float(rho)
    if not np.isfinite(rho) or not -1 < rho < 1:
        raise ValueError("rho must be finite and strictly between -1 and 1")
    x = _vector(x, "x")
    noise = np.array([[1.0, rho], [rho, 1.0]])
    precision = np.eye(2) + np.linalg.inv(noise)
    covariance = np.linalg.inv(precision)
    mean = np.linalg.solve(np.eye(2) + noise, x)
    log_evidence = -np.log(2*np.pi) - .5*np.linalg.slogdet(np.eye(2)+noise)[1] - .5*x@mean
    return GaussianModel(x, noise, precision, covariance, mean, float(log_evidence))


def _q(mean, variance):
    mean, variance = _vector(mean, "mean"), _vector(variance, "variance")
    if np.any(variance <= 0):
        raise ValueError("variances must be strictly positive")
    return mean, variance


def mean_field_optimum(model: GaussianModel) -> tuple[np.ndarray, np.ndarray]:
    """Return the reverse-KL-optimal diagonal Gaussian (mean, variance).

    Both outputs have shape (2,). The mean is the exact posterior mean;
    variance[j] = 1/precision[j,j], generally NOT covariance[j,j].
    Inputs are unchanged; no iteration or download occurs.
    """
    return model.mean.copy(), 1.0 / np.diag(model.precision)


def elbo(model: GaussianModel, mean, variance) -> float:
    """Compute E_q[log p(z)+log p(x|z)] + H(q), directly.

    mean and variance are length-2 vectors for q=N(mean,diag(variance)).
    Output is a scalar in nats; variance means sigma squared, not sigma.
    Includes every normalization constant; does NOT subtract KL from the
    evidence oracle internally. Inputs are not mutated.
    """
    mean, variance = _q(mean, variance)
    inv_noise = np.linalg.inv(model.noise)
    residual = model.x - mean
    prior = -.5*(2*np.log(2*np.pi) + mean@mean + variance.sum())
    likelihood = -.5*(2*np.log(2*np.pi) + np.linalg.slogdet(model.noise)[1]
                       + residual@inv_noise@residual + np.diag(inv_noise)@variance)
    entropy = .5*(2*(1+np.log(2*np.pi)) + np.log(variance).sum())
    return float(prior + likelihood + entropy)


def diagonal_kl(model: GaussianModel, mean, variance) -> float:
    """Return KL(q || exact posterior) in nats using the Gaussian KL formula.

    mean/variance: length-2 vectors, positive variance. This independently
    checks log_evidence - elbo; it is not a sample estimate or a symmetric
    distance. Inputs unchanged. A tiny negative roundoff near zero is possible.
    """
    mean, variance = _q(mean, variance)
    delta = mean - model.mean
    return float(.5*(np.diag(model.precision)@variance
                     + delta@model.precision@delta - 2
                     - np.linalg.slogdet(model.precision)[1]
                     - np.log(variance).sum()))


def elbo_gradient(model: GaussianModel, mean, log_std) -> tuple[np.ndarray, np.ndarray]:
    """Analytic ELBO gradients with respect to mean and log standard deviation.

    Inputs are length-2 vectors. variance=exp(2*log_std). Returns two length-2
    arrays (dL/dmean, dL/dlog_std), for gradient ASCENT. No state is changed.
    Using log standard deviation keeps variances positive.
    """
    mean, log_std = _vector(mean, "mean"), _vector(log_std, "log_std")
    variance = np.exp(2*log_std)
    _q(mean, variance)
    return -model.precision@(mean-model.mean), 1-np.diag(model.precision)*variance


def reparameterized_gradient(model: GaussianModel, mean, log_std,
                             samples: int = 4096, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the same gradients by z=mean+exp(log_std)*epsilon.

    samples: positive integer; seed initializes a local NumPy generator.
    Analytic Gaussian entropy supplies the +1 in the log-std gradient.
    Returns two (2,) sample-average gradients. No global random state,
    downloads or fitting. Estimates fluctuate; no per-step ascent guarantee.
    """
    if int(samples) != samples or samples < 1:
        raise ValueError("samples must be a positive integer")
    mean, log_std = _vector(mean, "mean"), _vector(log_std, "log_std")
    std = np.exp(log_std)
    _q(mean, std**2)
    eps = np.random.default_rng(seed).normal(size=(int(samples), 2))
    z = mean + std*eps
    grad_z = -(z-model.mean)@model.precision
    return grad_z.mean(axis=0), (grad_z*std*eps).mean(axis=0)+1


def run_vi(model: GaussianModel, method: str = "cavi", steps: int = 30,
           initial_mean=(2.0, -2.0), initial_log_std=(0.0, 0.0),
           learning_rate: float = .25) -> dict:
    """Run sequential Gaussian CAVI or analytic ELBO gradient ascent.

    method is 'cavi' or 'gradient'; steps is a nonnegative integer. CAVI
    updates coordinate 0 then 1 using the newest values; a recorded step is
    one complete sweep. Gradient updates mean and log_std jointly with an
    Armijo backtracking line search from learning_rate (>0).
    Returns means/variances of shape (steps+1,2), elbos (steps+1,),
    step_sizes (steps,); NaN for CAVI, and method. Row 0 is initialization.
    Inputs/model are not mutated. Raises on invalid inputs or line-search
    failure; never replaces a failed calculation with a fabricated result.
    """
    if method not in {"cavi", "gradient"}:
        raise ValueError("method must be cavi or gradient")
    if int(steps) != steps or not 0 <= steps <= 10000:
        raise ValueError("steps must be an integer from 0 to 10000")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    mean, log_std = _vector(initial_mean, "initial_mean"), _vector(initial_log_std, "initial_log_std")
    variance = np.exp(2*log_std)
    _q(mean, variance)
    means, variances, bounds = [mean.copy()], [variance.copy()], [elbo(model, mean, variance)]
    rates = []
    for _ in range(int(steps)):
        if method == "cavi":
            for j in range(2):
                delta = mean-model.mean
                other = model.precision[j]@delta - model.precision[j,j]*delta[j]
                mean[j] = model.mean[j] - other/model.precision[j,j]
                variance[j] = 1/model.precision[j,j]
            log_std = .5*np.log(variance)
            rates.append(np.nan)
        else:
            gm, ga = elbo_gradient(model, mean, log_std)
            rate = float(learning_rate)
            old = bounds[-1]
            for trial in range(60):
                candidate_mean, candidate_a = mean+rate*gm, log_std+rate*ga
                with np.errstate(over="ignore", invalid="ignore"):
                    candidate_v = np.exp(2*candidate_a)
                if np.isfinite(candidate_v).all() and np.all(candidate_v > 0):
                    candidate_bound = elbo(model, candidate_mean, candidate_v)
                    if candidate_bound >= old + 1e-4*rate*(gm@gm+ga@ga) - 1e-12:
                        break
                rate *= .5
            else:
                raise RuntimeError("ELBO line search failed; inspect the input")
            mean, log_std, variance = candidate_mean, candidate_a, candidate_v
            rates.append(rate)
        means.append(mean.copy())
        variances.append(variance.copy())
        bounds.append(elbo(model, mean, variance))
    return {"means":np.array(means), "variances":np.array(variances),
            "elbos":np.array(bounds), "step_sizes":np.array(rates), "method":method}


def diagnostics(model: GaussianModel, mean, variance) -> dict:
    """Return scalar/JSON-ready diagnostics for one diagonal approximation.

    Separates approximation gap at the optimal diagonal q from optimization
    gap above that optimum. Gaps are KL quantities in nats. Only a numerical
    optimization gap within 1e-10 of zero is rounded to zero for display.
    """
    mean, variance = _q(mean, variance)
    opt_m, opt_v = mean_field_optimum(model)
    approximation = diagonal_kl(model, opt_m, opt_v)
    total = diagonal_kl(model, mean, variance)
    optimization = total-approximation
    if abs(optimization) < 1e-10:
        optimization = 0.0
    return {"log_evidence":model.log_evidence, "ELBO":elbo(model,mean,variance),
            "reverse_KL":total, "approximation_gap":approximation,
            "optimization_gap":optimization, "q_mean":mean.tolist(),
            "posterior_mean":model.mean.tolist(), "q_variance":variance.tolist(),
            "posterior_marginal_variance":np.diag(model.covariance).tolist(),
            "posterior_correlation":float(model.covariance[0,1]/np.sqrt(np.prod(np.diag(model.covariance))))}


def trace_figure(model: GaussianModel, trace: dict):
    """Return a Matplotlib Figure: covariance geometry and ELBO trace.

    Contours have unit Mahalanobis radius (not 68% joint credible regions).
    trace is a run_vi result. Creates no files or server; caller may savefig.
    """
    from matplotlib.figure import Figure
    fig = Figure(figsize=(11,4.2), layout="constrained")
    ax, bx = fig.subplots(1,2)
    angle = np.linspace(0,2*np.pi,200)
    circle = np.vstack([np.cos(angle),np.sin(angle)])
    best_m, best_v = mean_field_optimum(model)
    for mean, cov, color, label, style in [
        (model.mean, model.covariance, "#244a83", "Exact posterior", "-"),
        (best_m, np.diag(best_v), "#159584", "Best mean field", "--"),
        (trace["means"][-1], np.diag(trace["variances"][-1]), "#db7650", "Current q", "-")]:
        values, vectors = np.linalg.eigh(cov)
        points = mean[:,None] + vectors@np.diag(np.sqrt(values))@circle
        ax.plot(points[0],points[1],style,color=color,label=label,lw=2)
    ax.plot(trace["means"][:,0],trace["means"][:,1],".-",color="#db7650",alpha=.5,label="Mean path")
    ax.set(xlabel="z1", ylabel="z2", title="Unit-Mahalanobis contours")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    bx.plot(trace["elbos"],color="#244a83",label="ELBO",lw=2)
    bx.axhline(model.log_evidence,color="#555555",ls=":",label="Log evidence")
    bx.axhline(elbo(model,best_m,best_v),color="#159584",ls="--",label="Best mean-field ELBO")
    bx.set(xlabel="Sweep / gradient step",ylabel="Nats",title="A bound can improve and remain loose")
    bx.legend(fontsize=8)
    for panel in [ax,bx]:
        panel.grid(alpha=.18)
    return fig


def vi_view(rho: float, method: str, steps: int, learning_rate: float = .25) -> tuple:
    """App callback: settings -> (Figure, JSON-ready diagnostics).

    rho controls observation noise; method is cavi/gradient; steps counts full
    sweeps/steps. Each call constructs a fresh model and trace; no retained fit.
    learning_rate applies only to gradient ascent. Default observed x=(1,-1).
    """
    model = gaussian_model(float(rho))
    trace = run_vi(model, method, int(steps), learning_rate=float(learning_rate))
    metrics = diagnostics(model,trace["means"][-1],trace["variances"][-1])
    metrics["method"] = method
    metrics["steps"] = int(steps)
    if method == "gradient" and len(trace["step_sizes"]):
        metrics["last_accepted_step_size"] = float(trace["step_sizes"][-1])
    return trace_figure(model,trace), metrics


def build_vi_app():
    """Return an unlaunched Gradio Blocks app using vi_view.

    Requires Gradio 6.27.0. No server/download starts here. Call launch with
    share=True in Colab for a temporary link tied to the running runtime.
    Students extend the displayed diagnostics in their own exercise callback.
    """
    import gradio as gr
    with gr.Blocks(title="Variational inference explorer", fill_width=True) as app:
        gr.Markdown("# Variational inference explorer\nCompare exact inference, mean-field approximation and optimization.")
        with gr.Row():
            rho = gr.Slider(-.95,.95,value=.8,step=.05,label="Observation-noise correlation rho")
            method = gr.Dropdown(["cavi","gradient"],value="cavi",label="Method")
        with gr.Row():
            steps = gr.Slider(0,60,value=10,step=1,label="Sweeps / steps")
            rate = gr.Slider(.02,1,value=.25,step=.01,label="Initial gradient step size (gradient only)")
        button = gr.Button("Run inference",variant="primary")
        plot = gr.Plot(label="Posterior geometry and ELBO")
        metrics = gr.JSON(label="Inference diagnostics")
        button.click(vi_view,[rho,method,steps,rate],[plot,metrics],api_name="vi")
        app.load(vi_view,[rho,method,steps,rate],[plot,metrics])
        gr.Markdown("Fixed synthetic observation x=(1,-1). The posterior correlation differs from the noise correlation. Shared links expire with the runtime.")
    return app
