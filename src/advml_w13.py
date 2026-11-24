"""Small CPU DDPM for a two-component Gaussian mixture."""

from dataclasses import dataclass
from math import pi

import torch
from torch import nn
from torch.nn.functional import mse_loss


@dataclass(frozen=True)
class Schedule:
    """One-based DDPM coefficients with an identity at index zero."""

    beta: torch.Tensor
    alpha: torch.Tensor
    alpha_bar: torch.Tensor
    posterior_variance: torch.Tensor

    @property
    def steps(self):
        return len(self.beta) - 1


def linear_schedule(steps=160):
    """Return CPU coefficients with beta increasing from 0.0001 to 0.08."""
    beta = torch.cat((torch.zeros(1), torch.linspace(0.0001, 0.08, steps)))
    alpha = 1.0 - beta
    alpha_bar = alpha.cumprod(0)
    variance = torch.zeros_like(beta)
    variance[1:] = beta[1:] * (1.0 - alpha_bar[:-1]) / (1.0 - alpha_bar[1:])
    return Schedule(beta, alpha, alpha_bar, variance)


def gaussian_mixture(n, seed):
    """Sample n points [n,2] and component IDs [n], using a private CPU RNG."""
    rng = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-1.8, 0.0], [1.8, 0.0]])
    component = torch.randint(0, len(centers), (n,), generator=rng)
    points = centers[component] + 0.32 * torch.randn(n, 2, generator=rng)
    return points, component, centers


def add_noise(clean, time, epsilon, schedule):
    """Draw the forward marginal using clean/epsilon [batch,2] and time [batch]."""
    signal_power = schedule.alpha_bar[time, None]
    return signal_power.sqrt() * clean + (1.0 - signal_power).sqrt() * epsilon


class NoisePredictor(nn.Module):
    """Map each 2D noisy point and time to a predicted 2D noise vector."""

    def __init__(self, steps, width=64):
        super().__init__()
        self.steps = steps
        self.network = nn.Sequential(
            nn.Linear(5, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2),
        )

    def forward(self, points, time):
        normalized = time[:, None].to(points.dtype) / self.steps
        phase = (2.0 * pi) * normalized
        features = torch.cat((points, normalized, phase.sin(), phase.cos()), dim=1)
        return self.network(features)


def fit_predictor(train, schedule, updates=1800, batch_size=256, seed=24):
    """Train on random rows, time steps and noise; return the model and MSE trace."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = NoisePredictor(schedule.steps)
    rng = torch.Generator().manual_seed(seed + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    losses = []
    model.train()
    for _ in range(updates):
        indices = torch.randint(0, len(train), (batch_size,), generator=rng)
        time = torch.randint(1, schedule.steps + 1, (batch_size,), generator=rng)
        clean = train[indices]
        epsilon = torch.randn(clean.shape, generator=rng)
        predicted = model(add_noise(clean, time, epsilon, schedule), time)
        loss = mse_loss(predicted, epsilon)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model.eval(), losses


@torch.no_grad()
def reverse_step(model, points, step, noise, schedule):
    """Return x_(t-1) [batch,2] from x_t and a fresh standard Gaussian draw."""
    time = torch.full((len(points),), step, dtype=torch.long)
    prediction = model(points, time)
    mean = (points - schedule.beta[step] * prediction /
            (1.0 - schedule.alpha_bar[step]).sqrt()) / schedule.alpha[step].sqrt()
    return mean + schedule.posterior_variance[step].sqrt() * noise


@torch.no_grad()
def generate(model, initial, schedule, seed):
    """Run all reverse steps from initial [batch,2] and return fresh 2D samples."""
    rng = torch.Generator().manual_seed(seed)
    points = initial
    for step in range(schedule.steps, 0, -1):
        noise = torch.randn(points.shape, generator=rng)
        points = reverse_step(model, points, step, noise, schedule)
    return points


def mode_counts(points, centers, radius=0.70):
    """Count points within each of two disjoint center disks and outside both."""
    hits = torch.cdist(points, centers) < radius
    return [*hits.sum(dim=0).tolist(), (~hits.any(dim=1)).sum().item()]
