"""Exact checks and a small CPU generative model for Advanced ML W12."""
from __future__ import annotations

import math

import numpy as np


def finite_posterior(prior, likelihood):
    """Return joint masses, evidence, and Bayes posterior for an observed event."""
    prior = np.asarray(prior, dtype=float)
    likelihood = np.asarray(likelihood, dtype=float)
    if prior.ndim != 1 or likelihood.shape != prior.shape or not len(prior):
        raise ValueError("Prior and likelihood must be equal nonempty vectors.")
    if not np.all(np.isfinite(prior)) or not np.all(np.isfinite(likelihood)):
        raise ValueError("Probability masses must be finite.")
    if np.any(prior < 0) or not np.isclose(prior.sum(), 1) or np.any((likelihood < 0) | (likelihood > 1)):
        raise ValueError("Use normalized prior masses and likelihoods in [0, 1].")
    joint = prior * likelihood
    evidence = float(joint.sum())
    if evidence <= 0:
        raise ValueError("The observed event has zero evidence.")
    return joint, evidence, joint / evidence


def finite_elbo(prior, likelihood, q):
    """Independent log-evidence check and the two distinct KL divergences."""
    joint, evidence, posterior = finite_posterior(prior, likelihood)
    q = np.asarray(q, dtype=float)
    if q.shape != joint.shape or not np.isfinite(q).all() or np.any(q < 0) or not np.isclose(q.sum(), 1):
        raise ValueError("q must be normalized and use the same states.")
    positive = q > 0
    if np.any(joint[positive] <= 0):
        raise ValueError("This exercise requires q to stay in the joint support.")
    mass = q[positive]
    likelihood = np.asarray(likelihood)[positive]
    prior = np.asarray(prior)[positive]
    expected_loglik = float(mass @ np.log(likelihood))
    prior_kl = float(mass @ np.log(mass / prior))
    posterior_kl = float(mass @ np.log(mass / posterior[positive]))
    return dict(expected_loglik=expected_loglik, prior_kl=prior_kl, posterior_kl=posterior_kl,
                elbo=expected_loglik-prior_kl, log_evidence=math.log(evidence), posterior=posterior)


def shape_images(n, seed):
    """Draw a binary circle/square image; the hidden factors are for plotting."""
    rng = np.random.default_rng(seed)
    position = rng.uniform(4.5, 11, size=(n, 2))
    radius = rng.uniform(2.3, 3.6, size=n)
    kind = rng.integers(0, 2, size=n)
    y, x = np.mgrid[:16, :16]
    dx = x[None] - position[:, 0, None, None]
    dy = y[None] - position[:, 1, None, None]
    circle = dx**2 + dy**2 <= radius[:, None, None]**2
    square = np.maximum(np.abs(dx), np.abs(dy)) <= radius[:, None, None] / 1.12
    shape = np.where(kind[:, None, None] == 0, circle, square)
    probability = np.where(shape, .97, .025)
    image = (rng.random(shape.shape) < probability).astype(np.float32)
    return image.reshape(n, 256), {"kind": kind, "center": position, "radius": radius}


def reparameterize(mu, logvar, epsilon):
    """Same-shaped tensors with z = mu + sigma*epsilon and a fixed base draw."""
    if mu.shape != logvar.shape or mu.shape != epsilon.shape:
        raise ValueError("mu, logvar, and epsilon must share [batch, latent] shape.")
    sigma = (0.5 * logvar).exp()
    return mu + sigma * epsilon


def diagonal_kl(mu, logvar):
    """One KL per image, summing latent coordinates."""
    if mu.ndim != 2 or mu.shape != logvar.shape:
        raise ValueError("Expected [batch, latent] arrays with identical shape.")
    return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=-1)


def vae_objective(logits, x, mu, logvar, beta=1.0, *, kl_fn=diagonal_kl):
    """Bernoulli NLL sums pixels, KL sums latents, both retain batch until the end."""
    import torch.nn.functional as F

    if logits.ndim != 2 or logits.shape != x.shape or len(x) != len(mu):
        raise ValueError("Expected [batch, pixels] logits and aligned latent rows.")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative.")
    nll = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=-1)
    kl = kl_fn(mu, logvar)
    if kl.shape != nll.shape:
        raise ValueError("NLL and KL must each have shape [batch].")
    return (nll + beta * kl).mean(), nll, kl


def build_model(variational=True):
    import torch
    from torch import nn

    class ShapeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(256, 80), nn.ReLU(), nn.Linear(80, 40), nn.ReLU())
            self.mean = nn.Linear(40, 2)
            self.log_variance = nn.Linear(40, 2)
            self.decoder = nn.Sequential(nn.Linear(2, 40), nn.ReLU(), nn.Linear(40, 80), nn.ReLU(),
                                         nn.Linear(80, 256))
            self.variational = variational

        def encode(self, x):
            h = self.encoder(x)
            mu = self.mean(h)
            lv = self.log_variance(h).clamp(-9, 6) if self.variational else torch.zeros_like(mu)
            return mu, lv

    return ShapeModel()


def train_model(x_train, *, beta=1.0, variational=True, epochs=15, seed=1742,
                draw_fn=reparameterize, objective_fn=vae_objective):
    """Fit only training rows. Student-edited functions are the actual callbacks."""
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    split_rng = torch.Generator().manual_seed(seed + 1)
    noise_rng = torch.Generator().manual_seed(seed + 2)
    model = build_model(variational)
    x_train = torch.as_tensor(x_train, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    history = []
    for epoch in range(epochs):
        loss_sum = 0.0
        for batch in torch.randperm(len(x_train), generator=split_rng).split(64):
            x = x_train[batch]
            mu, lv = model.encode(x)
            if variational:
                eps = torch.randn(mu.shape, generator=noise_rng)
                z = draw_fn(mu, lv, eps)
                loss, _, _ = objective_fn(model.decoder(z), x, mu, lv, beta)
            else:
                loss = F.binary_cross_entropy_with_logits(model.decoder(mu), x, reduction="sum") / len(x)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Training objective became nonfinite.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(x)
        history.append({"epoch": epoch + 1, "training_objective": loss_sum / len(x_train)})
    model.eval()
    return model, history


def evaluate_model(model, evaluation, draws=4, seed=920):
    """Use the same held-out rows and the same noise draws for each VAE."""
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(evaluation, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mu, lv = model.encode(x)
        if model.variational:
            nll = []
            for _ in range(draws):
                eps = torch.randn(mu.shape, generator=g)
                z = reparameterize(mu, lv, eps)
                nll.append(F.binary_cross_entropy_with_logits(model.decoder(z), x).item())
            return {"heldout_nll_nats_per_pixel": float(np.mean(nll)),
                    "heldout_kl_nats_per_image": diagonal_kl(mu, lv).mean().item()}
        nll = F.binary_cross_entropy_with_logits(model.decoder(mu), x).item()
        return {"heldout_nll_nats_per_pixel": nll, "heldout_kl_nats_per_image": None}


def decoded_means(model, z):
    import torch

    with torch.no_grad():
        z = torch.as_tensor(z, dtype=torch.float32).reshape(-1, 2)
        return model.decoder(z).sigmoid().cpu().numpy().reshape(-1, 16, 16)


def mean_codes(model, x):
    import torch

    with torch.no_grad():
        mu, _ = model.encode(torch.as_tensor(x, dtype=torch.float32))
        return mu.cpu().numpy()
