# Generative Models, Latent Variable Models, and Variational Autoencoders · Quiz explanations

Six original formative checks for the approved W12 scope.

## 1. Which terms do you need to generate one observation from a latent-variable model?

Draw one z from the prior, then draw x from the conditional decoder distribution at that z.

## 2. What normalizes the two joint masses after observing the light turn on?

Divide both joint masses by their sum, the evidence for the observed event.

## 3. Which KL is zero when the inference distribution equals the true posterior?

The posterior KL that measures the evidence gap. The encoder-to-prior KL can remain positive.

## 4. Why does the scale use half of log variance in the exponent?

Standard deviation is the square root of variance, so its logarithm is half the log variance.

## 5. Which reductions keep the VAE loss on a per-image scale?

Sum negative log-likelihood over pixels, sum KL over latent coordinates, combine the two per image, then average images.

## 6. How do prior generation and latent interpolation choose their decoder inputs?

Prior generation samples z from the prior. Interpolation chooses points along a segment between two specified codes. The endpoints in the lab are encoder means.
