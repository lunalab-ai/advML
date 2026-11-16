# W12 · Calculation and Experiment Workbook

**Advanced Machine Learning for Artificial Intelligence · November 17, 2026**

Keep this sheet beside the lecture note. Record a prediction before running a cell and one inference after observing it.

## Q01 · Evidence and posterior · S02 / A01

The prior is \((0.75,0.25)\) and the likelihoods of \(x=1\) are \((0.2,0.8)\).

| Quantity | State 0 | State 1 |
| --- | --- | --- |
| Joint mass at \(x=1\) | ______ | ______ |
| Posterior given \(x=1\) | ______ | ______ |

Evidence ______. Check the sum of your posterior ______. Explain why the likelihood entries alone are insufficient.

## Q02 · Two KLs · S03 / A02

Hold the model fixed with prior \((0.5,0.5)\). Fill each row in nats.

| \(q\) | Expected log likelihood | Prior KL | ELBO | Posterior KL |
| --- | --- | --- | --- | --- |
| \((0.5,0.5)\) | ___ | ___ | ___ | ___ |
| \((0.2,0.8)\) | ___ | ___ | ___ | ___ |

Which row makes the bound tight? What does the zero in each row refer to?

## Q03 · A differentiable draw · S04 / A03–A04

Use \(\mu=1\), \(\sigma=0.5\), \(\varepsilon=-2\). Write \(\ell\), the correct z, the output from the flagged line, and the analytic derivative with respect to \(\ell\). Use the central difference to check. Change the noise sign and predict the new derivative sign.

## Q04 · Loss accounting · S05 / B02

Take logits \((0,0)\), targets \((0,1)\) and the Gaussian from Q03. Compute the per-image NLL, KL and total for \(\beta=1\). Temporarily replace the pixel sum with a mean. What new total would you expect, and why does the check fail? Restore the original line.

## Q05 · Controlled comparison · S06 / B03–B04

Use the fixed held-out rows. Predict both directions before changing the line for beta.

| Model | Held-out NLL (nats/pixel) | Held-out KL (nats/image) |
| --- | --- | --- |
| AE | ___ | ___ |
| VAE, beta=0.25 | ___ | ___ |
| VAE, beta=1 | ___ | ___ |
| Your beta trial | ___ | ___ |

Record which configuration was changed and which three settings were held fixed. Write one observation and one limitation.

## Q06 · Which path produces this image? · S07 / B05–B06

Explain how z was obtained for the three strips. Pick two coordinates with the same second component. Record the change in the decoded probability map, export both rows, and decide whether the pixel mean alone captures the change. Describe one additional check you would use if the encoder KL became very small.

## Checkpoint quiz

Try these six questions before checking the separate explanations.

1. Which terms do you need to generate one observation from a latent-variable model?
2. What normalizes the two joint masses after observing the light turn on?
3. Which KL is zero when the inference distribution equals the true posterior?
4. Why does the scale use half of log variance in the exponent?
5. Which reductions keep the VAE loss on a per-image scale?
6. How do prior generation and latent interpolation choose their decoder inputs?
