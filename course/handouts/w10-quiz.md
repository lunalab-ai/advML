# W4-W7 Conceptual Rebuild and Midterm Q3-Q5 · Quiz explanations

New conceptual checkpoints for accepted W10 review; separate explanations are public by course policy. Original Q3–Q5 detailed answers remain private.

## 1. How do a variational distribution and an importance proposal use the symbol q differently?

In VI, q is an approximating distribution optimized within a chosen family. In importance sampling, q is the draw-generating proposal. The roles may be connected, but their objectives must be stated.

## 2. Why do direct IS and SNIS use different denominators?

Direct IS is a proposal-draw sample average of the weighted function, dividing by draw count. SNIS normalizes by raw-weight mass; normalized weights already sum to one and receive no extra division by draw count.

## 3. Why must repeated draws remain separate in weight ESS?

Each draw contributes to the sum and squared-weight sum, even if its value repeats. Collapsing equal states changes the diagnostic. Weight ESS describes concentration over draws, not MCMC autocorrelation or coverage.

## 4. When does a Gaussian NLL comparison reduce to an SSE comparison?

With the same observation count and common known variance, the normalization term is common and NLL differences scale SSE differences. Different variances change both normalization and scaling, so SSE alone is insufficient.

## 5. Why can a correct mean paired difference fail to prove correct run alignment?

Using every value once makes the mean difference equal A mean minus B mean under any permutation. Win counts can also coincide. Validate unique IDs, identical ID sets and the entire ID-labelled difference vector.

## 6. Which uncertainty changes when posterior simulation draws increase?

With fixed data/model and iid posterior draws, Monte Carlo error in an estimated mean decreases with draw count. The exact posterior and exact latent/new-response predictive uncertainties remain unchanged.
