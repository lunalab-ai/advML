# Monte Carlo Inference: Importance Sampling, MCMC, and Sequential Monte Carlo · Quiz explanations

Original formative checkpoint explanations for the accepted W5 lesson; no graded assignment.

## 1. Why can posterior SD stay the same while MCSE decreases?

Posterior SD describes uncertainty in a latent value under the fixed model and data. MCSE describes repeated-run uncertainty in a numerical expectation estimate. With iid draws and finite variance, the latter scales as SD of the function values divided by square root N. More posterior draws do not constitute additional observed data. In the Gaussian example, SD stays about 0.636 while MCSE changes from about 0.064 at N=100 to 0.032 at N=400.

## 2. Why is full support necessary but insufficient for reliable importance sampling?

If the proposal assigns zero density to a required target region, samples never visit that region and observed weights cannot recover its contribution. Even with full support, a proposal with inadequate tails may produce a very large or infinite second moment for the weighted integrand. A narrow Gaussian illustrates the distinction: its support is the entire plane, yet its importance estimator may be unstable. Weight ESS can warn about observed concentration but cannot certify unseen support coverage.

## 3. Why does cancellation of the normalizing constant not make finite-sample SNIS unbiased?

SNIS divides a random weighted-function average by a random weight average. The unknown common multiplier cancels algebraically, but expectation of a ratio is generally not the ratio of expectations. Under appropriate support and integrability conditions with a finite positive normalizer, both averages converge and their ratio is consistent. This asymptotic statement does not imply finite-sample unbiasedness. The worked example gives direct IS 1.6 and SNIS 1.333... for one realization, versus exact mean 1.3; one realization alone cannot establish bias.

## 4. Why does discarding rejected MH proposals bias the empirical distribution?

Rejected proposals leave the current state unchanged, and that repeated state records residence time. Keeping only changes produces an embedded jump chain with different stationary frequencies. With target masses (0.6,0.3,0.1) and uniform proposals to the other states, the changed-state frequencies approach (0.4,0.4,0.2). The repair is to record the current state after the acceptance decision on every transition, not to record only inside the accept branch.

## 5. Why does an invariant target not guarantee that a short MCMC run is accurate?

Invariance says that applying the transition to the target distribution leaves it unchanged. It does not bound the time required to approach that distribution from an arbitrary start. Ergodicity conditions and sufficiently effective finite-run exploration matter. A small-step chain can remain locally confined; a multimodal chain may never cross between modes. Inspect traces, dispersed initializations, function-specific diagnostics, and, where available, an independent reference.

## 6. Why are weight ESS and MCMC ESS not interchangeable?

Weight ESS is computed from concentration of normalized importance or particle weights. MCMC ESS is derived from dependence of a chosen observable across correlated draws and chains. Uniform weights do not imply independent MCMC information, and high weight ESS cannot detect unvisited regions. MCMC ESS and MCSE also depend on which function is being estimated; first moments, second moments and tail events can have different numerical accuracy.

## 7. Why do bootstrap particle-filter weights simplify to observation likelihoods after resampling?

In the SIS ratio, the bootstrap proposal is the state transition, so transition density cancels against proposal density. The remaining incremental factor is the new observation likelihood. General SIS still multiplies this increment by the previous weight. Immediately after resampling, previous weights are equal, so the new relative weights are proportional only to the likelihood. The initial prior-proposal step has the same cancellation logic for the first observation.

## 8. Why can resampling increase weight ESS while reducing ancestor diversity?

Resampling copies ancestors and assigns equal weights to the new slots. Equal normalized weights force the weight-ESS formula to N. Several slots may nevertheless be copies of one ancestor, so the number of distinct initial ancestors can fall. No new observation was introduced. Later propagation may separate positions but does not restore lost initial identities; this distinction is especially consequential for path estimation and smoothing.

## 9. Why should particle-filter error against the Kalman mean be separated from error against latent truth?

The Kalman mean is the exact same-model, same-observation filtering expectation in this linear-Gaussian benchmark. Error against it isolates particle approximation error. Error against the latent truth also includes uncertainty remaining after noisy observations and can persist even for exact inference. The shaded Kalman interval is a pointwise posterior state interval, not MCSE of the particle mean or a simultaneous interval over every time.

## 10. How can a near-one R-hat coexist with a badly wrong posterior mean?

If all chains remain in the same mode, their within- and between-chain behavior can look similar while they all miss another region. The actual figure gives rank R-hat about 1.006 with estimated mean about 3.006, although the symmetric target mean is zero. Starting across both modes can reveal disagreement, but a nearly correct pooled mean may still arise by symmetry without mixing. Check actual movement, multiple observables, initialization sensitivity and appropriate reference calculations.
